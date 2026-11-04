from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .common import (
    PIECE_END,
    PROMPT_END,
    atomic_json,
    benchmark_items,
    file_sha256,
    generation_notes,
    output_paths,
    render_midi,
    write_csv,
)
from .encoders.clamp3 import Clamp3Encoder
from .metrics import aggregate_prompt_scores, summarize, paired_comparisons


PIECE_FIELDS = (
    "cid",
    "source",
    "sample_index",
    "cosine_similarity",
    "cosine_distance",
)
PROMPT_FIELDS = (
    "cid",
    "source",
    "sample_count",
    "cosine_similarity",
    "cosine_distance",
    "within_prompt_std_similarity",
)
SUMMARY_FIELDS = (
    "source",
    "count",
    "mean_similarity",
    "std_similarity",
    "sem_similarity",
    "median_similarity",
    "mean_distance",
    "mean_within_prompt_std_similarity",
    "bootstrap_ci95_low",
    "bootstrap_ci95_high",
    "bootstrap_distance_ci95_low",
    "bootstrap_distance_ci95_high",
)
PAIRED_FIELDS = (
    "source_a",
    "source_b",
    "count",
    "mean_delta_similarity_a_minus_b",
    "bootstrap_ci95_low",
    "bootstrap_ci95_high",
    "win_rate_a",
    "tie_rate",
)
_SOURCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def parse_candidates(values: Sequence[str]) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not _SOURCE_RE.fullmatch(name):
            raise ValueError(
                f"invalid candidate {value!r}; expected NAME=/path/to/output"
            )
        if name == "reference" or name in candidates:
            raise ValueError(f"reserved or duplicate candidate name: {name}")
        root = Path(raw_path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        candidates[name] = root
    if not candidates:
        raise ValueError("at least one --candidate is required")
    return candidates


def _continuation_local(notes: list[dict[str, int]]) -> list[dict[str, int]]:
    continuation_length = PIECE_END - PROMPT_END
    return [
        {
            "start": note["start"] - PROMPT_END,
            "pitch": note["pitch"],
            "duration": min(
                note["duration"],
                continuation_length - (note["start"] - PROMPT_END),
            ),
        }
        for note in notes
    ]


def _render_atomic(notes: list[dict[str, int]], path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(raw_temporary)
    try:
        render_midi(notes, temporary)
        if path.is_file() and file_sha256(path) == file_sha256(temporary):
            return False
        os.replace(temporary, path)
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def _sample_relative(
    source: str,
    cid: str,
    sample_index: int,
    n_samples: int,
    suffix: str,
) -> Path:
    if n_samples == 1:
        return Path(source) / f"{cid}{suffix}"
    return Path(source) / cid / f"sample_{sample_index:02d}{suffix}"


def prepare_inputs(
    test_root: Path,
    candidates: dict[str, Path],
    midi_root: Path,
    *,
    limit: int | None = None,
    n_samples: int = 1,
) -> tuple[list[Path], list[str]]:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    changed: list[Path] = []
    cids: list[str] = []
    items = benchmark_items(test_root)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        items = items[:limit]
    for item in items:
        cids.append(item.cid)
        for source, root in candidates.items():
            for sample_index in range(1, n_samples + 1):
                notes = generation_notes(
                    output_paths(root, item.cid, sample_index)[0]
                )
                relative = _sample_relative(
                    source, item.cid, sample_index, n_samples, ".mid"
                )
                if _render_atomic(
                    _continuation_local(notes), midi_root / relative
                ):
                    changed.append(relative)
        notes = generation_notes(item.reference_path)
        relative = Path("reference") / f"{item.cid}.mid"
        if _render_atomic(
            _continuation_local(notes), midi_root / relative
        ):
            changed.append(relative)
    return changed, cids


def _feature(path: Path) -> np.ndarray:
    feature = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if feature.size == 0 or not np.isfinite(feature).all():
        raise ValueError(f"invalid CLaMP 3 feature: {path}")
    norm = np.linalg.norm(feature)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"zero-norm CLaMP 3 feature: {path}")
    return feature / norm


def score_features(
    feature_root: Path,
    candidates: dict[str, Path],
    cids: Sequence[str],
    *,
    n_samples: int = 1,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cid in cids:
        reference = _feature(feature_root / "reference" / f"{cid}.npy")
        for source in candidates:
            for sample_index in range(1, n_samples + 1):
                relative = _sample_relative(
                    source, cid, sample_index, n_samples, ".npy"
                )
                candidate = _feature(feature_root / relative)
                if candidate.shape != reference.shape:
                    raise ValueError(
                        f"feature shape mismatch for {source}/{cid}: "
                        f"{candidate.shape} != {reference.shape}"
                    )
                similarity = float(np.clip(candidate @ reference, -1.0, 1.0))
                rows.append(
                    {
                        "cid": cid,
                        "source": source,
                        "sample_index": sample_index,
                        "cosine_similarity": similarity,
                        "cosine_distance": 1.0 - similarity,
                    }
                )
        rows.append(
            {
                "cid": cid,
                "source": "reference",
                "sample_index": 1,
                "cosine_similarity": 1.0,
                "cosine_distance": 0.0,
            }
        )
    return rows


def invalid_features(
    feature_root: Path,
    sources: Sequence[str],
    cids: Sequence[str],
    *,
    n_samples: int = 1,
) -> list[Path]:
    invalid: list[Path] = []
    for source in sources:
        for cid in cids:
            sample_count = 1 if source == "reference" else n_samples
            for sample_index in range(1, sample_count + 1):
                if source == "reference":
                    relative = Path(source) / f"{cid}.npy"
                else:
                    relative = _sample_relative(
                        source, cid, sample_index, n_samples, ".npy"
                    )
                try:
                    _feature(feature_root / relative)
                except (OSError, ValueError):
                    invalid.append(relative.with_suffix(".mid"))
    return invalid


def _load_run_config(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def run(args: argparse.Namespace) -> Path:
    if args.n_samples < 1:
        raise ValueError("n_samples must be positive")
    test_root = args.test_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    candidates = parse_candidates(args.candidate)
    encoder = Clamp3Encoder(args.clamp3_root, args.clamp3_python)
    encoder_metadata = encoder.metadata()
    run_config = {
        "bootstrap_samples": args.bootstrap_samples,
        "candidates": {name: str(path) for name, path in candidates.items()},
        "encoder": encoder_metadata,
        "limit": args.limit,
        "n_samples": args.n_samples,
        "representation": "canonical_continuation_midi_rebased_to_zero",
        "seed": args.seed,
        "test_root": str(test_root),
    }
    config_path = output_root / "run_config.json"
    previous = _load_run_config(config_path)
    if previous is not None and previous != run_config:
        raise ValueError(
            f"existing run has a different configuration: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(config_path, run_config)

    midi_root = output_root / "inputs"
    feature_root = output_root / "features"
    changed, cids = prepare_inputs(
        test_root,
        candidates,
        midi_root,
        limit=args.limit,
        n_samples=args.n_samples,
    )
    stale = sorted(
        set(changed)
        | set(
            invalid_features(
                feature_root,
                [*candidates, "reference"],
                cids,
                n_samples=args.n_samples,
            )
        )
    )
    if stale:
        encoder.extract(midi_root, feature_root, changed=stale)

    rows = score_features(
        feature_root, candidates, cids, n_samples=args.n_samples
    )
    prompt_rows = aggregate_prompt_scores(rows)
    summary = summarize(
        prompt_rows, seed=args.seed, bootstrap_samples=args.bootstrap_samples
    )
    paired = paired_comparisons(
        prompt_rows,
        list(candidates),
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    write_csv(output_root / "piece_metrics.csv", rows, list(PIECE_FIELDS))
    write_csv(
        output_root / "prompt_metrics.csv",
        prompt_rows,
        list(PROMPT_FIELDS),
    )
    write_csv(output_root / "summary.csv", summary, list(SUMMARY_FIELDS))
    write_csv(output_root / "paired_comparisons.csv", paired, list(PAIRED_FIELDS))
    atomic_json(
        output_root / "summary.json",
        {
            "distance": "1 - cosine_similarity",
            "higher_is_better": True,
            "metric": "cosine_similarity",
            "rows": summary,
        },
    )
    atomic_json(
        output_root / "paired_comparisons.json",
        {
            "delta": "cosine_similarity(source_a) - cosine_similarity(source_b)",
            "higher_is_better": True,
            "rows": paired,
        },
    )
    atomic_json(
        output_root / "run_manifest.json",
        {
            "candidate_count": len(candidates),
            "encoder": encoder_metadata,
            "feature_count": len(list(feature_root.rglob("*.npy"))),
            "piece_count": len(cids),
            "sample_count_per_candidate": args.n_samples,
            "piece_metrics_sha256": file_sha256(output_root / "piece_metrics.csv"),
            "prompt_metrics_sha256": file_sha256(
                output_root / "prompt_metrics.csv"
            ),
            "paired_comparisons_sha256": file_sha256(
                output_root / "paired_comparisons.csv"
            ),
            "run_config": run_config,
            "summary_sha256": file_sha256(output_root / "summary.csv"),
        },
    )
    return output_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score generated continuations against references with CLaMP 3 C2."
    )
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help="candidate output root; repeat for each model",
    )
    parser.add_argument("--clamp3-root", type=Path, required=True)
    parser.add_argument("--clamp3-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
