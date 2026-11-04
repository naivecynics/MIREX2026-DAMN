import json
import math
from pathlib import Path

import mido
import numpy as np
import pytest

from evaluation.common import atomic_json, output_paths
from evaluation.score_clamp3 import (
    aggregate_prompt_scores, invalid_features, paired_comparisons,
    parse_candidates, prepare_inputs, score_features, summarize,
)


def _write_benchmark(root: Path, cids: tuple[str, ...] = ("piece-a", "piece-b")) -> None:
    samples = []
    for index, cid in enumerate(cids):
        prompt = root / "prompts" / f"{cid}.json"
        reference = root / "references" / f"{cid}.json"
        atomic_json(prompt, {"prompt": [{"start": 16, "pitch": 60 + index, "duration": 4}]})
        atomic_json(reference, {"generation": [{"start": 80, "pitch": 64 + index, "duration": 8}]})
        samples.append(
            {
                "artifacts": {
                    "prompt_json": str(prompt.relative_to(root)),
                    "reference_json": str(reference.relative_to(root)),
                },
                "cid": cid,
                "genre": "test",
                "title": f"Piece {index}",
            }
        )
    atomic_json(root / "manifest.json", {"samples": samples})


def _write_outputs(root: Path, cids: tuple[str, ...], pitch: int) -> None:
    for index, cid in enumerate(cids):
        output, _ = output_paths(root, cid)
        atomic_json(output, {"generation": [{"start": 80, "pitch": pitch + index, "duration": 4}]})


def _write_sample_outputs(
    root: Path, cids: tuple[str, ...], pitch: int, n_samples: int
) -> None:
    for cid_index, cid in enumerate(cids):
        for sample_index in range(1, n_samples + 1):
            output, _ = output_paths(root, cid, sample_index)
            atomic_json(
                output,
                {
                    "generation": [
                        {
                            "start": 80,
                            "pitch": pitch + cid_index + sample_index - 1,
                            "duration": 4,
                        }
                    ]
                },
            )


def test_clamp3_inputs_are_canonical_continuations_and_scores_are_stable(
    tmp_path: Path,
) -> None:
    cids = ("piece-a", "piece-b")
    test_root = tmp_path / "test"
    candidate_root = tmp_path / "candidate"
    midi_root = tmp_path / "midi"
    feature_root = tmp_path / "features"
    _write_benchmark(test_root, cids)
    _write_outputs(candidate_root, cids, 70)

    changed, actual_cids = prepare_inputs(
        test_root, {"model": candidate_root}, midi_root
    )
    assert actual_cids == list(cids)
    assert len(changed) == 4
    assert prepare_inputs(test_root, {"model": candidate_root}, midi_root)[0] == []
    midi = mido.MidiFile(midi_root / "model" / "piece-a.mid")
    absolute = 0
    starts = []
    for message in mido.merge_tracks(midi.tracks):
        absolute += message.time
        if message.type == "note_on" and message.velocity > 0:
            starts.append(absolute)
    assert starts == [0]

    for cid in cids:
        (feature_root / "reference").mkdir(parents=True, exist_ok=True)
        (feature_root / "model").mkdir(parents=True, exist_ok=True)
        np.save(feature_root / "reference" / f"{cid}.npy", [[1.0, 0.0]])
        vector = [[1.0, 0.0]] if cid == "piece-a" else [[0.0, 1.0]]
        np.save(feature_root / "model" / f"{cid}.npy", vector)

    rows = score_features(feature_root, {"model": candidate_root}, cids)
    model_rows = [row for row in rows if row["source"] == "model"]
    assert [row["cosine_similarity"] for row in model_rows] == [1.0, 0.0]
    reference_rows = [row for row in rows if row["source"] == "reference"]
    assert all(row["cosine_distance"] == 0.0 for row in reference_rows)
    assert invalid_features(feature_root, ["model", "reference"], cids) == []
    (feature_root / "model" / "piece-a.npy").write_bytes(b"broken")
    assert invalid_features(feature_root, ["model", "reference"], cids) == [
        Path("model/piece-a.mid")
    ]
    summary = summarize(rows, seed=7, bootstrap_samples=100)
    model_summary = next(row for row in summary if row["source"] == "model")
    assert model_summary["mean_similarity"] == pytest.approx(0.5)
    assert model_summary["sem_similarity"] == pytest.approx(0.5)


def test_clamp3_candidate_parser_rejects_ambiguous_names(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    assert parse_candidates([f"v1={root}"]) == {"v1": root.resolve()}
    with pytest.raises(ValueError):
        parse_candidates([f"reference={root}"])
    with pytest.raises(ValueError):
        parse_candidates([f"bad/name={root}"])


def test_clamp3_multi_sample_scores_are_averaged_within_prompt_first(
    tmp_path: Path,
) -> None:
    cids = ("piece-a", "piece-b")
    n_samples = 3
    test_root = tmp_path / "test"
    candidate_root = tmp_path / "candidate"
    midi_root = tmp_path / "midi"
    feature_root = tmp_path / "features"
    _write_benchmark(test_root, cids)
    _write_sample_outputs(candidate_root, cids, 70, n_samples)

    changed, actual_cids = prepare_inputs(
        test_root,
        {"model": candidate_root},
        midi_root,
        n_samples=n_samples,
    )
    assert actual_cids == list(cids)
    assert len(changed) == len(cids) * n_samples + len(cids)
    assert (midi_root / "model" / "piece-a" / "sample_03.mid").is_file()

    for cid in cids:
        (feature_root / "reference").mkdir(parents=True, exist_ok=True)
        (feature_root / "model" / cid).mkdir(parents=True, exist_ok=True)
        np.save(feature_root / "reference" / f"{cid}.npy", [[1.0, 0.0]])
        for sample_index, vector in enumerate(
            ([[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]]), start=1
        ):
            np.save(
                feature_root
                / "model"
                / cid
                / f"sample_{sample_index:02d}.npy",
                vector,
            )

    rows = score_features(
        feature_root,
        {"model": candidate_root},
        cids,
        n_samples=n_samples,
    )
    model_rows = [row for row in rows if row["source"] == "model"]
    assert len(model_rows) == len(cids) * n_samples
    prompt_rows = aggregate_prompt_scores(rows)
    model_prompts = [row for row in prompt_rows if row["source"] == "model"]
    assert len(model_prompts) == len(cids)
    assert all(row["sample_count"] == n_samples for row in model_prompts)
    assert all(
        row["cosine_similarity"] == pytest.approx(2 / 3)
        for row in model_prompts
    )
    summary = summarize(prompt_rows, seed=7, bootstrap_samples=100)
    model_summary = next(row for row in summary if row["source"] == "model")
    assert model_summary["count"] == len(cids)
    assert model_summary["mean_similarity"] == pytest.approx(2 / 3)
    assert model_summary["mean_within_prompt_std_similarity"] == pytest.approx(
        1 / math.sqrt(3)
    )


def test_clamp3_paired_comparison_preserves_piece_pairing() -> None:
    rows = [
        {"cid": "a", "source": "new", "cosine_similarity": 0.9},
        {"cid": "a", "source": "old", "cosine_similarity": 0.8},
        {"cid": "b", "source": "new", "cosine_similarity": 0.2},
        {"cid": "b", "source": "old", "cosine_similarity": 0.4},
    ]
    paired = paired_comparisons(
        rows, ["new", "old"], seed=3, bootstrap_samples=100
    )
    assert len(paired) == 1
    assert paired[0]["mean_delta_similarity_a_minus_b"] == pytest.approx(-0.05)
    assert paired[0]["win_rate_a"] == pytest.approx(0.5)



