from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

from src.infer import generate_continuation, load_model

from .common import (
    atomic_json,
    benchmark_items,
    config_sha256,
    file_sha256,
    generation_notes,
    output_paths,
    prompt_notes,
    reusable_output,
    seed_for_cid,
    write_done,
)


def run(args: argparse.Namespace) -> int:
    if args.n_samples < 1:
        raise ValueError("n_samples must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    items = benchmark_items(args.test_root)
    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    items = items[args.offset :]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        items = items[: args.limit]
    accelerator, model, model_config = load_model(checkpoint)
    run_config: dict[str, object] = {
        "engine": "nested_music_transformer",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
        "model_config": {
            key: getattr(model_config, key)
            for key in model_config.__dataclass_fields__
        },
        "max_notes": args.max_notes,
        "min_generation_end": args.min_generation_end,
        "max_notes_per_onset": args.max_notes_per_onset,
        "sample_count": 1,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }
    if model_config.eos_token_id is not None:
        run_config["eos_token_id"] = model_config.eos_token_id
    config_hash = config_sha256(run_config)
    records: list[dict[str, object]] = []
    failures = 0
    manifest_path = output_root / (
        "run_manifest.json"
        if args.offset == 0
        else f"run_manifest.offset_{args.offset:04d}.json"
    )

    total_jobs = len(items) * args.n_samples
    for item_index, item in enumerate(items):
        input_hash = file_sha256(item.prompt_path)
        for sample_index in range(1, args.n_samples + 1):
            index = item_index * args.n_samples + sample_index
            output_path, done_path = output_paths(
                output_root, item.cid, sample_index
            )
            seed = seed_for_cid(args.seed, item.cid, sample_index)
            if reusable_output(
                output_path,
                done_path,
                input_sha256=input_hash,
                config_sha256=config_hash,
            ):
                status = "skipped"
            else:
                try:
                    prompt = prompt_notes(item.prompt_path)
                    generator = torch.Generator(device=accelerator.device)
                    generator.manual_seed(seed)
                    generated = generate_continuation(
                        model,
                        model_config,
                        prompt,
                        generator=generator,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_notes=args.max_notes,
                        min_generation_end=args.min_generation_end,
                        max_notes_per_onset=args.max_notes_per_onset,
                        autocast_context=accelerator.autocast,
                    )
                    atomic_json(output_path, {"generation": generated})
                    generation_notes(output_path)
                    write_done(
                        done_path,
                        output_path,
                        input_sha256=input_hash,
                        config_sha256=config_hash,
                        seed=seed,
                    )
                    status = "generated"
                except Exception as exc:  # keep a long job resumable
                    failures += 1
                    status = "failed"
                    print(
                        f"[{index:04d}/{total_jobs}] {item.cid} "
                        f"sample {sample_index:02d}: {exc}",
                        file=sys.stderr,
                    )
                    records.append(
                        {
                            "cid": item.cid,
                            "error": repr(exc),
                            "sample_index": sample_index,
                            "seed": seed,
                            "status": status,
                        }
                    )
                    atomic_json(
                        manifest_path,
                        {
                            "config": run_config,
                            "items": records,
                            "requested_sample_count": args.n_samples,
                        },
                    )
                    continue
            count = len(generation_notes(output_path))
            records.append(
                {
                    "cid": item.cid,
                    "note_count": count,
                    "sample_index": sample_index,
                    "seed": seed,
                    "status": status,
                }
            )
            print(
                f"[{index:04d}/{total_jobs}] {item.cid} "
                f"sample {sample_index:02d}: {status} ({count} notes)",
                flush=True,
            )
            atomic_json(
                manifest_path,
                {
                    "config": run_config,
                    "items": records,
                    "requested_sample_count": args.n_samples,
                },
            )

    atomic_json(
        manifest_path,
        {
            "config": run_config,
            "failures": failures,
            "items": records,
            "requested_sample_count": args.n_samples,
            "start_offset": args.offset,
            "total": total_jobs,
        },
    )
    return 1 if failures else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for our nested model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-notes", type=int, default=512)
    parser.add_argument("--min-generation-end", type=int, default=240)
    parser.add_argument("--max-notes-per-onset", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
