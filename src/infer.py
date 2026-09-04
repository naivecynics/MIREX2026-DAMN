from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path
from typing import Callable, ContextManager, Sequence

import torch
from accelerate import Accelerator

from .config import Config
from .model import NestedMusicTransformer


PROMPT_END = 80
PIECE_END = 272
TOKEN_MAX = 127


def _load_config(checkpoint: Path) -> Config:
    candidates = (checkpoint / "config.json", checkpoint.parent / "config.json")
    config_path = next((path for path in candidates if path.is_file()), None)
    if config_path is None:
        print(
            "warning: config.json not found beside checkpoint; "
            "using frozen DAMN defaults",
            file=sys.stderr,
        )
        return Config()

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    known = {field.name: field for field in fields(Config)}
    values = {key: value for key, value in raw.items() if key in known}
    for key, field in known.items():
        if key in values and field.type is Path:
            values[key] = Path(values[key]).expanduser()
    return Config(**values)


def _mixed_precision(config: Config) -> str:
    if (
        config.mixed_precision == "bf16"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return "bf16"
    return "no"


def load_model(
    checkpoint: Path,
) -> tuple[Accelerator, NestedMusicTransformer, Config]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint}")
    config = _load_config(checkpoint)
    accelerator = Accelerator(mixed_precision=_mixed_precision(config))
    model = NestedMusicTransformer(config)
    prepared = accelerator.prepare(model)
    accelerator.load_state(str(checkpoint))
    model = accelerator.unwrap_model(prepared)
    model.eval()
    return accelerator, model, config


def normalize_notes(notes: object, *, prompt: bool = False) -> list[dict[str, int]]:
    if not isinstance(notes, list):
        raise ValueError("notes must be a list")
    normalized: list[dict[str, int]] = []
    for index, note in enumerate(notes):
        if not isinstance(note, dict):
            raise ValueError(f"note {index} must be an object")
        try:
            values = (note["start"], note["pitch"], note["duration"])
        except KeyError as exc:
            raise ValueError(
                f"note {index} must contain integer start/pitch/duration"
            ) from exc
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in values
        ):
            raise ValueError(
                f"note {index} must contain integer start/pitch/duration"
            )
        start, pitch, duration = values
        if start < 0 or (prompt and start >= PROMPT_END):
            limit = f"[0, {PROMPT_END - 1}]" if prompt else "non-negative"
            raise ValueError(f"note {index} start must be {limit}")
        if not 0 <= pitch <= TOKEN_MAX:
            raise ValueError(f"note {index} pitch must be in [0, 127]")
        if duration < 1:
            raise ValueError(f"note {index} duration must be positive")
        normalized.append(
            {"start": start, "pitch": pitch, "duration": duration}
        )
    normalized = sorted(
        normalized,
        key=lambda note: (note["start"], note["pitch"], note["duration"]),
    )
    if not prompt:
        return normalized

    merged: dict[tuple[int, int], int] = {}
    for note in normalized:
        key = (note["start"], note["pitch"])
        merged[key] = max(note["duration"], merged.get(key, 0))
    return [
        {"start": start, "pitch": pitch, "duration": duration}
        for (start, pitch), duration in sorted(merged.items())
    ]


def absolute_to_tokens(notes: list[dict[str, int]]) -> torch.Tensor:
    tokens: list[list[int]] = []
    previous_start = 0
    for note in notes:
        start = note["start"]
        if start < previous_start:
            raise ValueError("notes must be sorted by non-decreasing start")
        tokens.append(
            [
                min(start - previous_start, TOKEN_MAX),
                note["pitch"],
                min(note["duration"], TOKEN_MAX),
            ]
        )
        previous_start = start
    return torch.tensor(tokens, dtype=torch.long).reshape(-1, 3)


def absolute_starts_tensor(notes: list[dict[str, int]]) -> torch.Tensor:
    return torch.tensor(
        [note["start"] for note in notes],
        dtype=torch.long,
    )


def sample_token(
    logits: torch.Tensor,
    *,
    generator: torch.Generator,
    temperature: float,
    top_p: float,
    top_k: int,
    min_token: int = 0,
    max_token: int | None = None,
) -> int:
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    scores = logits.float().clone()
    scores[:min_token] = -torch.inf
    if max_token is not None:
        scores[max_token + 1 :] = -torch.inf
    if not torch.isfinite(scores).any():
        raise RuntimeError("sampling constraints removed every token")
    if temperature == 0:
        return int(scores.argmax().item())
    scores /= temperature

    if top_k:
        keep = min(top_k, scores.numel())
        threshold = torch.topk(scores, keep).values[-1]
        scores[scores < threshold] = -torch.inf

    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        probabilities = torch.softmax(sorted_scores, dim=-1)
        remove = probabilities.cumsum(dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[sorted_indices[remove]] = -torch.inf

    probabilities = torch.softmax(scores, dim=-1)
    return int(
        torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).item()
    )


@torch.inference_mode()
def generate_continuation(
    model: NestedMusicTransformer,
    config: Config,
    prompt: list[dict[str, int]],
    *,
    generator: torch.Generator,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 0,
    max_notes: int = 512,
    max_notes_per_onset: int = 32,
    min_generation_end: int = 240,
    autocast_context: Callable[[], ContextManager[object]] = nullcontext,
) -> list[dict[str, int]]:
    if max_notes < 1:
        raise ValueError("max_notes must be positive")
    if max_notes_per_onset < 1:
        raise ValueError("max_notes_per_onset must be positive")
    prompt = normalize_notes(prompt, prompt=True)
    history = list(prompt)
    generated: list[dict[str, int]] = []
    generated_end = PROMPT_END
    last_start = history[-1]["start"] if history else 0
    current_onset_count = sum(
        note["start"] == last_start for note in history
    )

    for _ in range(max_notes):
        token_history = absolute_to_tokens(history).to(
            next(model.parameters()).device
        )
        position_history = absolute_starts_tensor(history).to(
            token_history.device
        )
        with autocast_context():
            score_hidden = model.next_score_hidden(
                token_history.unsqueeze(0),
                position_history.unsqueeze(0),
            )
            start_logits = model.note_transformer.next_logits(score_hidden)[0]

        min_delta = max(0, PROMPT_END - last_start)
        last_pitch = history[-1]["pitch"] if history else None
        if (
            current_onset_count >= max_notes_per_onset
            or last_pitch == TOKEN_MAX
        ):
            min_delta = max(min_delta, 1)
        start = sample_token(
            start_logits,
            generator=generator,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_token=min_delta,
        )
        eos = config.eos_token_id
        ends_early = eos is not None and start == eos
        if start <= TOKEN_MAX:
            ends_early = ends_early or last_start + start >= PIECE_END
        if generated_end < min_generation_end and ends_early:
            start = sample_token(
                start_logits,
                generator=generator,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_token=min_delta,
                max_token=min(TOKEN_MAX, PIECE_END - last_start - 1),
            )
        if eos is not None and start == eos:
            break
        next_start = last_start + start
        if next_start >= PIECE_END:
            break

        start_tensor = torch.tensor(
            [start],
            dtype=torch.long,
            device=score_hidden.device,
        )
        with autocast_context():
            pitch_logits = model.note_transformer.next_logits(
                score_hidden,
                start=start_tensor,
            )[0]
        pitch = sample_token(
            pitch_logits,
            generator=generator,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_token=(
                last_pitch + 1
                if start == 0 and last_pitch is not None
                else 0
            ),
        )

        pitch_tensor = torch.tensor(
            [pitch],
            dtype=torch.long,
            device=score_hidden.device,
        )
        with autocast_context():
            duration_logits = model.note_transformer.next_logits(
                score_hidden,
                start=start_tensor,
                pitch=pitch_tensor,
            )[0]
        duration = sample_token(
            duration_logits,
            generator=generator,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_token=1,
        )
        duration = min(duration, PIECE_END - next_start)

        note = {
            "start": next_start,
            "pitch": pitch,
            "duration": duration,
        }
        generated.append(note)
        generated_end = max(generated_end, next_start + duration)
        history.append(note)
        if next_start == last_start:
            current_onset_count += 1
        else:
            last_start = next_start
            current_onset_count = 1

    return generated


def _read_prompt(path: Path) -> list[dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "prompt" not in payload:
        raise ValueError("input JSON must contain a prompt field")
    return normalize_notes(payload["prompt"], prompt=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_inference(
    *,
    checkpoint: Path,
    input_path: Path,
    output_dir: Path,
    n_samples: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_notes: int,
    max_notes_per_onset: int,
    min_generation_end: int = 240,
) -> list[Path]:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    accelerator, model, config = load_model(checkpoint)
    prompt = _read_prompt(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index in range(n_samples):
        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(seed + index)
        generated = generate_continuation(
            model,
            config,
            prompt,
            generator=generator,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_notes=max_notes,
            max_notes_per_onset=max_notes_per_onset,
            min_generation_end=min_generation_end,
            autocast_context=accelerator.autocast,
        )
        output_path = output_dir / f"sample_{index + 1:02d}.json"
        write_json(output_path, {"generation": generated})
        outputs.append(output_path)
        print(f"[{index + 1}/{n_samples}] {output_path.name}", flush=True)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MIREX continuations.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-notes", type=int, default=512)
    parser.add_argument("--max-notes-per-onset", type=int, default=32)
    parser.add_argument("--min-generation-end", type=int, default=240)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_inference(
        checkpoint=args.checkpoint,
        input_path=args.input,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_notes=args.max_notes,
        max_notes_per_onset=args.max_notes_per_onset,
        min_generation_end=args.min_generation_end,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
