from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mido


PROMPT_END = 80
PIECE_END = 272
TPQ = 480
TICKS_PER_STEP = TPQ // 4


@dataclass(frozen=True)
class BenchmarkItem:
    cid: str
    prompt_path: Path
    reference_path: Path
    metadata: dict[str, object]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("manifest artifact path must be a string")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact escapes benchmark root: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def benchmark_items(root: Path) -> list[BenchmarkItem]:
    root = root.expanduser().resolve()
    manifest = load_json(root / "manifest.json")
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("samples"), list
    ):
        raise ValueError("benchmark manifest must contain a samples list")
    items: list[BenchmarkItem] = []
    seen: set[str] = set()
    for raw in manifest["samples"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("cid"), str):
            raise ValueError("every benchmark sample must contain a string cid")
        cid = raw["cid"]
        if cid in seen or not cid or Path(cid).name != cid:
            raise ValueError(f"invalid or duplicate cid: {cid!r}")
        artifacts = raw.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"{cid}: missing artifacts")
        seen.add(cid)
        items.append(
            BenchmarkItem(
                cid=cid,
                prompt_path=_artifact(root, artifacts.get("prompt_json")),
                reference_path=_artifact(
                    root, artifacts.get("reference_json")
                ),
                metadata=raw,
            )
        )
    return sorted(items, key=lambda item: item.cid)


def notes_from_payload(
    payload: object,
    key: str,
    *,
    start_min: int,
    start_max: int,
) -> list[dict[str, int]]:
    if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
        raise ValueError(f"payload must contain a {key} list")
    notes: list[dict[str, int]] = []
    previous_start = -1
    for index, raw in enumerate(payload[key]):
        if not isinstance(raw, dict):
            raise ValueError(f"{key} note {index} must be an object")
        try:
            start, pitch, duration = (
                raw["start"],
                raw["pitch"],
                raw["duration"],
            )
        except KeyError as exc:
            raise ValueError(f"{key} note {index} is missing a field") from exc
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (start, pitch, duration)
        ):
            raise ValueError(f"{key} note {index} fields must be integers")
        if not start_min <= start <= start_max:
            raise ValueError(
                f"{key} note {index} start {start} is outside "
                f"[{start_min}, {start_max}]"
            )
        if not 0 <= pitch <= 127:
            raise ValueError(f"{key} note {index} pitch is outside [0, 127]")
        if duration < 1:
            raise ValueError(f"{key} note {index} duration must be positive")
        if start < previous_start:
            raise ValueError(f"{key} notes must have non-decreasing starts")
        previous_start = start
        notes.append(
            {"start": start, "pitch": pitch, "duration": duration}
        )
    return notes


def prompt_notes(path: Path) -> list[dict[str, int]]:
    return notes_from_payload(
        load_json(path), "prompt", start_min=0, start_max=PROMPT_END - 1
    )


def generation_notes(path: Path) -> list[dict[str, int]]:
    return notes_from_payload(
        load_json(path),
        "generation",
        start_min=PROMPT_END,
        start_max=PIECE_END - 1,
    )


def seed_for_cid(base_seed: int, cid: str, sample_index: int = 1) -> int:
    payload = f"{base_seed}:{cid}:{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**63
    )


def output_paths(root: Path, cid: str, sample_index: int = 1) -> tuple[Path, Path]:
    directory = root / cid
    return (
        directory / f"sample_{sample_index:02d}.json",
        directory / f"sample_{sample_index:02d}.done.json",
    )


def reusable_output(
    output_path: Path,
    done_path: Path,
    *,
    input_sha256: str,
    config_sha256: str,
) -> bool:
    try:
        done = load_json(done_path)
        generation_notes(output_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(done, dict)
        and done.get("input_sha256") == input_sha256
        and done.get("config_sha256") == config_sha256
        and done.get("output_sha256") == file_sha256(output_path)
    )


def write_done(
    done_path: Path,
    output_path: Path,
    *,
    input_sha256: str,
    config_sha256: str,
    seed: int,
) -> None:
    atomic_json(
        done_path,
        {
            "config_sha256": config_sha256,
            "input_sha256": input_sha256,
            "output_sha256": file_sha256(output_path),
            "seed": seed,
        },
    )


def config_sha256(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _note_track(
    notes: Iterable[dict[str, int]],
    *,
    channel: int = 0,
    name: str | None = None,
    include_meta: bool = False,
) -> mido.MidiTrack:
    track = mido.MidiTrack()
    if name is not None:
        track.append(mido.MetaMessage("track_name", name=name, time=0))
    if include_meta:
        track.append(
            mido.MetaMessage(
                "time_signature", numerator=4, denominator=4, time=0
            )
        )
        track.append(
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0)
        )
    track.append(
        mido.Message("program_change", program=0, channel=channel, time=0)
    )
    events: list[tuple[int, int, int, mido.Message]] = []
    for note in notes:
        start = note["start"]
        duration = min(note["duration"], PIECE_END - start)
        if duration < 1:
            continue
        start_tick = start * TICKS_PER_STEP
        end_tick = (start + duration) * TICKS_PER_STEP
        events.append(
            (
                start_tick,
                1,
                note["pitch"],
                mido.Message(
                    "note_on",
                    note=note["pitch"],
                    velocity=64,
                    channel=channel,
                ),
            )
        )
        events.append(
            (
                end_tick,
                0,
                note["pitch"],
                mido.Message(
                    "note_off",
                    note=note["pitch"],
                    velocity=0,
                    channel=channel,
                ),
            )
        )
    previous_tick = 0
    for tick, _, _, message in sorted(events, key=lambda event: event[:3]):
        message.time = tick - previous_tick
        track.append(message)
        previous_tick = tick
    return track


def render_midi(notes: Iterable[dict[str, int]], path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=TPQ)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120)))
    midi.tracks.append(meta)
    midi.tracks.append(_note_track(notes))
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def render_multitrack_midi(
    tracks: Iterable[tuple[str, Iterable[dict[str, int]]]], path: Path
) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=TPQ)
    for channel, (name, notes) in enumerate(tracks):
        midi.tracks.append(
            _note_track(
                notes,
                channel=channel,
                name=name,
                include_meta=channel == 0,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def note_metrics(notes: list[dict[str, int]]) -> dict[str, int | float]:
    if not notes:
        return {
            "note_count": 0,
            "onset_count": 0,
            "pitch_min": -1,
            "pitch_max": -1,
            "pitch_range": 0,
            "max_onset_polyphony": 0,
            "mean_duration": 0.0,
        }
    onsets: dict[int, int] = {}
    for note in notes:
        onsets[note["start"]] = onsets.get(note["start"], 0) + 1
    pitches = [note["pitch"] for note in notes]
    return {
        "note_count": len(notes),
        "onset_count": len(onsets),
        "pitch_min": min(pitches),
        "pitch_max": max(pitches),
        "pitch_range": max(pitches) - min(pitches),
        "max_onset_polyphony": max(onsets.values()),
        "mean_duration": sum(note["duration"] for note in notes) / len(notes),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def blind_order(
    cid: str,
    seed: int,
    sources: Iterable[str] | None = None,
) -> list[str]:
    values = list(sources or ("ours", "musecoco", "reference"))
    random.Random(seed_for_cid(seed, cid)).shuffle(values)
    return values
