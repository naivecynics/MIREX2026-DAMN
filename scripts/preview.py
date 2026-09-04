#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mido


TPQ = 480
TICKS_PER_STEP = TPQ // 4


def load_notes(path: Path, key: str) -> list[dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
        raise ValueError(f"{path}: expected a {key!r} list")

    notes: list[dict[str, int]] = []
    for index, raw in enumerate(payload[key]):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: note {index} must be an object")
        values = [raw.get(name) for name in ("start", "pitch", "duration")]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError(f"{path}: note {index} fields must be integers")
        start, pitch, duration = values
        if start < 0 or not 0 <= pitch <= 127 or duration < 1:
            raise ValueError(f"{path}: invalid note {index}: {raw}")
        notes.append({"start": start, "pitch": pitch, "duration": duration})
    return notes


def midi_channel(index: int) -> int:
    channel = index % 15
    return channel if channel < 9 else channel + 1


def note_track(
    name: str, notes: list[dict[str, int]], *, channel: int, include_meta: bool
) -> mido.MidiTrack:
    track = mido.MidiTrack()
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
    track.append(mido.Message("program_change", program=0, channel=channel, time=0))

    events: list[tuple[int, int, int, mido.Message]] = []
    for note in notes:
        start_tick = note["start"] * TICKS_PER_STEP
        end_tick = (note["start"] + note["duration"]) * TICKS_PER_STEP
        events.append(
            (
                start_tick,
                1,
                note["pitch"],
                mido.Message(
                    "note_on", note=note["pitch"], velocity=64, channel=channel
                ),
            )
        )
        events.append(
            (
                end_tick,
                0,
                note["pitch"],
                mido.Message(
                    "note_off", note=note["pitch"], velocity=0, channel=channel
                ),
            )
        )

    previous_tick = 0
    for tick, _, _, message in sorted(events, key=lambda item: item[:3]):
        message.time = tick - previous_tick
        track.append(message)
        previous_tick = tick
    return track


def build_preview(prompt: Path, generated: list[Path], output: Path) -> None:
    tracks = [("Prompt", load_notes(prompt, "prompt"))]
    tracks.extend((path.stem, load_notes(path, "generation")) for path in generated)

    midi = mido.MidiFile(type=1, ticks_per_beat=TPQ)
    for index, (name, notes) in enumerate(tracks):
        midi.tracks.append(
            note_track(
                name,
                notes,
                channel=midi_channel(index),
                include_meta=index == 0,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a multitrack MIDI from a prompt and generations."
    )
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--generated", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_preview(args.prompt, args.generated, args.output)
    print(f"wrote {args.output} ({len(args.generated) + 1} tracks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
