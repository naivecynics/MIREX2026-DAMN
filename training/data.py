from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .config import TrainConfig as Config


class ParquetMusicDataset(IterableDataset):
    def __init__(
        self,
        config: Config,
        split: str,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _all_files(self) -> list[Path]:
        files = sorted(self.config.parquet_data_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"no Parquet shards found in {self.config.parquet_data_dir}"
            )
        return files

    def has_samples(self) -> bool:
        for path in self._all_files():
            table = pq.read_table(
                path,
                columns=["dataset_split"],
                filters=[("dataset_split", "=", self.split)],
            )
            if table.num_rows:
                return True
        return False

    def _files(self) -> list[Path]:
        files = self._all_files()
        if self.shuffle:
            random.Random(self.config.seed + self.epoch).shuffle(files)

        worker = get_worker_info()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]
        return files

    def _validate(
        self,
        slice_id: str,
        notes: list[list[int]],
        absolute_starts: list[int],
    ) -> None:
        if len(absolute_starts) != len(notes):
            raise ValueError(
                f"{slice_id}: absolute_starts and notes must have equal length"
            )
        vocab_sizes = (
            self.config.eos_token_id or self.config.start_vocab_size,
            self.config.pitch_vocab_size,
            self.config.duration_vocab_size,
        )
        for note_index, note in enumerate(notes):
            if len(note) != 3:
                raise ValueError(
                    f"{slice_id}: note {note_index} must contain S/P/D"
                )
            for field, value, vocab_size in zip("SPD", note, vocab_sizes):
                if not 0 <= value < vocab_size:
                    raise ValueError(
                        f"{slice_id}: {field}={value} at note {note_index} "
                        f"is outside [0, {vocab_size - 1}]"
                    )
            absolute_start = absolute_starts[note_index]
            if absolute_start < 0:
                raise ValueError(
                    f"{slice_id}: absolute start at note {note_index} "
                    "must be non-negative"
                )
            if note_index:
                previous_start = absolute_starts[note_index - 1]
                previous_pitch = notes[note_index - 1][1]
                if absolute_start < previous_start:
                    raise ValueError(
                        f"{slice_id}: absolute starts must be non-decreasing"
                    )
                if absolute_start == previous_start and note[1] <= previous_pitch:
                    raise ValueError(
                        f"{slice_id}: pitches within an onset must be "
                        "strictly increasing"
                    )

    def __iter__(self) -> Iterator[dict[str, object]]:
        rng = random.Random(self.config.seed + self.epoch)
        for path in self._files():
            table = pq.read_table(
                path,
                columns=[
                    "slice_id",
                    "dataset_split",
                    "notes",
                    "absolute_starts",
                    "barlines",
                ],
                filters=[("dataset_split", "=", self.split)],
            )
            rows = table.to_pylist()
            if self.shuffle:
                rng.shuffle(rows)

            for row in rows:
                slice_id = str(row["slice_id"])
                notes = row["notes"]
                absolute_starts = row["absolute_starts"]
                self._validate(slice_id, notes, absolute_starts)
                eos_token_id = self.config.eos_token_id
                if eos_token_id is not None:
                    if self.config.max_seq_len < 2:
                        raise ValueError(
                            "max_seq_len must allow at least one note and EOS"
                        )
                    real_note_limit = self.config.max_seq_len - 1
                    if len(notes) > real_note_limit:
                        raise ValueError(
                            f"{slice_id}: {len(notes)} real notes exceed the "
                            f"EOS-aware limit of {real_note_limit}; rebuild "
                            "Step 3 Parquet shards"
                        )
                    barlines = row["barlines"]
                    if not isinstance(barlines, list) or not barlines:
                        raise ValueError(f"{slice_id}: barlines must be non-empty")
                    slice_end = int(barlines[-1])
                    if absolute_starts and slice_end <= absolute_starts[-1]:
                        raise ValueError(
                            f"{slice_id}: final barline must follow the last note"
                        )
                    notes = [*notes, [eos_token_id, 0, 0]]
                    absolute_starts = [*absolute_starts, slice_end]
                    yield {
                        "slice_id": slice_id,
                        "notes": torch.tensor(notes, dtype=torch.long),
                        "absolute_starts": torch.tensor(
                            absolute_starts,
                            dtype=torch.long,
                        ),
                    }
                    continue

                for offset in range(0, len(notes), self.config.max_seq_len):
                    chunk = notes[offset : offset + self.config.max_seq_len]
                    if chunk:
                        yield {
                            "slice_id": slice_id,
                            "notes": torch.tensor(chunk, dtype=torch.long),
                            "absolute_starts": torch.tensor(
                                absolute_starts[
                                    offset : offset + self.config.max_seq_len
                                ],
                                dtype=torch.long,
                            ),
                        }


def collate_notes(batch: list[dict[str, object]]) -> dict[str, object]:
    max_length = max(item["notes"].shape[0] for item in batch)
    notes = torch.zeros(len(batch), max_length, 3, dtype=torch.long)
    absolute_starts = torch.zeros(len(batch), max_length, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_length, dtype=torch.long)

    for index, item in enumerate(batch):
        item_notes = item["notes"]
        length = item_notes.shape[0]
        notes[index, :length] = item_notes
        absolute_starts[index, :length] = item["absolute_starts"]
        attention_mask[index, :length] = 1

    return {
        "notes": notes,
        "absolute_starts": absolute_starts,
        "attention_mask": attention_mask,
    }
