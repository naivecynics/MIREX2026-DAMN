from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from training.config import TrainConfig as Config
from training.data import ParquetMusicDataset, collate_notes
from training.train import (
    _build_scheduler,
    _distributed_scheduler_steps,
    _lr_multiplier,
    music_loss,
    parse_args,
    train,
)

SCHEMA = pa.schema([
    ("slice_id", pa.string()),
    ("dataset_split", pa.string()),
    ("notes", pa.list_(pa.list_(pa.int64()))),
    ("absolute_starts", pa.list_(pa.int64())),
    ("barlines", pa.list_(pa.int64())),
])


def parquet_row(
    slice_id: str,
    notes: list[list[int]],
    split: str = "train",
    absolute_starts: list[int] | None = None,
) -> dict[str, object]:
    if absolute_starts is None:
        current_start = 0
        absolute_starts = []
        for note in notes:
            current_start += note[0]
            absolute_starts.append(current_start)
    return {
        "slice_id": slice_id,
        "sample_id": slice_id,
        "dataset": "test",
        "source_partition": "test",
        "relative_path": f"{slice_id}.mid",
        "sha256": "ab" * 32,
        "window_index": 0,
        "dataset_split": split,
        "notes": notes,
        "absolute_starts": absolute_starts,
        "prompt_end": 80,
        "barlines": list(range(16, 273, 16)),
    }


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.parquets = self.root / "parquets"
        self.parquets.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(self, **values: object) -> Config:
        defaults = {
            "base_path": self.root,
            "data_dir": self.root,
            "parquet_data_dir": self.parquets,
            "logging_dir": self.root / "logs",
            "wandb_dir": self.root / "wandb",
            "checkpoint_dir": self.root / "checkpoints",
            "max_seq_len": 5,
            "num_workers": 0,
        }
        defaults.update(values)
        return Config(**defaults)

    def write(self, rows: list[dict[str, object]]) -> None:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=SCHEMA),
            self.parquets / "test.parquet",
        )

    def test_chunk_collate_and_split(self) -> None:
        self.write(
            [
                parquet_row(
                    "train",
                    [[1, 60, 2], [0, 64, 2], [3, 67, 1], [2, 69, 1]],
                ),
                parquet_row("validation", [[1, 61, 1]], "validation"),
            ]
        )
        chunks = list(ParquetMusicDataset(self.config(), "train"))
        self.assertEqual([len(item["notes"]) for item in chunks], [5])
        self.assertEqual(chunks[0]["notes"][-1].tolist(), [128, 0, 0])
        batch = collate_notes(chunks)
        self.assertEqual(tuple(batch["notes"].shape), (1, 5, 3))
        self.assertEqual(
            batch["absolute_starts"].tolist(),
            [[1, 1, 4, 6, 272]],
        )
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1, 1]])

    def test_512_real_notes_plus_eos_fit_length_513(self) -> None:
        absolute_starts = [16 + index // 2 for index in range(512)]
        notes = []
        previous_start = 0
        for index, start in enumerate(absolute_starts):
            notes.append([start - previous_start, 60 + index % 2, 1])
            previous_start = start
        self.write(
            [parquet_row("maximum", notes, absolute_starts=absolute_starts)]
        )
        item = next(
            iter(ParquetMusicDataset(self.config(max_seq_len=513), "train"))
        )
        self.assertEqual(tuple(item["notes"].shape), (513, 3))
        self.assertEqual(item["notes"][-1].tolist(), [128, 0, 0])

    def test_513_real_note_row_exceeds_model_capacity(self) -> None:
        notes = [[1, 60, 1] for _ in range(513)]
        self.write([parquet_row("too-long", notes)])
        with self.assertRaisesRegex(ValueError, "rebuild Step 3"):
            next(
                iter(
                    ParquetMusicDataset(
                        self.config(max_seq_len=513),
                        "train",
                    )
                )
            )

    def test_vocab_error_names_slice(self) -> None:
        self.write([parquet_row("bad-slice", [[128, 60, 1]])])
        with self.assertRaisesRegex(ValueError, "bad-slice.*S=128"):
            next(iter(ParquetMusicDataset(self.config(), "train")))

    def test_rejects_noncanonical_onset_pitch_order(self) -> None:
        self.write(
            [
                parquet_row(
                    "bad-order",
                    [[1, 64, 1], [0, 60, 1]],
                    absolute_starts=[1, 1],
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            next(iter(ParquetMusicDataset(self.config(), "train")))


class LossTest(unittest.TestCase):
    def test_padding_is_ignored_and_all_heads_receive_gradients(self) -> None:
        notes = torch.tensor([[[1, 60, 2], [0, 0, 0]]])
        mask = torch.tensor([[1, 0]])
        logits = (
            torch.randn(1, 2, 129, requires_grad=True),
            torch.randn(1, 2, 128, requires_grad=True),
            torch.randn(1, 2, 128, requires_grad=True),
        )
        loss, _ = music_loss(logits, notes, mask)
        changed = tuple(value.detach().clone() for value in logits)
        for value in changed:
            value[:, 1] = 1_000
        changed_loss, _ = music_loss(changed, notes, mask)
        self.assertTrue(torch.allclose(loss.detach(), changed_loss))
        loss.backward()
        self.assertTrue(all(value.grad is not None for value in logits))

    def test_eos_only_affects_start_loss(self) -> None:
        notes = torch.tensor([[[1, 60, 2], [128, 0, 0]]])
        mask = torch.ones(1, 2, dtype=torch.long)
        logits = (
            torch.randn(1, 2, 129, requires_grad=True),
            torch.randn(1, 2, 128, requires_grad=True),
            torch.randn(1, 2, 128, requires_grad=True),
        )
        loss, _ = music_loss(logits, notes, mask, eos_token_id=128)
        changed = tuple(value.detach().clone() for value in logits)
        changed[1][:, 1] = 1_000
        changed[2][:, 1] = -1_000
        changed_loss, _ = music_loss(
            changed,
            notes,
            mask,
            eos_token_id=128,
        )
        self.assertTrue(torch.allclose(loss.detach(), changed_loss))
        loss.backward()
        self.assertGreater(logits[0].grad[:, 1].abs().sum().item(), 0)
        self.assertEqual(logits[1].grad[:, 1].abs().sum().item(), 0)
        self.assertEqual(logits[2].grad[:, 1].abs().sum().item(), 0)


class TrainingCliTest(unittest.TestCase):
    def test_training_overrides(self) -> None:
        args = parse_args(
            [
                "--batch-size",
                "32",
                "--model-size",
                "tiny",
                "--score-position-mode",
                "index",
                "--gradient-accumulation-steps",
                "2",
                "--max-steps",
                "123",
                "--warmup-steps",
                "10",
                "--anneal-steps",
                "23",
                "--min-learning-rate",
                "0.00003",
                "--eval-steps",
                "20",
                "--save-steps",
                "50",
            ]
        )
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.model_size, "tiny")
        self.assertEqual(args.score_position_mode, "index")
        self.assertEqual(args.gradient_accumulation_steps, 2)
        self.assertEqual(args.max_steps, 123)
        self.assertEqual(args.warmup_steps, 10)
        self.assertEqual(args.anneal_steps, 23)
        self.assertEqual(args.min_learning_rate, 3e-5)
        self.assertEqual(args.eval_steps, 20)
        self.assertEqual(args.save_steps, 50)

    def test_non_positive_training_override_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--batch-size", "0"])
        with self.assertRaises(SystemExit):
            parse_args(["--gradient-accumulation-steps", "0"])
        with self.assertRaises(SystemExit):
            parse_args(["--warmup-steps", "-1"])
        with self.assertRaises(SystemExit):
            parse_args(["--min-learning-rate", "-1"])

    def test_distributed_scheduler_preserves_global_steps(self) -> None:
        config = Config(
            warmup_steps=1_000,
            anneal_steps=10_000,
            max_steps=99_684,
        )
        self.assertEqual(
            _distributed_scheduler_steps(config, 2),
            (2_000, 20_000, 199_368),
        )

    def test_warmup_constant_cosine_schedule_boundaries(self) -> None:
        values = {
            step: _lr_multiplier(
                step,
                warmup_steps=2_000,
                anneal_steps=39_682,
                total_steps=198_410,
                min_lr_ratio=0.1,
            )
            for step in (0, 2_000, 158_728, 198_410)
        }
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[2_000], 1.0)
        self.assertEqual(values[158_728], 1.0)
        self.assertAlmostEqual(values[198_410], 0.1)

    def test_scheduler_rejects_overlapping_phases(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=3e-4)
        config = Config(max_steps=10, warmup_steps=6, anneal_steps=5)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            _build_scheduler(optimizer, config, 1)


class TrainingSmokeTest(unittest.TestCase):
    def test_two_steps_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parquets = root / "parquets"
            parquets.mkdir()
            rows = [
                parquet_row(
                    f"sample-{index}",
                    [[1, 60 + index, 2], [0, 64 + index, 2]],
                )
                for index in range(2)
            ]
            pq.write_table(
                pa.Table.from_pylist(rows, schema=SCHEMA),
                parquets / "train.parquet",
            )
            config = Config(
                base_path=root,
                data_dir=root,
                parquet_data_dir=parquets,
                logging_dir=root / "logs",
                wandb_dir=root / "wandb",
                checkpoint_dir=root / "checkpoints",
                max_seq_len=8,
                score_hidden_size=16,
                score_intermediate_size=32,
                score_num_layers=1,
                score_num_heads=4,
                note_intermediate_size=32,
                note_num_layers=1,
                note_num_heads=4,
                batch_size=1,
                warmup_steps=0,
                anneal_steps=1,
                max_steps=2,
                num_workers=0,
                log_steps=1,
                eval_steps=1,
                save_steps=1,
                max_checkpoints=1,
                mixed_precision="no",
            )
            self.assertEqual(
                train(
                    config,
                    use_wandb=False,
                    run_name="smoke",
                ),
                2,
            )
            run_dir = config.checkpoint_dir / "smoke"
            checkpoint = run_dir / "step_00000002"
            self.assertTrue((checkpoint / "trainer_state.json").is_file())
            self.assertFalse((run_dir / "step_00000001").exists())

            config.max_steps = 3
            self.assertEqual(
                train(config, use_wandb=False, resume=checkpoint),
                3,
            )
            self.assertTrue((run_dir / "step_00000003").is_dir())


if __name__ == "__main__":
    unittest.main()
