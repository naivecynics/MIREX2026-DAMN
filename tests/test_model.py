from __future__ import annotations

import unittest

import torch

from training.config import TrainConfig as Config
from src.model import NestedMusicTransformer, ScoreTransformer


def tiny_config() -> Config:
    return Config(
        max_seq_len=8,
        score_hidden_size=16,
        score_intermediate_size=32,
        score_num_layers=1,
        score_num_heads=4,
        note_intermediate_size=32,
        note_num_layers=1,
        note_num_heads=4,
    )


class AbsolutePositionTest(unittest.TestCase):
    def test_default_medium_model_configuration_and_parameter_count(self) -> None:
        config = Config()
        self.assertEqual(
            (
                config.score_hidden_size,
                config.score_intermediate_size,
                config.score_num_layers,
                config.score_num_heads,
                config.note_intermediate_size,
                config.note_num_layers,
                config.note_num_heads,
            ),
            (768, 3072, 14, 12, 1536, 2, 12),
        )
        model = NestedMusicTransformer(config)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            144_243_585,
        )
        self.assertEqual(
            model.score_transformer.transformer.config.head_dim,
            64,
        )
        self.assertEqual(
            model.note_transformer.transformer.config.head_dim,
            64,
        )

    def test_tiny_model_preset_and_parameter_count(self) -> None:
        config = Config.from_model_size("tiny")
        self.assertEqual(
            (
                config.score_hidden_size,
                config.score_intermediate_size,
                config.score_num_layers,
                config.score_num_heads,
                config.note_intermediate_size,
                config.note_num_layers,
                config.note_num_heads,
            ),
            (384, 1536, 8, 6, 768, 1, 6),
        )
        model = NestedMusicTransformer(config)
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            20_506_369,
        )
        self.assertEqual(
            model.score_transformer.transformer.config.head_dim,
            64,
        )
        self.assertEqual(
            model.note_transformer.transformer.config.head_dim,
            64,
        )

    def test_eos_expands_only_the_start_vocabulary(self) -> None:
        model = NestedMusicTransformer(tiny_config())
        score = model.score_transformer
        self.assertEqual(score.start_embedding.num_embeddings, 129)
        self.assertEqual(score.pitch_embedding.num_embeddings, 128)
        self.assertEqual(score.duration_embedding.num_embeddings, 128)
        self.assertIs(
            model.note_transformer.start_head.weight,
            score.start_embedding.weight,
        )

    def test_shifted_positions_share_onset_and_reserve_bos(self) -> None:
        absolute_starts = torch.tensor([[16, 16, 20, 20]])
        self.assertEqual(
            ScoreTransformer.shifted_position_ids(absolute_starts).tolist(),
            [[0, 17, 17, 21]],
        )

    def test_index_positions_use_note_order(self) -> None:
        config = tiny_config()
        config.score_position_mode = "index"
        score = ScoreTransformer(config)
        absolute_starts = torch.tensor([[16, 16, 20, 20]])
        self.assertEqual(
            score.position_ids(absolute_starts).tolist(),
            [[0, 1, 2, 3]],
        )

    def test_rejects_unknown_score_position_mode(self) -> None:
        config = tiny_config()
        config.score_position_mode = "unknown"
        with self.assertRaisesRegex(ValueError, "score_position_mode"):
            ScoreTransformer(config)

    def test_current_target_content_and_position_do_not_leak(self) -> None:
        torch.manual_seed(7)
        model = NestedMusicTransformer(tiny_config()).eval()
        notes = torch.tensor([[[16, 60, 4], [0, 64, 4], [4, 67, 2]]])
        absolute_starts = torch.tensor([[16, 16, 20]])
        changed = notes.clone()
        changed[:, 2] = torch.tensor([127, 12, 127])
        changed_starts = absolute_starts.clone()
        changed_starts[:, 2] = 999

        with torch.no_grad():
            original_logits = model(
                notes,
                absolute_starts=absolute_starts,
            )[0]
            changed_logits = model(
                changed,
                absolute_starts=changed_starts,
            )[0]

        self.assertTrue(
            torch.equal(original_logits[:, 2], changed_logits[:, 2])
        )
        self.assertEqual(
            model.score_transformer.transformer.config.max_position_embeddings,
            4096,
        )


if __name__ == "__main__":
    unittest.main()
