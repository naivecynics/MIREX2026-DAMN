from __future__ import annotations

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3Model

from .config import Config


def _qwen_config(config: Config, *, note: bool = False) -> Qwen3Config:
    if note:
        hidden_size = config.score_hidden_size
        intermediate_size = config.note_intermediate_size
        num_layers = config.note_num_layers
        num_heads = config.note_num_heads
        max_position_embeddings = 3
    else:
        hidden_size = config.score_hidden_size
        intermediate_size = config.score_intermediate_size
        num_layers = config.score_num_layers
        num_heads = config.score_num_heads
        max_position_embeddings = config.score_max_position_embeddings

    return Qwen3Config(
        vocab_size=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=hidden_size // num_heads,
        max_position_embeddings=max_position_embeddings,
        use_cache=False,
    )


class ScoreTransformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.score_position_mode not in {"absolute", "index"}:
            raise ValueError(
                "score_position_mode must be 'absolute' or 'index'"
            )
        dim = config.score_hidden_size
        self.position_mode = config.score_position_mode
        self.transformer = Qwen3Model(_qwen_config(config))
        self.transformer.embed_tokens.requires_grad_(False)
        self.start_embedding = nn.Embedding(config.start_vocab_size, dim)
        self.pitch_embedding = nn.Embedding(config.pitch_vocab_size, dim)
        self.duration_embedding = nn.Embedding(config.duration_vocab_size, dim)
        self.bos = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.bos, std=0.02)

    @staticmethod
    def shifted_position_ids(
        absolute_starts: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = torch.zeros_like(absolute_starts)
        position_ids[:, 1:] = absolute_starts[:, :-1] + 1
        return position_ids

    @staticmethod
    def index_position_ids(reference: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            reference.shape[1],
            dtype=reference.dtype,
            device=reference.device,
        )
        return positions.unsqueeze(0).expand(reference.shape[0], -1)

    def position_ids(self, absolute_starts: torch.Tensor) -> torch.Tensor:
        if self.position_mode == "absolute":
            return self.shifted_position_ids(absolute_starts)
        return self.index_position_ids(absolute_starts)

    def forward(
        self,
        notes: torch.Tensor,
        absolute_starts: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if absolute_starts.shape != notes.shape[:2]:
            raise ValueError(
                "absolute_starts must have the same batch and sequence "
                "dimensions as notes"
            )
        note_embeddings = (
            self.start_embedding(notes[..., 0])
            + self.pitch_embedding(notes[..., 1])
            + self.duration_embedding(notes[..., 2])
        )
        bos = self.bos.expand(notes.shape[0], -1, -1)
        shifted_embeddings = torch.cat((bos, note_embeddings[:, :-1]), dim=1)
        position_ids = self.position_ids(absolute_starts)
        return self.transformer(
            inputs_embeds=shifted_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state


class NoteTransformer(nn.Module):
    def __init__(
        self,
        config: Config,
        start_embedding: nn.Embedding,
        pitch_embedding: nn.Embedding,
        duration_embedding: nn.Embedding,
    ) -> None:
        super().__init__()
        dim = config.score_hidden_size
        self.transformer = Qwen3Model(_qwen_config(config, note=True))
        self.transformer.embed_tokens.requires_grad_(False)
        self.start_embedding = start_embedding
        self.pitch_embedding = pitch_embedding
        self.start_head = nn.Linear(dim, config.start_vocab_size)
        self.pitch_head = nn.Linear(dim, config.pitch_vocab_size)
        self.duration_head = nn.Linear(dim, config.duration_vocab_size)
        self.start_head.weight = start_embedding.weight
        self.pitch_head.weight = pitch_embedding.weight
        self.duration_head.weight = duration_embedding.weight

    def forward(
        self,
        score_hidden: torch.Tensor,
        notes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = score_hidden.shape
        local_inputs = torch.stack(
            (
                score_hidden,
                self.start_embedding(notes[..., 0]),
                self.pitch_embedding(notes[..., 1]),
            ),
            dim=2,
        ).reshape(batch_size * seq_len, 3, -1)

        hidden = self.transformer(
            inputs_embeds=local_inputs,
            use_cache=False,
        ).last_hidden_state
        hidden = hidden.reshape(batch_size, seq_len, 3, -1)

        return (
            self.start_head(hidden[:, :, 0]),
            self.pitch_head(hidden[:, :, 1]),
            self.duration_head(hidden[:, :, 2]),
        )

    def next_logits(
        self,
        score_hidden: torch.Tensor,
        *,
        start: torch.Tensor | None = None,
        pitch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pitch is not None and start is None:
            raise ValueError("pitch conditioning requires start")

        inputs = [score_hidden]
        if start is not None:
            inputs.append(self.start_embedding(start))
        if pitch is not None:
            inputs.append(self.pitch_embedding(pitch))
        local_inputs = torch.stack(inputs, dim=1)
        hidden = self.transformer(
            inputs_embeds=local_inputs,
            use_cache=False,
        ).last_hidden_state[:, -1]

        if start is None:
            return self.start_head(hidden)
        if pitch is None:
            return self.pitch_head(hidden)
        return self.duration_head(hidden)


class NestedMusicTransformer(nn.Module):
    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = config or Config()
        self.score_transformer = ScoreTransformer(self.config)
        score = self.score_transformer
        self.note_transformer = NoteTransformer(
            self.config,
            score.start_embedding,
            score.pitch_embedding,
            score.duration_embedding,
        )

    def forward(
        self,
        notes: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        absolute_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if absolute_starts is None:
            raise ValueError("absolute_starts is required")
        score_hidden = self.score_transformer(
            notes,
            absolute_starts,
            attention_mask,
        )
        return self.note_transformer(score_hidden, notes)

    def next_score_hidden(
        self,
        notes: torch.Tensor,
        absolute_starts: torch.Tensor,
    ) -> torch.Tensor:
        if notes.ndim != 3 or notes.shape[-1] != 3:
            raise ValueError("notes must have shape [B, T, 3]")
        if absolute_starts.shape != notes.shape[:2]:
            raise ValueError(
                "absolute_starts must have shape [B, T] matching notes"
            )
        if self.config.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2 for inference")
        context = notes[:, -(self.config.max_seq_len - 1) :]
        context_starts = absolute_starts[
            :, -(self.config.max_seq_len - 1) :
        ]
        dummy = torch.zeros(
            notes.shape[0],
            1,
            3,
            dtype=notes.dtype,
            device=notes.device,
        )
        inputs = torch.cat((context, dummy), dim=1)
        dummy_start = torch.zeros(
            notes.shape[0],
            1,
            dtype=absolute_starts.dtype,
            device=absolute_starts.device,
        )
        input_starts = torch.cat((context_starts, dummy_start), dim=1)
        attention_mask = torch.ones(
            inputs.shape[:2],
            dtype=torch.long,
            device=inputs.device,
        )
        return self.score_transformer(
            inputs,
            input_starts,
            attention_mask,
        )[:, -1]

