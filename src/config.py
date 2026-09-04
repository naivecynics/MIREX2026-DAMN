from dataclasses import dataclass


EOS_TOKEN_ID = 128


@dataclass
class Config:
    start_vocab_size: int = 129
    pitch_vocab_size: int = 128
    duration_vocab_size: int = 128
    max_seq_len: int = 513
    score_max_position_embeddings: int = 4096
    score_position_mode: str = "absolute"

    score_hidden_size: int = 768
    score_intermediate_size: int = 3072
    score_num_layers: int = 14
    score_num_heads: int = 12

    note_intermediate_size: int = 1536
    note_num_layers: int = 2
    note_num_heads: int = 12

    mixed_precision: str = "bf16"

    @property
    def eos_token_id(self) -> int | None:
        return EOS_TOKEN_ID if self.start_vocab_size > EOS_TOKEN_ID else None
