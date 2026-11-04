from dataclasses import dataclass
from pathlib import Path

from src.config import Config


MODEL_PRESETS = {
    "mini": dict(score_hidden_size=384, score_intermediate_size=1536,
                 score_num_layers=8, score_num_heads=6,
                 note_intermediate_size=768, note_num_layers=1, note_num_heads=6),
    "small": dict(score_hidden_size=640, score_intermediate_size=2560,
                  score_num_layers=12, score_num_heads=10,
                  note_intermediate_size=1280, note_num_layers=2, note_num_heads=10),
    "medium": dict(score_hidden_size=768, score_intermediate_size=3072,
                   score_num_layers=14, score_num_heads=12,
                   note_intermediate_size=1536, note_num_layers=2, note_num_heads=12),
}
# Historical experiment name for DAMN-Mini.
MODEL_PRESETS["tiny"] = MODEL_PRESETS["mini"]


@dataclass
class TrainConfig(Config):
    """Portable training settings, separate from the model/checkpoint config."""

    base_path: Path = Path(".")
    data_dir: Path | None = None
    parquet_data_dir: Path | None = None
    logging_dir: Path | None = None
    wandb_dir: Path | None = None
    checkpoint_dir: Path | None = None

    batch_size: int = 24  # Per GPU, not global.
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    anneal_steps: int = 42162
    min_learning_rate: float = 3e-5
    max_steps: int = 210810
    max_grad_norm: float = 1.0
    num_workers: int = 4
    seed: int = 42
    log_steps: int = 10
    eval_steps: int = 2000
    save_steps: int = 5000
    max_checkpoints: int = 3
    validation_batches: int = 100
    wandb_project: str = "mirex2026"

    def __post_init__(self) -> None:
        self.base_path = Path(self.base_path).expanduser()
        self.data_dir = Path(self.data_dir or self.base_path / "data").expanduser()
        for name, default in {
            "parquet_data_dir": self.data_dir / "parquets",
            "logging_dir": self.base_path / "logs",
            "wandb_dir": self.base_path / "wandb",
            "checkpoint_dir": self.base_path / "checkpoints",
        }.items():
            setattr(self, name, Path(getattr(self, name) or default).expanduser())

    @classmethod
    def from_model_size(cls, model_size: str, **overrides: object) -> "TrainConfig":
        if model_size not in MODEL_PRESETS:
            raise ValueError(f"unknown model size: {model_size!r}")
        return cls(**(MODEL_PRESETS[model_size] | overrides))
