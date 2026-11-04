import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from .config import TrainConfig as Config


def _distributed_scheduler_steps(
    config: Config,
    num_processes: int,
) -> tuple[int, int, int]:
    """Express the prepared scheduler in global optimizer steps."""
    return (
        config.warmup_steps * num_processes,
        config.anneal_steps * num_processes,
        config.max_steps * num_processes,
    )


def _lr_multiplier(
    current_step: int,
    *,
    warmup_steps: int,
    anneal_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    """Linear warmup, constant learning rate, then cosine cooldown."""
    if warmup_steps and current_step < warmup_steps:
        return current_step / warmup_steps

    anneal_start = total_steps - anneal_steps
    if not anneal_steps or current_step < anneal_start:
        return 1.0

    progress = min(1.0, (current_step - anneal_start) / anneal_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Config,
    num_processes: int,
) -> LambdaLR:
    if config.warmup_steps < 0 or config.anneal_steps < 0:
        raise ValueError("scheduler step counts must be non-negative")
    if config.warmup_steps + config.anneal_steps > config.max_steps:
        raise ValueError("warmup_steps + anneal_steps must not exceed max_steps")
    if not 0 <= config.min_learning_rate <= config.learning_rate:
        raise ValueError(
            "min_learning_rate must be between zero and learning_rate"
        )

    warmup_steps, anneal_steps, total_steps = _distributed_scheduler_steps(
        config,
        num_processes,
    )
    min_lr_ratio = config.min_learning_rate / config.learning_rate
    return LambdaLR(
        optimizer,
        lambda step: _lr_multiplier(
            step,
            warmup_steps=warmup_steps,
            anneal_steps=anneal_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )



