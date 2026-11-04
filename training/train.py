from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list, set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import MODEL_PRESETS, TrainConfig as Config
from .loss import music_loss
from .schedule import _build_scheduler, _distributed_scheduler_steps, _lr_multiplier
from .data import ParquetMusicDataset, collate_notes
from src.model import NestedMusicTransformer


def _config_dict(config: Config) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _precision(config: Config) -> str:
    if (
        config.mixed_precision == "bf16"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return "bf16"
    return "no"


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader | None,
    max_batches: int,
    eos_token_id: int | None,
) -> dict[str, float] | None:
    if loader is None:
        return None
    model.eval()
    totals = torch.zeros(5, device=accelerator.device)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        notes = batch["notes"]
        absolute_starts = batch["absolute_starts"]
        attention_mask = batch["attention_mask"]
        with accelerator.autocast():
            logits = model(
                notes,
                attention_mask=attention_mask,
                absolute_starts=absolute_starts,
            )
            total, losses = music_loss(
                logits,
                notes,
                attention_mask,
                eos_token_id,
            )
        totals += torch.stack(
            (
                total.detach(),
                losses["start"].detach(),
                losses["pitch"].detach(),
                losses["duration"].detach(),
                torch.ones((), device=accelerator.device),
            )
        )

    totals = accelerator.reduce(totals, reduction="sum")
    model.train()
    if totals[-1].item() == 0:
        return None
    means = totals[:4] / totals[-1]
    return {
        "validation/loss": means[0].item(),
        "validation/start_loss": means[1].item(),
        "validation/pitch_loss": means[2].item(),
        "validation/duration_loss": means[3].item(),
    }


def save_checkpoint(
    accelerator: Accelerator,
    run_dir: Path,
    global_step: int,
    max_checkpoints: int,
) -> Path:
    path = run_dir / f"step_{global_step:08d}"
    accelerator.save_state(str(path))
    if accelerator.is_main_process:
        (path / "trainer_state.json").write_text(
            json.dumps({"global_step": global_step}, indent=2) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoints = sorted(
            child
            for child in run_dir.iterdir()
            if child.is_dir()
            and re.fullmatch(r"step_\d{8}", child.name) is not None
        )
        for old_checkpoint in checkpoints[:-max_checkpoints]:
            shutil.rmtree(old_checkpoint)
    accelerator.wait_for_everyone()
    return path


def _run_directory(
    accelerator: Accelerator,
    checkpoint_dir: Path,
    run_name: str | None,
    resume: Path | None,
) -> Path:
    if resume is not None and resume.parent.parent == checkpoint_dir:
        return resume.parent

    names = [
        run_name
        or (
            datetime.now().strftime("run_%Y%m%d_%H%M%S_")
            + uuid4().hex[:8]
        )
        if accelerator.is_main_process
        else None
    ]
    broadcast_object_list(names)
    name = str(names[0])
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"invalid run name: {name!r}")
    return checkpoint_dir / name


def train(
    config: Config,
    *,
    use_wandb: bool = True,
    resume: Path | None = None,
    run_name: str | None = None,
) -> int:
    if config.max_checkpoints < 1:
        raise ValueError("max_checkpoints must be at least 1")
    set_seed(config.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=_precision(config),
        log_with="wandb" if use_wandb else None,
        project_dir=str(config.logging_dir),
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
    )
    run_dir = _run_directory(
        accelerator,
        config.checkpoint_dir,
        run_name,
        resume,
    )
    if accelerator.is_main_process:
        for path in (
            config.logging_dir,
            config.wandb_dir,
            run_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        if not config_path.exists():
            config_path.write_text(
                json.dumps(_config_dict(config), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
    accelerator.wait_for_everyone()

    if use_wandb:
        accelerator.init_trackers(
            config.wandb_project,
            config=_config_dict(config),
            init_kwargs={
                "wandb": {
                    "dir": str(config.wandb_dir),
                    "name": run_dir.name,
                }
            },
        )

    train_dataset = ParquetMusicDataset(config, "train", shuffle=True)
    validation_dataset = ParquetMusicDataset(config, "validation")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_notes,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=collate_notes,
            pin_memory=torch.cuda.is_available(),
        )
        if validation_dataset.has_samples()
        else None
    )

    model = NestedMusicTransformer(config)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _build_scheduler(
        optimizer,
        config,
        accelerator.num_processes,
    )
    if validation_loader is None:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            scheduler,
        )
    else:
        model, optimizer, train_loader, validation_loader, scheduler = (
            accelerator.prepare(
                model,
                optimizer,
                train_loader,
                validation_loader,
                scheduler,
            )
        )

    global_step = 0
    if resume is not None:
        accelerator.load_state(str(resume))
        state_path = resume / "trainer_state.json"
        global_step = json.loads(state_path.read_text(encoding="utf-8"))[
            "global_step"
        ]

    optimizer.zero_grad()
    epoch = 0
    last_checkpoint: Path | None = None
    while global_step < config.max_steps:
        train_dataset.set_epoch(epoch)
        saw_batch = False
        for batch in train_loader:
            saw_batch = True
            notes = batch["notes"]
            absolute_starts = batch["absolute_starts"]
            attention_mask = batch["attention_mask"]
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    logits = model(
                        notes,
                        attention_mask=attention_mask,
                        absolute_starts=absolute_starts,
                    )
                    loss, losses = music_loss(
                        logits,
                        notes,
                        attention_mask,
                        config.eos_token_id,
                    )
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(),
                        config.max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue
            global_step += 1

            if global_step % config.log_steps == 0:
                logs = {
                    "train/loss": accelerator.reduce(
                        loss.detach(), reduction="mean"
                    ).item(),
                    "train/start_loss": accelerator.reduce(
                        losses["start"].detach(), reduction="mean"
                    ).item(),
                    "train/pitch_loss": accelerator.reduce(
                        losses["pitch"].detach(), reduction="mean"
                    ).item(),
                    "train/duration_loss": accelerator.reduce(
                        losses["duration"].detach(), reduction="mean"
                    ).item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                }
                if grad_norm is not None:
                    logs["train/gradient_norm"] = float(grad_norm)
                accelerator.log(logs, step=global_step)

            if global_step % config.eval_steps == 0:
                metrics = evaluate(
                    accelerator,
                    model,
                    validation_loader,
                    config.validation_batches,
                    config.eos_token_id,
                )
                if metrics is not None:
                    accelerator.log(metrics, step=global_step)

            if global_step % config.save_steps == 0:
                last_checkpoint = save_checkpoint(
                    accelerator,
                    run_dir,
                    global_step,
                    config.max_checkpoints,
                )
            if global_step >= config.max_steps:
                break

        if not saw_batch:
            raise RuntimeError("training split contains no samples")
        epoch += 1

    expected = run_dir / f"step_{global_step:08d}"
    if last_checkpoint != expected:
        save_checkpoint(
            accelerator,
            run_dir,
            global_step,
            config.max_checkpoints,
        )
    accelerator.end_training()
    return global_step


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("must be at least 1")
        return parsed

    def non_negative_int(value: str) -> int:
        parsed = int(value)
        if parsed < 0:
            raise argparse.ArgumentTypeError("must be non-negative")
        return parsed

    def non_negative_float(value: str) -> float:
        parsed = float(value)
        if parsed < 0:
            raise argparse.ArgumentTypeError("must be non-negative")
        return parsed

    parser = argparse.ArgumentParser(description="Train the nested music model.")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--parquet-data-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=non_negative_int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-steps", type=positive_int)
    parser.add_argument("--validation-batches", type=positive_int)
    parser.add_argument("--mixed-precision", choices=("no", "bf16"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--model-size", choices=MODEL_PRESETS)
    parser.add_argument(
        "--score-position-mode",
        choices=("absolute", "index"),
    )
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int)
    parser.add_argument("--max-steps", type=positive_int)
    parser.add_argument("--warmup-steps", type=non_negative_int)
    parser.add_argument("--anneal-steps", type=non_negative_int)
    parser.add_argument("--min-learning-rate", type=non_negative_float)
    parser.add_argument("--eval-steps", type=positive_int)
    parser.add_argument("--save-steps", type=positive_int)
    parser.add_argument("--max-checkpoints", type=positive_int)
    parser.add_argument("--run-name")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume is not None:
        from dataclasses import fields

        config_path = args.resume.parent / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"resume requires the run config: {config_path}")
        raw = json.loads(config_path.read_text())
        known = {field.name for field in fields(Config)}
        config = Config(**{key: value for key, value in raw.items() if key in known})
        if args.model_size is not None or args.score_position_mode is not None:
            raise ValueError("do not change model architecture when resuming")
    else:
        path_overrides = {
            key: getattr(args, key)
            for key in ("base_path", "parquet_data_dir", "checkpoint_dir")
            if getattr(args, key) is not None
        }
        config = Config.from_model_size(args.model_size or "medium", **path_overrides)
    for key in ("learning_rate", "num_workers", "seed", "log_steps",
                "validation_batches", "mixed_precision"):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.score_position_mode is not None:
        config.score_position_mode = args.score_position_mode
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.anneal_steps is not None:
        config.anneal_steps = args.anneal_steps
    if args.min_learning_rate is not None:
        config.min_learning_rate = args.min_learning_rate
    if args.eval_steps is not None:
        config.eval_steps = args.eval_steps
    if args.save_steps is not None:
        config.save_steps = args.save_steps
    if args.max_checkpoints is not None:
        config.max_checkpoints = args.max_checkpoints
    train(
        config,
        use_wandb=not args.no_wandb,
        resume=args.resume,
        run_name=args.run_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
