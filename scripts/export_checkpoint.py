"""Export only inference weights and model config, not optimizer or local paths."""

import argparse
import json
import shutil
import sys
from dataclasses import asdict, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config


def export(checkpoint: Path, output: Path) -> None:
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    config_path = next((path for path in (checkpoint / "config.json", checkpoint.parent / "config.json") if path.is_file()), None)
    if config_path is None:
        raise FileNotFoundError("checkpoint must have a model or run config.json")
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite a nonempty directory: {output}")
    raw = json.loads(config_path.read_text())
    keys = {field.name for field in fields(Config)}
    config = Config(**{key: value for key, value in raw.items() if key in keys})
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weights, output / weights.name)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
