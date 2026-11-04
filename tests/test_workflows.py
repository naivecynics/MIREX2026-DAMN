import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_model

from evaluation.common import atomic_json, benchmark_items
from evaluation.run_ours import parse_args, run
from evaluation.score_clamp3 import parse_args as score_args, run as score_run
from scripts.export_checkpoint import export
from src.model import NestedMusicTransformer
from training.config import TrainConfig


def test_portable_paths_and_model_presets(tmp_path):
    config = TrainConfig.from_model_size("mini", base_path=tmp_path)
    assert config.parquet_data_dir == tmp_path / "data/parquets"
    assert config.checkpoint_dir == tmp_path / "checkpoints"
    assert config.score_hidden_size == 384
    assert config.note_num_layers == 1


def test_export_batch_generation_and_resume(tmp_path):
    config = TrainConfig(
        base_path=tmp_path, score_hidden_size=16, score_intermediate_size=32,
        score_num_layers=1, score_num_heads=4, note_intermediate_size=32,
        note_num_layers=1, note_num_heads=4, mixed_precision="no",
    )
    checkpoint = tmp_path / "run/step_00000002"
    checkpoint.mkdir(parents=True)
    model = NestedMusicTransformer(config)
    save_model(model, str(checkpoint / "model.safetensors"))
    raw = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    atomic_json(checkpoint.parent / "config.json", raw)
    exported = tmp_path / "export"
    export(checkpoint, exported)
    assert set(p.name for p in exported.iterdir()) == {"config.json", "model.safetensors"}
    assert "base_path" not in json.loads((exported / "config.json").read_text())
    assert (exported / "model.safetensors").read_bytes() == (checkpoint / "model.safetensors").read_bytes()
    with pytest.raises(FileExistsError):
        export(checkpoint, exported)

    benchmark = tmp_path / "benchmark"
    atomic_json(benchmark / "prompts/piece.json", {"prompt": [{"start": 16, "pitch": 60, "duration": 4}]})
    atomic_json(benchmark / "references/piece.json", {"generation": [{"start": 80, "pitch": 64, "duration": 4}]})
    atomic_json(benchmark / "manifest.json", {"samples": [{"cid": "piece", "artifacts": {"prompt_json": "prompts/piece.json", "reference_json": "references/piece.json"}}]})
    output = tmp_path / "outputs"
    args = parse_args(["--checkpoint", str(exported), "--test-root", str(benchmark),
                       "--output-root", str(output), "--n-samples", "2",
                       "--max-notes", "2", "--min-generation-end", "0"])
    assert run(args) == 0
    path = output / "piece/sample_01.json"
    before = path.read_bytes()
    assert run(args) == 0
    assert path.read_bytes() == before
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert all(item["status"] == "skipped" for item in manifest["items"])
    path.write_text("broken")
    assert run(args) == 0
    assert path.read_bytes() == before


def test_scoring_pipeline_with_test_encoder(tmp_path, monkeypatch):
    """Exercise orchestration without downloading the external CLaMP3 model."""
    import evaluation.score_clamp3 as scorer

    class TestEncoder:
        def __init__(self, *args):
            pass

        def metadata(self):
            return {"encoder": "test-only"}

        def extract(self, midi_root, feature_root, *, changed):
            for relative in changed:
                destination = (feature_root / relative).with_suffix(".npy")
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.save(destination, [1.0, 0.0])

    monkeypatch.setattr(scorer, "Clamp3Encoder", TestEncoder)
    benchmark = tmp_path / "benchmark"
    atomic_json(benchmark / "prompts/piece.json", {"prompt": []})
    atomic_json(benchmark / "references/piece.json", {"generation": []})
    atomic_json(benchmark / "manifest.json", {"samples": [{"cid": "piece", "artifacts": {"prompt_json": "prompts/piece.json", "reference_json": "references/piece.json"}}]})
    candidate = tmp_path / "candidate"
    for index in (1, 2):
        atomic_json(candidate / f"piece/sample_{index:02d}.json", {"generation": []})
    args = score_args(["--test-root", str(benchmark), "--candidate", f"mini={candidate}",
                       "--n-samples", "2", "--clamp3-root", "unused", "--clamp3-python", "unused",
                       "--output-root", str(tmp_path / "scores"), "--bootstrap-samples", "20"])
    result = score_run(args)
    summary = json.loads((result / "summary.json").read_text())
    assert next(row for row in summary["rows"] if row["source"] == "mini")["count"] == 1
    assert score_run(args) == result
