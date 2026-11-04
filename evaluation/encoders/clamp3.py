from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Clamp3Encoder:
    """Thin adapter around the official CLaMP 3 feature extractor.

    CLaMP 3 deliberately lives in its own checkout and Python environment: its
    pinned Transformers dependency is incompatible with the training runtime.
    """

    root: Path
    python: Path

    def validate(self) -> None:
        root = self.root.expanduser().resolve()
        python = self.python.expanduser().resolve()
        required = (
            root / "preprocessing" / "midi" / "batch_midi2mtf.py",
            root / "code" / "extract_clamp3.py",
            root / "code" / "config.py",
            python,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "CLaMP 3 installation is incomplete: " + ", ".join(missing)
            )
        config = (root / "code" / "config.py").read_text(encoding="utf-8")
        if '"weights_clamp3_c2"' not in config:
            raise ValueError(
                "CLaMP 3 must use the symbolic C2 checkpoint; change the "
                'CLAMP3_WEIGHTS_PATH prefix in code/config.py to "weights_clamp3_c2"'
            )
        if not list((root / "code").glob("weights_clamp3_c2*.pth")):
            raise FileNotFoundError(
                f"no CLaMP 3 C2 checkpoint found under {root / 'code'}"
            )

    def extract(
        self,
        midi_root: Path,
        feature_root: Path,
        *,
        changed: Iterable[Path] = (),
    ) -> None:
        """Extract global features while retaining the official on-disk cache."""

        self.validate()
        root = self.root.expanduser().resolve()
        python = self.python.expanduser().resolve()
        midi_root = midi_root.resolve()
        feature_root = feature_root.resolve()
        mtf_root = feature_root.parent / "mtf"
        mtf_root.mkdir(parents=True, exist_ok=True)

        changed = tuple(changed)
        for relative in changed:
            feature = (feature_root / relative).with_suffix(".npy")
            if feature.is_file():
                feature.unlink()
            # The official converter may skip existing MTF files. Invalidate
            # both cache layers so changed JSON cannot reuse old MIDI tokens.
            for cached in (mtf_root / relative.parent).glob(relative.stem + ".*"):
                if cached.is_file():
                    cached.unlink()

        subprocess.run(
            [
                str(python),
                str(root / "preprocessing" / "midi" / "batch_midi2mtf.py"),
                str(midi_root),
                str(mtf_root),
                "--m3_compatible",
            ],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "extract_clamp3.py",
                str(mtf_root),
                str(feature_root),
                "--get_global",
            ],
            cwd=root / "code",
            check=True,
        )

    def metadata(self) -> dict[str, object]:
        self.validate()
        root = self.root.expanduser().resolve()
        checkpoint = next(
            iter(sorted((root / "code").glob("weights_clamp3_c2*.pth")))
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "config_sha256": _sha256(root / "code" / "config.py"),
            "encoder": "clamp3_c2",
            "feature": "global",
            "python": str(self.python.expanduser().resolve()),
            "repository": str(root),
            "repository_commit": commit,
            "repository_dirty": dirty,
        }
