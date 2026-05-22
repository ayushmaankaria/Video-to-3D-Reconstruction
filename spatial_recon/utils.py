from __future__ import annotations

import colorsys
import json
import shlex
import sys
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(directory: str | Path) -> list[Path]:
    """Image files in ``directory``, sorted lexicographically (matches the
    zero-padded names written by ``video.extract_frames``)."""
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"Not a directory: {d}")
    files = [p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: p.name)
    return files


def write_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=str))
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


# Deterministic per-class color palette. Class 0 ("unknown") is gray.
_PALETTE_CACHE: dict[int, tuple[int, int, int]] = {0: (140, 140, 140)}


def semantic_color(label_id: int) -> tuple[int, int, int]:
    label_id = int(label_id)
    if label_id in _PALETTE_CACHE:
        return _PALETTE_CACHE[label_id]
    # Golden-ratio hue spacing keeps neighbouring class ids visually separated.
    hue = (label_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    color = (int(r * 255), int(g * 255), int(b * 255))
    _PALETTE_CACHE[label_id] = color
    return color


def command_string(_args=None) -> str:
    """Reconstruct the invocation line for the report."""
    return " ".join(shlex.quote(s) for s in sys.argv)