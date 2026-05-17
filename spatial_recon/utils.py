from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def resize_label_map_nearest(label_map: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.fromarray(label_map.astype(np.int32))
    return np.asarray(img.resize((w, h), Image.Resampling.NEAREST)).astype(np.int32)


def semantic_color(label_id: int) -> tuple[int, int, int]:
    palette = [
        (160, 160, 160),
        (66, 135, 245),
        (245, 166, 35),
        (80, 180, 120),
        (220, 80, 95),
        (155, 95, 220),
        (40, 180, 190),
        (235, 215, 75),
        (110, 110, 110),
        (235, 120, 45),
        (95, 190, 80),
        (70, 95, 210),
        (210, 85, 170),
        (55, 150, 150),
        (180, 130, 70),
        (120, 160, 230),
    ]
    if label_id <= 0:
        return palette[0]
    if label_id < len(palette):
        return palette[label_id]
    x = int(label_id * 2654435761) & 0xFFFFFFFF
    r = int(80 + (x & 0x7F))
    g = int(80 + ((x >> 8) & 0x7F))
    b = int(80 + ((x >> 16) & 0x7F))
    return (r, g, b)


def palette_for_labels(label_ids: Iterable[int]) -> dict[int, tuple[int, int, int]]:
    return {int(label_id): semantic_color(int(label_id)) for label_id in sorted(set(label_ids))}


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if image.max(initial=0) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def command_string(args: object) -> str:
    values = vars(args)
    parts = ["python", "-m", "spatial_recon.cli"]
    command = values.get("command")
    if command:
        parts.append(str(command))
    for key, value in values.items():
        if key == "command" or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            parts.append(flag)
        else:
            parts.extend([flag, str(value)])
    return " ".join(parts)
