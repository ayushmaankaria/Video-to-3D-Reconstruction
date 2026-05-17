from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - surfaced when extraction is requested.
    cv2 = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: object):
        return iterable

from .utils import ensure_dir, list_images


def _sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _hybrid_indices(cap: cv2.VideoCapture, frame_count: int, max_frames: int) -> list[int]:
    if max_frames >= frame_count:
        return list(range(frame_count))

    bins = np.linspace(0, frame_count, max_frames + 1, dtype=int)
    selected: list[int] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        hi = max(hi, lo + 1)
        candidates = np.linspace(lo, hi - 1, min(8, hi - lo), dtype=int)
        best_idx = int(candidates[0])
        best_score = -1.0
        for idx in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            score = _sharpness(frame)
            if score > best_score:
                best_score = score
                best_idx = int(idx)
        selected.append(best_idx)
    return sorted(set(selected))


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    max_frames: int = 24,
    mode: str = "hybrid",
    image_size: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Extract a compact, sharp, temporally spread frame set from a video."""
    if cv2 is None:
        raise ImportError("Frame extraction needs opencv-python. Install requirements.txt first.")

    video_path = Path(video_path)
    out_dir = ensure_dir(out_dir)

    existing = list_images(out_dir)
    if existing and not overwrite:
        return existing

    for old in existing:
        old.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise ValueError(f"Video has no readable frames: {video_path}")

    if mode == "uniform":
        indices = np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int).tolist()
    elif mode == "hybrid":
        indices = _hybrid_indices(cap, frame_count, max_frames)
    else:
        raise ValueError(f"Unknown frame mode '{mode}'. Use 'hybrid' or 'uniform'.")

    output_paths: list[Path] = []
    for out_idx, frame_idx in enumerate(tqdm(indices, desc="extracting frames")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        if image_size:
            h, w = frame.shape[:2]
            scale = image_size / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        out_path = out_dir / f"{out_idx:04d}.jpg"
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        output_paths.append(out_path)

    cap.release()
    if not output_paths:
        raise RuntimeError(f"No frames were extracted from {video_path}")
    return output_paths
