from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: object):
        return iterable

from .utils import ensure_dir, list_images


# VGGT-Omega's vision tower runs internally at its own resolution, but for
# pixel-level provenance (query.py samples masks at source pixels) we want
# the on-disk frames at a reasonable size. 1280 long-side is a good compromise
# between fidelity and disk / SAM 3 throughput.
DEFAULT_IMAGE_SIZE = 1280


@dataclass
class FramesMeta:
    """Persisted metadata about the extracted frame set."""
    source: str
    num_frames: int
    fps_source: float
    width: int
    height: int
    source_width: int
    source_height: int
    image_size: int
    mode: str

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "FramesMeta":
        return cls(**json.loads(Path(path).read_text()))


def load_frames_meta(run_dir: str | Path) -> FramesMeta:
    """Read ``frames_meta.json`` written by ``extract_frames``."""
    return FramesMeta.from_json(Path(run_dir) / "frames_meta.json")


def _compute_target_hw(src_w: int, src_h: int, image_size: Optional[int]) -> tuple[int, int]:
    """Return (H, W) after resizing so the long side equals ``image_size``."""
    if not image_size or image_size <= 0:
        return src_h, src_w
    long_side = max(src_w, src_h)
    if long_side == image_size:
        return src_h, src_w
    scale = image_size / float(long_side)
    new_w = max(2, int(round(src_w * scale)))
    new_h = max(2, int(round(src_h * scale)))
    new_w -= new_w % 2
    new_h -= new_h % 2
    return new_h, new_w


def _laplacian_var(gray: np.ndarray) -> float:
    """Variance-of-Laplacian sharpness score (higher is sharper)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _select_indices_uniform(n_total: int, max_frames: int) -> np.ndarray:
    """Evenly-spaced source-frame indices."""
    if n_total <= max_frames:
        return np.arange(n_total, dtype=np.int64)
    return np.linspace(0, n_total - 1, num=max_frames).round().astype(np.int64)


def _select_indices_hybrid(
    cap: "cv2.VideoCapture",
    n_total: int,
    max_frames: int,
) -> np.ndarray:
    """Pick the sharpest frame from each of ``max_frames`` uniform bins.

    Decoding every frame just to score sharpness would dominate runtime, so
    we sample at most ~6 candidates per bin and pick the sharpest.
    """
    if n_total <= max_frames:
        return np.arange(n_total, dtype=np.int64)

    bin_edges = np.linspace(0, n_total, num=max_frames + 1).astype(np.int64)
    selected: list[int] = []
    samples_per_bin = 6

    for b in range(max_frames):
        s, e = int(bin_edges[b]), int(bin_edges[b + 1])
        if e <= s:
            continue
        candidates = np.linspace(s, e - 1, num=min(samples_per_bin, e - s)).astype(np.int64)
        best_idx, best_score = int(candidates[0]), -1.0
        for ci in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ci))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = _laplacian_var(gray)
            if score > best_score:
                best_score = score
                best_idx = int(ci)
        selected.append(best_idx)

    return np.asarray(sorted(set(selected)), dtype=np.int64)


def extract_frames(
    video_path: str | Path,
    frames_dir: str | Path,
    *,
    max_frames: int = 24,
    mode: str = "hybrid",
    image_size: Optional[int] = DEFAULT_IMAGE_SIZE,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> list[Path]:
    """Decode ``video_path`` into ``frames_dir`` and return the written paths.

    Parameters
    ----------
    max_frames :
        Total frames to keep.
    mode :
        ``"hybrid"`` picks the sharpest frame per uniform time bin (recommended
        for hand-held phone video). ``"uniform"`` is plain even spacing.
    image_size :
        Long-side resize target; pass ``None`` or ``0`` to keep source res.
    """
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is required. Install with `pip install opencv-python` "
            "(or opencv-python-headless on servers/Colab)."
        )
    if mode not in ("hybrid", "uniform"):
        raise ValueError(f"mode must be 'hybrid' or 'uniform', got {mode!r}")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    frames_dir = Path(frames_dir)
    if frames_dir.exists() and any(frames_dir.glob(f"*.{image_format}")):
        if overwrite:
            shutil.rmtree(frames_dir)
        else:
            existing = sorted(p for p in frames_dir.iterdir()
                              if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if existing:
                # Reuse: load metadata if present, otherwise rebuild a meta file.
                run_dir = frames_dir.parent
                if (run_dir / "frames_meta.json").exists():
                    print(f"[Video] Reusing {len(existing)} frames in {frames_dir}")
                    return existing
    ensure_dir(frames_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if src_w <= 0 or src_h <= 0 or n_total <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video dimensions / frame count from {video_path}")

    # --- pick source indices ---
    if mode == "uniform":
        keep_idx = _select_indices_uniform(n_total, max_frames)
    else:
        keep_idx = _select_indices_hybrid(cap, n_total, max_frames)

    if len(keep_idx) == 0:
        cap.release()
        raise RuntimeError(f"No frames selected from {video_path}.")

    out_h, out_w = _compute_target_hw(src_w, src_h, image_size)
    pad = max(4, int(math.ceil(math.log10(len(keep_idx) + 1))))
    written: list[Path] = []

    pbar = tqdm(total=len(keep_idx), desc="Extracting frames")
    try:
        for out_count, src_idx in enumerate(keep_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            if (frame_bgr.shape[1], frame_bgr.shape[0]) != (out_w, out_h):
                frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            out_path = frames_dir / f"{out_count:0{pad}d}.{image_format}"
            if image_format.lower() in ("jpg", "jpeg"):
                Image.fromarray(frame_rgb).save(out_path, quality=jpeg_quality, subsampling=0)
            else:
                Image.fromarray(frame_rgb).save(out_path)
            written.append(out_path)
            pbar.update(1)
    finally:
        cap.release()
        pbar.close()

    if not written:
        raise RuntimeError(f"Decoded 0 frames from {video_path}.")

    meta = FramesMeta(
        source=str(video_path.resolve()),
        num_frames=len(written),
        fps_source=float(src_fps if src_fps > 0 else 30.0),
        width=int(out_w),
        height=int(out_h),
        source_width=int(src_w),
        source_height=int(src_h),
        image_size=int(image_size or 0),
        mode=mode,
    )
    meta.to_json(frames_dir.parent / "frames_meta.json")
    print(f"[Video] Wrote {len(written)} frames @ {out_w}x{out_h} "
          f"({mode}) to {frames_dir}")
    return written