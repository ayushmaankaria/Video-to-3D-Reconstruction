from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .video import extract_frames, load_frames_meta
from .predict import predict_geometry, load_predictions
from .semantics import run_semantics, load_label_maps
from .fusion import fuse_predictions, save_fused_cloud
from .query import query_scene
from .export import write_ply, write_glb, write_html_viewer, write_legend, semantic_colors_for_cloud
from .report import write_report
from .utils import ensure_dir, command_string


def run_pipeline(
    *,
    video_path: str | Path,
    run_dir: str | Path,
    checkpoint: str | Path,
    image_resolution: int = 512,
    max_frames: int = 24,
    target_fps: float | None = 4.0,
    concepts: Iterable[str] | None = None,
    score_threshold: float = 0.5,
    conf_percentile: float = 35.0,
    sample_stride: int = 2,
    voxel_size: float = 0.015,
    max_points: int | None = 450_000,
    no_semantics: bool = False,
    device: str | None = None,
    overwrite_frames: bool = False,
    overwrite_predictions: bool = False,
    overwrite_semantics: bool = False,
) -> dict:
    """End-to-end: video → frames → VGGT-Omega → SAM 3 → fused cloud + exports.

    Notes
    -----
    ``target_fps`` is accepted for CLI back-compat but is largely superseded by
    ``max_frames`` combined with sharpness-aware per-bin selection
    (``mode="hybrid"`` inside :func:`extract_frames`). When both are provided
    we use ``target_fps`` only to *cap* the number of frames so we don't
    oversample short clips.
    """
    run_dir = ensure_dir(Path(run_dir))

    # 1) Frames
    frames_meta_path = run_dir / "frames_meta.json"
    frames_dir = run_dir / "frames"
    if frames_meta_path.exists() and not overwrite_frames:
        meta = load_frames_meta(run_dir)
        print(f"[Pipeline] Reusing {meta.num_frames} frames @ {meta.width}x{meta.height}")
    else:
        effective_max = max_frames
        if target_fps and target_fps > 0:
            try:
                import cv2  # local import: extract_frames already requires it
                cap = cv2.VideoCapture(str(video_path))
                src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
                if src_fps > 0 and n_total > 0:
                    duration = n_total / src_fps
                    fps_cap = max(1, int(duration * target_fps))
                    effective_max = min(max_frames, fps_cap)
            except Exception as exc:
                print(f"[Pipeline] target_fps probe failed ({exc}); "
                      f"falling back to max_frames={max_frames}")

        extract_frames(
            video_path,
            frames_dir,
            max_frames=effective_max,
            mode="hybrid",
            overwrite=overwrite_frames,
        )
        meta = load_frames_meta(run_dir)

    # 2) Geometry
    preds_path = run_dir / "predictions.npz"
    if preds_path.exists() and not overwrite_predictions:
        predictions = load_predictions(run_dir)
        print("[Pipeline] Reusing predictions.npz")
    else:
        predictions = predict_geometry(
            run_dir,
            checkpoint=checkpoint,
            image_resolution=image_resolution,
            device=device,
        )

    # 3) Semantics
    semantics_dir = run_dir / "semantics"
    if no_semantics:
        label_maps, labels = [], {0: "unknown"}
    elif (semantics_dir / "labels.json").exists() and not overwrite_semantics:
        label_maps, labels = load_label_maps(semantics_dir)
        print(f"[Pipeline] Reusing {len(label_maps)} SAM 3 label maps")
    else:
        run_semantics(
            run_dir / "frames",
            semantics_dir,
            concepts=list(concepts) if concepts else None,
            score_threshold=score_threshold,
            device=device,
        )
        label_maps, labels = load_label_maps(semantics_dir)

    # 4) Fusion
    cloud = fuse_predictions(
        predictions,
        label_maps=label_maps if label_maps else None,
        labels=labels,
        conf_percentile=conf_percentile,
        sample_stride=sample_stride,
        voxel_size=voxel_size,
        max_points=max_points,
    )
    print(f"[Pipeline] Fused {len(cloud.points):,} points across "
          f"{len(set(cloud.source_frame.tolist()))} frames")

    # 5) Exports
    exports = ensure_dir(run_dir / "exports")
    save_fused_cloud(exports / "fused_points.npz", cloud)
    write_ply(exports / "reconstruction_rgb.ply",
              {"points": cloud.points, "colors": cloud.colors})
    sem_colors = semantic_colors_for_cloud(cloud)
    write_ply(exports / "reconstruction_semantic.ply",
              {"points": cloud.points, "colors": sem_colors})
    try:
        write_glb(exports / "reconstruction_semantic.glb",
                  points=cloud.points, colors=sem_colors)
    except Exception as exc:
        print(f"[Pipeline] GLB export skipped: {exc}")
    write_html_viewer(exports / "viewer.html", cloud,
                      extrinsics=predictions.get("extrinsics"))
    write_legend(exports / "semantic_legend.json", cloud)
    write_report(
        run_dir / "REPORT.md", cloud,
        frame_count=meta.num_frames,
        video_path=str(video_path),
        checkpoint=str(checkpoint),
        semantic_model=None if no_semantics else "sam3",
        command=command_string(),
    )

    print(f"[Pipeline] Done. Open {exports / 'viewer.html'}")
    return {"run_dir": str(run_dir), "meta": meta, "cloud_size": len(cloud.points)}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spatial_recon.pipeline",
                                description="Video → 3D → text query (programmatic CLI)")
    p.add_argument("video")
    p.add_argument("--name", required=True)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image-resolution", type=int, default=512, choices=[256, 512])
    p.add_argument("--max-frames", type=int, default=24)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--concepts", type=str, default=None)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--conf-percentile", type=float, default=35.0)
    p.add_argument("--sample-stride", type=int, default=2)
    p.add_argument("--voxel-size", type=float, default=0.015)
    p.add_argument("--max-points", type=int, default=450_000)
    p.add_argument("--no-semantics", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite-frames", action="store_true")
    p.add_argument("--overwrite-predictions", action="store_true")
    p.add_argument("--overwrite-semantics", action="store_true")
    p.add_argument("--query", default=None, help="Optional text query after fusion.")
    p.add_argument("--topk-percent", type=float, default=10.0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    concepts = [c.strip() for c in args.concepts.split(",")] if args.concepts else None
    result = run_pipeline(
        video_path=args.video,
        run_dir=Path(args.runs_root) / args.name,
        checkpoint=args.checkpoint,
        image_resolution=args.image_resolution,
        max_frames=args.max_frames,
        target_fps=args.fps,
        concepts=concepts,
        score_threshold=args.score_threshold,
        conf_percentile=args.conf_percentile,
        sample_stride=args.sample_stride,
        voxel_size=args.voxel_size,
        max_points=args.max_points,
        no_semantics=args.no_semantics,
        device=args.device,
        overwrite_frames=args.overwrite_frames,
        overwrite_predictions=args.overwrite_predictions,
        overwrite_semantics=args.overwrite_semantics,
    )
    if args.query:
        query_scene(result["run_dir"], args.query,
                    topk_percent=args.topk_percent, device=args.device)


if __name__ == "__main__":
    main()