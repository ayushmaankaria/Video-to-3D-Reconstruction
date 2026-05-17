from __future__ import annotations

from pathlib import Path
from argparse import Namespace


def launch() -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("Install gradio to use the app: `pip install gradio`.") from exc

    from .cli import run_pipeline

    def reconstruct(video, max_frames, conf_percentile, voxel_size):
        if video is None:
            return None, "Upload a video first."
        out = Path("runs/gradio_desk")
        args = Namespace(
            video=video,
            images_dir=None,
            out=str(out),
            max_frames=int(max_frames),
            frame_mode="hybrid",
            image_size=1280,
            checkpoint="facebook/VGGT-1B",
            semantic_model="facebook/mask2former-swin-large-ade-semantic",
            device="auto",
            conf_percentile=float(conf_percentile),
            sample_stride=2,
            voxel_size=float(voxel_size),
            max_points=450_000,
            use_point_map=False,
            no_semantics=False,
            overwrite_frames=True,
            overwrite_predictions=False,
            overwrite_semantics=False,
            command="run",
        )
        run_pipeline(args)
        glb = out / "exports" / "reconstruction_semantic.glb"
        report = (out / "REPORT.md").read_text()
        if not glb.exists():
            return None, report + "\n\nGLB export was not created. Install `trimesh` to enable Model3D output."
        return str(glb), report

    with gr.Blocks(title="VGGT Semantic Desk Reconstruction") as demo:
        gr.Markdown("# VGGT Semantic Desk Reconstruction")
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Phone video")
                max_frames = gr.Slider(8, 48, value=24, step=1, label="Frames")
                conf = gr.Slider(0, 80, value=35, step=1, label="Confidence percentile")
                voxel = gr.Slider(0.0, 0.05, value=0.015, step=0.005, label="Voxel size")
                btn = gr.Button("Reconstruct", variant="primary")
            with gr.Column():
                model = gr.Model3D(label="Semantic reconstruction")
                report = gr.Markdown()
        btn.click(reconstruct, [video, max_frames, conf, voxel], [model, report])
    demo.launch(share=True)


if __name__ == "__main__":
    launch()
