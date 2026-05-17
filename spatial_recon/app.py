from __future__ import annotations

from pathlib import Path


def launch() -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("Install gradio to use the app: `pip install gradio`.") from exc

    from .cli import build_parser, run_pipeline

    def reconstruct(video, max_frames, conf_percentile, voxel_size):
        if video is None:
            return None, "Upload a video first."
        out = Path("runs/gradio_desk")
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--video",
                video,
                "--out",
                str(out),
                "--max-frames",
                str(int(max_frames)),
                "--conf-percentile",
                str(float(conf_percentile)),
                "--voxel-size",
                str(float(voxel_size)),
                "--overwrite-frames",
            ]
        )
        run_pipeline(args)
        return str(out / "exports" / "reconstruction_semantic.glb"), (out / "REPORT.md").read_text()

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

