"""
Wan2.2 - Gradio Preview Interface
Web UI for text-to-video, image-to-video, text-image-to-video generation.
"""
import os
import sys
import random
import tempfile
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import gradio as gr

TASKS = {
    "Text to Video (T2V-A14B)": "t2v-A14B",
    "Image to Video (I2V-A14B)": "i2v-A14B",
    "Text+Image to Video (TI2V-5B)": "ti2v-5B",
}

SIZE_OPTIONS = {
    "1280×720 (Landscape HD)": "1280*720",
    "720×1280 (Portrait HD)": "720*1280",
    "832×480 (Landscape)": "832*480",
    "480×832 (Portrait)": "480*832",
}

EXAMPLE_PROMPTS = {
    "t2v-A14B": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "i2v-A14B": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression.",
    "ti2v-5B": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
}

TI2V_SIZES = {"704×1280 (Portrait)": "704*1280", "1280×704 (Landscape)": "1280*704"}


def get_sizes_for_task(task_label: str) -> list:
    task = TASKS.get(task_label, "t2v-A14B")
    if task == "ti2v-5B":
        return list(TI2V_SIZES.keys())
    return list(SIZE_OPTIONS.keys())


def update_ui(task_label: str):
    task = TASKS.get(task_label, "t2v-A14B")
    show_image = task in ("i2v-A14B", "ti2v-5B")
    sizes = get_sizes_for_task(task_label)
    default_prompt = EXAMPLE_PROMPTS.get(task, "")
    return (
        gr.update(visible=show_image),
        gr.update(choices=sizes, value=sizes[0]),
        gr.update(value=default_prompt),
    )


def generate_video(
    task_label: str,
    ckpt_dir: str,
    prompt: str,
    image_path,
    size_label: str,
    frame_num: int,
    sample_steps: int,
    guide_scale: float,
    seed: int,
    offload_model: bool,
):
    task = TASKS.get(task_label, "t2v-A14B")

    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None, "❌ يرجى تحديد مسار نموذج صحيح (Checkpoint directory)."

    if not prompt.strip():
        return None, "❌ يرجى كتابة وصف للفيديو (Prompt)."

    if task in ("i2v-A14B", "ti2v-5B") and image_path is None:
        return None, "❌ يرجى رفع صورة لهذه المهمة."

    try:
        import torch
        import wan
        from wan.configs import (
            MAX_AREA_CONFIGS,
            SIZE_CONFIGS,
            WAN_CONFIGS,
        )
        from wan.utils.utils import save_video
        from PIL import Image
    except ImportError as e:
        return None, f"❌ خطأ في استيراد المكتبات: {e}"

    if task == "ti2v-5B":
        size_map = TI2V_SIZES
    else:
        size_map = SIZE_OPTIONS

    size_key = size_map.get(size_label, list(size_map.values())[0])
    cfg = WAN_CONFIGS[task]

    if seed < 0:
        seed = random.randint(0, sys.maxsize)

    actual_steps = sample_steps if sample_steps > 0 else cfg.sample_steps
    actual_scale = guide_scale if guide_scale > 0 else cfg.sample_guide_scale
    actual_frames = frame_num if frame_num > 0 else cfg.frame_num

    device = 0 if torch.cuda.is_available() else "cpu"

    try:
        img = None
        if image_path is not None:
            img = Image.open(image_path).convert("RGB")

        if task == "t2v-A14B":
            pipeline = wan.WanT2V(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                device_id=device,
                rank=0,
            )
            video = pipeline.generate(
                prompt,
                size=SIZE_CONFIGS[size_key],
                frame_num=actual_frames,
                sampling_steps=actual_steps,
                guide_scale=actual_scale,
                seed=seed,
                offload_model=offload_model,
            )
        elif task == "i2v-A14B":
            pipeline = wan.WanI2V(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                device_id=device,
                rank=0,
            )
            video = pipeline.generate(
                prompt,
                img,
                max_area=MAX_AREA_CONFIGS[size_key],
                frame_num=actual_frames,
                sampling_steps=actual_steps,
                guide_scale=actual_scale,
                seed=seed,
                offload_model=offload_model,
            )
        elif task == "ti2v-5B":
            pipeline = wan.WanTI2V(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                device_id=device,
                rank=0,
            )
            video = pipeline.generate(
                prompt,
                img=img,
                size=SIZE_CONFIGS[size_key],
                max_area=MAX_AREA_CONFIGS[size_key],
                frame_num=actual_frames,
                sampling_steps=actual_steps,
                guide_scale=actual_scale,
                seed=seed,
                offload_model=offload_model,
            )
        else:
            return None, f"❌ المهمة '{task}' غير مدعومة في هذه الواجهة."

        out_path = os.path.join(
            tempfile.gettempdir(),
            f"wan_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
        )
        save_video(
            tensor=video[None],
            save_file=out_path,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        return out_path, f"✅ تم إنشاء الفيديو بنجاح! البذرة المستخدمة: {seed}"

    except Exception as e:
        return None, f"❌ حدث خطأ أثناء التوليد: {e}"


def build_ui():
    with gr.Blocks(
        title="Wan2.2 - معاينة النموذج",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; padding: 20px 0 10px; }
        .header h1 { font-size: 2.2em; font-weight: 700; color: #5c6bc0; }
        .header p  { color: #666; margin-top: 4px; }
        .generate-btn { background: #5c6bc0 !important; color: white !important; }
        """,
    ) as demo:

        gr.HTML("""
        <div class="header">
          <h1>🎬 Wan2.2</h1>
          <p>نموذج توليد الفيديو المتقدم — Text-to-Video · Image-to-Video · Text+Image-to-Video</p>
        </div>
        """)

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────────
            with gr.Column(scale=1):
                task_dd = gr.Dropdown(
                    choices=list(TASKS.keys()),
                    value=list(TASKS.keys())[0],
                    label="المهمة / Task",
                )

                ckpt_input = gr.Textbox(
                    label="مسار النموذج / Checkpoint Directory",
                    placeholder="/path/to/Wan2.2-T2V-A14B",
                )

                prompt_box = gr.Textbox(
                    label="الوصف / Prompt",
                    lines=4,
                    placeholder="اكتب وصفاً للفيديو الذي تريد توليده...",
                    value=EXAMPLE_PROMPTS["t2v-A14B"],
                )

                image_upload = gr.Image(
                    label="الصورة المرجعية / Reference Image",
                    type="filepath",
                    visible=False,
                )

                with gr.Accordion("إعدادات متقدمة / Advanced Settings", open=False):
                    size_dd = gr.Dropdown(
                        choices=list(SIZE_OPTIONS.keys()),
                        value=list(SIZE_OPTIONS.keys())[0],
                        label="دقة الفيديو / Resolution",
                    )
                    frame_slider = gr.Slider(
                        minimum=0, maximum=129, step=4, value=0,
                        label="عدد الإطارات / Frame Count (0 = افتراضي)",
                    )
                    steps_slider = gr.Slider(
                        minimum=0, maximum=100, step=1, value=0,
                        label="خطوات التوليد / Sampling Steps (0 = افتراضي)",
                    )
                    scale_slider = gr.Slider(
                        minimum=0.0, maximum=20.0, step=0.5, value=0.0,
                        label="مقياس التوجيه / Guide Scale (0 = افتراضي)",
                    )
                    seed_num = gr.Number(
                        value=-1,
                        label="البذرة / Seed (-1 = عشوائي)",
                        precision=0,
                    )
                    offload_cb = gr.Checkbox(
                        value=True,
                        label="تفريغ النموذج من الذاكرة بعد كل خطوة (موصى به للذاكرة المحدودة)",
                    )

                gen_btn = gr.Button("🚀 توليد الفيديو", variant="primary", elem_classes="generate-btn")

            # ── Right column: output ─────────────────────────────────────
            with gr.Column(scale=1):
                video_out = gr.Video(label="الفيديو المُولَّد / Generated Video", height=420)
                status_box = gr.Textbox(label="الحالة / Status", interactive=False)

        # ── Examples ─────────────────────────────────────────────────────
        gr.Examples(
            examples=[
                [list(TASKS.keys())[0], "", EXAMPLE_PROMPTS["t2v-A14B"], None],
                [list(TASKS.keys())[1], "", EXAMPLE_PROMPTS["i2v-A14B"], "examples/i2v_input.JPG"],
                [list(TASKS.keys())[2], "", EXAMPLE_PROMPTS["ti2v-5B"], None],
            ],
            inputs=[task_dd, ckpt_input, prompt_box, image_upload],
            label="أمثلة جاهزة / Ready Examples",
        )

        # ── Event wiring ──────────────────────────────────────────────────
        task_dd.change(
            fn=update_ui,
            inputs=[task_dd],
            outputs=[image_upload, size_dd, prompt_box],
        )

        gen_btn.click(
            fn=generate_video,
            inputs=[
                task_dd, ckpt_input, prompt_box, image_upload,
                size_dd, frame_slider, steps_slider, scale_slider,
                seed_num, offload_cb,
            ],
            outputs=[video_out, status_box],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
