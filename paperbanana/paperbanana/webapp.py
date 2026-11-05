"""Gradio web interface for PaperBanana."""

from __future__ import annotations

import html
import io
import random
from pathlib import Path

import gradio as gr
from PIL import Image

from . import __version__
from .config import Settings, load_settings
from .data.bench import DATASET_ID, ReferencePool, bench_status, download_bench, load_test_cases
from .examples import EXAMPLES, PLOT_EXAMPLE
from .pipeline import GenerationRequest, Pipeline, StageEvent
from .providers import ProviderError
from .styles import ASPECT_RATIOS, IMAGE_SIZES, style_choices

KEY_PROVIDERS = {"Google Gemini": "google_api_key", "OpenAI": "openai_api_key", "OpenRouter": "openrouter_api_key"}
STAGE_LABELS = [("retrieve", "Retrieve"), ("plan", "Plan"), ("style", "Style"), ("visualize", "Generate"), ("critic", "Critique")]

CSS = """
.pb-header { display:flex; align-items:center; gap:14px; padding: 6px 0 2px; }
.pb-header h1 { margin:0; font-size: 1.7rem; letter-spacing:-0.01em; }
.pb-header p { margin:2px 0 0; opacity:.75; }
.pb-banana { font-size: 2.2rem; line-height:1; }
.pb-steps { display:flex; gap:8px; flex-wrap:wrap; margin: 2px 0 8px; }
.pb-step { padding: 5px 12px; border-radius: 999px; font-size: .82rem; font-weight: 600; border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued); background: var(--background-fill-secondary); }
.pb-step.active { border-color: #e8b40c; color: #7a5a00; background: #fff4c7; }
.pb-step.done { border-color: #2e8b57; color: #1d5c3a; background: #e3f5ea; }
.pb-step.warn { border-color: #d9822b; color: #7a3e00; background: #fdebd6; }
.dark .pb-step.active { color:#ffe08a; background:#4a3a00; border-color:#b58900; }
.dark .pb-step.done { color:#9fe2bf; background:#123b27; border-color:#2e8b57; }
.dark .pb-step.warn { color:#ffc896; background:#4a2a0a; border-color:#d9822b; }
.pb-log { display:flex; flex-direction:column-reverse; font-size: .86rem; line-height:1.45; max-height: 240px; overflow-y:auto; padding: 8px 10px; border: 1px solid var(--border-color-primary); border-radius: 8px; background: var(--background-fill-secondary); }
.pb-log div { white-space: pre-wrap; }
.pb-result img { object-fit: contain; }
.pb-score { display:inline-block; padding: 2px 8px; border-radius: 6px; background: var(--background-fill-secondary); margin-right:6px; font-size:.85rem; }
footer { display:none !important; }
"""


def _steps_html(state: dict[str, str]) -> str:
    pills = []
    for key, label in STAGE_LABELS:
        cls = state.get(key, "")
        icon = {"done": "✓ ", "active": "⟳ ", "warn": "! "}.get(cls, "")
        pills.append(f'<span class="pb-step {cls}">{icon}{label}</span>')
    return '<div class="pb-steps">' + "".join(pills) + "</div>"


def _log_html(lines: list[str]) -> str:
    # column-reverse keeps the newest line in view without any scripting: render newest first in the DOM.
    return '<div class="pb-log">' + "".join(f"<div>{html.escape(l)}</div>" for l in reversed(lines)) + "</div>"


def _verdict_md(verdict) -> str:
    if verdict is None:
        return "_Critic has not reviewed this figure yet._"
    scores = " ".join(f'<span class="pb-score">{k}: <b>{v}</b>/10</span>' for k, v in verdict.scores.items())
    status = "**Passed**" if verdict.passed else "**Needs revision**"
    issues = "\n".join(f"- {i}" for i in verdict.issues) or "- none"
    out = f"{status} · mean {verdict.mean_score}/10\n\n{scores}\n\n**Issues**\n{issues}"
    if verdict.revision_instructions:
        out += f"\n\n**Revision instructions**\n{verdict.revision_instructions}"
    return out


def _pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _gallery(references) -> list:
    items = []
    for r in references:
        b = r.image_bytes()
        if b:
            items.append((_pil(b), f"{r.id} · {r.caption_short(120)}"))
    return items


def build_app(settings: Settings) -> gr.Blocks:
    bench = bench_status(settings.data_dir)
    providers = settings.configured_providers()

    def config_banner() -> str:
        cfg = f"`{settings.config_path.name}`" if settings.config_path else "template defaults (no `configs/model_config.yaml` yet)"
        creds = ", ".join(providers) if providers else "none - enter an API key below"
        st = bench_status(settings.data_dir)
        b = ", ".join(f"{t}: " + "/".join(f"{s} {n}" for s, n in v.items()) for t, v in st.items()) or "not downloaded"
        return f"**Config:** {cfg} · **Credentials:** {creds} · **Benchmark cache:** {b}"

    with gr.Blocks(title="PaperBanana") as demo:
        gr.HTML(
            '<div class="pb-header"><span class="pb-banana">🍌</span><div><h1>PaperBanana</h1>'
            "<p>Paste a method paragraph and a caption. Five agents retrieve similar figures, plan, style, draw and critique "
            "a publication-ready illustration or plot.</p></div></div>"
        )
        banner = gr.Markdown(config_banner())

        with gr.Row(equal_height=False):
            # ------------------------------------------------------------ inputs
            with gr.Column(scale=5, min_width=380):
                with gr.Row():
                    example_dd = gr.Dropdown(
                        choices=[e["title"] for e in EXAMPLES] + [PLOT_EXAMPLE["title"]], label="Examples", value=None, scale=4, min_width=220
                    )
                    load_example_btn = gr.Button("Load example", scale=1, min_width=120)
                    load_bench_btn = gr.Button("Benchmark case", scale=1, min_width=130, interactive=bool(bench.get("diagram", {}).get("test")))
                method_text = gr.Textbox(
                    label="Method text (or data for a plot)",
                    placeholder="Paste the method section paragraph(s) from your paper ...",
                    lines=12, max_lines=30,
                )
                intent = gr.Textbox(
                    label="Figure caption / visual intent",
                    placeholder="Figure 1: Overview of our framework. ...",
                    lines=3,
                )
                with gr.Row():
                    figure_type = gr.Radio(choices=[("Diagram", "diagram"), ("Plot", "plot")], value="diagram", label="Figure type", scale=1)
                    aspect = gr.Dropdown(choices=ASPECT_RATIOS, value=settings.default_aspect_ratio, label="Aspect ratio", scale=1)
                    size = gr.Dropdown(choices=IMAGE_SIZES, value=settings.default_image_size, label="Resolution", info="4K needs gemini-3-pro-image", scale=1)
                with gr.Row():
                    style = gr.Dropdown(choices=style_choices("diagram"), value=settings.default_style, label="Style preset", scale=2, min_width=220)
                    style_notes = gr.Textbox(label="Style notes (optional)", placeholder="e.g. use our lab's blue/orange palette, label panels (a)-(c)", scale=3)
                with gr.Accordion("Models & API key", open=not providers):
                    with gr.Row():
                        key_provider = gr.Dropdown(choices=list(KEY_PROVIDERS), value="Google Gemini", label="API key provider", scale=1)
                        api_key = gr.Textbox(label="API key", type="password", placeholder="Leave empty to use the config file / environment", scale=2)
                    with gr.Row():
                        text_model = gr.Textbox(label="Text / planning model", value=settings.text_model)
                        image_model = gr.Textbox(label="Image model", value=settings.image_model)
                    gr.Markdown(
                        "Routing by prefix: `gemini-*` → Gemini, `gpt-*`/`gpt-image-*` → OpenAI, `openrouter/<model>` → OpenRouter, "
                        "`local/<model>` and `ollama/<model>` → local servers (text only). Keys typed here are used for this session only."
                    )
                with gr.Accordion("Pipeline options", open=False):
                    with gr.Row():
                        critic_rounds = gr.Slider(0, 4, value=settings.critic_rounds, step=1, label="Max critic rounds")
                        top_k = gr.Slider(0, 6, value=settings.retrieval_top_k, step=1, label="Reference figures (top-k)")
                    with gr.Row():
                        use_stylist = gr.Checkbox(value=True, label="Run Stylist agent")
                        style_refs = gr.Checkbox(value=False, label="Pass references to the image model as style examples")
                        save_run = gr.Checkbox(value=True, label="Save run to outputs/")
                generate_btn = gr.Button("Generate figure", variant="primary", size="lg")

            # ------------------------------------------------------------ outputs
            with gr.Column(scale=6, min_width=420):
                steps = gr.HTML(_steps_html({}))
                result_img = gr.Image(label="Generated figure", type="pil", interactive=False, height=460, elem_classes=["pb-result"], buttons=["download", "fullscreen"])
                summary_md = gr.Markdown("")
                log_html = gr.HTML(_log_html(["Ready. Paste your text and click Generate."]))
                with gr.Tabs():
                    with gr.Tab("Critic"):
                        verdict_md = gr.Markdown(_verdict_md(None))
                    with gr.Tab("Plan"):
                        plan_md = gr.Markdown("")
                    with gr.Tab("Image prompt / code"):
                        prompt_box = gr.Code(label="Final prompt or matplotlib code", language=None, lines=14)
                    with gr.Tab("References"):
                        refs_gallery = gr.Gallery(label="Retrieved reference figures (PaperBananaBench ref split)", columns=3, height=260, object_fit="contain")
                    with gr.Tab("Drafts"):
                        drafts_gallery = gr.Gallery(label="Every draft, in order", columns=3, height=260, object_fit="contain")
                    with gr.Tab("Benchmark"):
                        gr.Markdown(f"Downloads **{DATASET_ID}** from the Hugging Face Hub into `{settings.data_dir}`. The `ref` split powers retrieval; the `test` split gives you ready-made cases.")
                        with gr.Row():
                            bench_tasks = gr.CheckboxGroup(choices=["diagram", "plot"], value=["diagram"], label="Subsets")
                            bench_btn = gr.Button("Download benchmark")
                        bench_log = gr.Markdown("")
                        gt_img = gr.Image(label="Ground-truth figure of the loaded benchmark case", type="pil", interactive=False, height=220)

        # ------------------------------------------------------------ callbacks
        def load_example(title):
            for e in EXAMPLES:
                if e["title"] == title:
                    return e["method_text"], e["intent"], "diagram"
            if title == PLOT_EXAMPLE["title"]:
                return PLOT_EXAMPLE["method_text"], PLOT_EXAMPLE["intent"], "plot"
            return gr.skip(), gr.skip(), gr.skip()

        load_example_btn.click(load_example, inputs=example_dd, outputs=[method_text, intent, figure_type])

        def load_bench_case():
            cases = load_test_cases(settings.data_dir, "diagram", "test")
            if not cases:
                raise gr.Error("Benchmark not downloaded yet. Use the Benchmark tab.")
            c = random.choice(cases)
            gt = _pil(c.image_bytes()) if c.image_bytes() else None
            ratio = c.rounded_ratio if c.rounded_ratio in ASPECT_RATIOS else settings.default_aspect_ratio
            return c.content, c.visual_intent, "diagram", ratio, gt, f"Loaded **{c.id}** ({c.category}). Ground truth shown in the Benchmark tab."

        load_bench_btn.click(load_bench_case, outputs=[method_text, intent, figure_type, aspect, gt_img, summary_md])

        def download(tasks, progress=gr.Progress()):
            if not tasks:
                raise gr.Error("Pick at least one subset.")
            lines: list[str] = []

            def log(msg):
                lines.append(msg)
                progress(0, desc=msg[:80])

            try:
                counts = download_bench(settings.data_dir, tasks=tasks, token=settings.key("hf_token") or None, progress=log)
            except Exception as exc:  # noqa: BLE001
                raise gr.Error(f"Download failed: {exc}") from exc
            return (
                "Done: " + ", ".join(f"{t} " + "/".join(f"{s} {n}" for s, n in v.items()) for t, v in counts.items()),
                config_banner(),
                gr.Button(interactive=True),
            )

        bench_btn.click(download, inputs=bench_tasks, outputs=[bench_log, banner, load_bench_btn])

        def on_type_change(ftype):
            return gr.Dropdown(choices=style_choices(ftype))

        figure_type.change(on_type_change, inputs=figure_type, outputs=style)

        def generate(method_text_v, intent_v, ftype, aspect_v, size_v, style_v, notes_v, key_provider_v, api_key_v,
                     text_model_v, image_model_v, rounds_v, top_k_v, use_stylist_v, style_refs_v, save_v):
            overrides = {"text_model": text_model_v.strip(), "image_model": image_model_v.strip()}
            if api_key_v and api_key_v.strip():
                overrides[KEY_PROVIDERS[key_provider_v]] = api_key_v.strip()
            s = settings.with_overrides(**overrides)
            req = GenerationRequest(
                method_text=method_text_v or "", intent=intent_v or "", figure_type=ftype, aspect_ratio=aspect_v, image_size=size_v,
                style_preset=style_v, style_notes=notes_v or "", critic_rounds=int(rounds_v), retrieval_top_k=int(top_k_v),
                use_stylist=bool(use_stylist_v), use_critic=int(rounds_v) > 0, style_references_to_image_model=bool(style_refs_v),
                text_model=s.text_model, image_model=s.image_model, save=bool(save_v),
            )
            try:
                req.validate()
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

            state: dict[str, str] = {}
            lines: list[str] = []
            drafts: list = []
            updates = {
                steps: _steps_html(state), log_html: _log_html(["Starting ..."]), summary_md: "", verdict_md: _verdict_md(None),
                plan_md: "", prompt_box: "", refs_gallery: [], drafts_gallery: [], result_img: None,
            }
            yield updates
            try:
                for ev in Pipeline(s).run(req):
                    yield from _apply_event(ev, state, lines, drafts)
            except ProviderError as exc:
                lines.append(f"✗ {exc}")
                for k, v in state.items():
                    if v == "active":
                        state[k] = "warn"
                yield {steps: _steps_html(state), log_html: _log_html(lines), summary_md: f"**Error:** {exc}"}
                raise gr.Error(str(exc)) from exc
            except Exception as exc:  # noqa: BLE001
                lines.append(f"✗ unexpected error: {exc}")
                yield {log_html: _log_html(lines), summary_md: f"**Error:** {exc}"}
                raise gr.Error(f"Unexpected error: {exc}") from exc

        def _apply_event(ev: StageEvent, state, lines, drafts):
            out: dict = {}
            if ev.stage != "done":
                if ev.status == "start":
                    state[ev.stage] = "active"
                elif ev.status == "warning":
                    state[ev.stage] = "warn"
                else:
                    state[ev.stage] = "done"
                prefix = {"start": "…", "done": "✓", "warning": "!", "info": "·"}[ev.status]
                lines.append(f"{prefix} {ev.message}")
                out[steps] = _steps_html(state)
                out[log_html] = _log_html(lines)
            if ev.references:
                out[refs_gallery] = _gallery(ev.references)
            if ev.text and ev.stage == "plan":
                out[plan_md] = ev.text if not ev.text.lstrip().startswith(("import", "from", "#")) else f"```python\n{ev.text}\n```"
                if ev.text.lstrip().startswith(("import", "from")):
                    out[prompt_box] = ev.text
            if ev.text and ev.stage == "style":
                out[prompt_box] = ev.text
            if ev.image is not None:
                img = _pil(ev.image)
                out[result_img] = img
                if ev.stage == "visualize":
                    drafts.append((img, f"Draft {len(drafts) + 1}"))
                    out[drafts_gallery] = list(drafts)
            if ev.verdict is not None:
                out[verdict_md] = _verdict_md(ev.verdict)
            if ev.stage == "done" and ev.result is not None:
                r = ev.result
                for k in state:
                    if state[k] == "active":
                        state[k] = "done"
                out[steps] = _steps_html(state)
                out[summary_md] = "✅ " + r.summary().replace("\n", "  \n")
                out[prompt_box] = r.code or r.prompt
                lines.append(f"✓ Done in {r.elapsed_s:.0f}s" + (f" · saved to {r.run_dir}" if r.run_dir else ""))
                out[log_html] = _log_html(lines)
            yield out

        generate_btn.click(
            generate,
            inputs=[method_text, intent, figure_type, aspect, size, style, style_notes, key_provider, api_key, text_model, image_model,
                    critic_rounds, top_k, use_stylist, style_refs, save_run],
            outputs=[steps, log_html, summary_md, verdict_md, plan_md, prompt_box, refs_gallery, drafts_gallery, result_img],
            show_progress="minimal",
        )
    return demo


def launch(config_path: str | Path | None = None, host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    settings = load_settings(config_path)
    demo = build_app(settings)
    print(f"PaperBanana {__version__} · text={settings.text_model} · image={settings.image_model}")
    demo.queue(default_concurrency_limit=2).launch(
        server_name=host, server_port=port, share=share, show_error=True,
        theme=gr.themes.Soft(primary_hue="amber", neutral_hue="slate"), css=CSS,
    )


if __name__ == "__main__":  # pragma: no cover
    launch()
