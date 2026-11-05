"""Command line interface: ``paperbanana <command>`` (or ``python -m paperbanana``)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from . import __version__
from .config import USER_CONFIG_PATH, Settings, load_settings
from .data.bench import DATASET_ID, SPLITS, TASKS, bench_status, download_bench, load_test_cases
from .pipeline import GenerationRequest, Pipeline
from .providers import ProviderError
from .styles import ASPECT_RATIOS, IMAGE_SIZES, STYLE_PRESETS


def _settings_from_args(args: argparse.Namespace) -> Settings:
    s = load_settings(getattr(args, "config", None))
    overrides = {}
    for k in ("text_model", "image_model"):
        if getattr(args, k, None):
            overrides[k] = getattr(args, k)
    if getattr(args, "api_key", None):
        overrides[_key_name_for(getattr(args, "key_provider", None), overrides.get("image_model") or s.image_model)] = args.api_key
    return s.with_overrides(**overrides)


def _key_name_for(provider: str | None, model: str) -> str:
    from .providers.router import provider_for

    p = provider or provider_for(model)
    return {"gemini": "google_api_key", "openai": "openai_api_key", "openrouter": "openrouter_api_key"}.get(p, "google_api_key")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="path to a model_config.yaml (default: configs/model_config.yaml)")
    p.add_argument("--text-model", help="override the text/multimodal model (e.g. gemini-3.1-pro-preview, openrouter/openai/gpt-5)")
    p.add_argument("--image-model", help="override the image model (e.g. gemini-3.1-flash-image, gpt-image-2)")
    p.add_argument("--api-key", help="API key for the provider of --image-model (or --key-provider)")
    p.add_argument("--key-provider", choices=["gemini", "openai", "openrouter"], help="which provider --api-key belongs to")


def _add_generation_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--type", dest="figure_type", choices=["diagram", "plot"], default="diagram")
    p.add_argument("--aspect", choices=ASPECT_RATIOS, help="aspect ratio (default from config)")
    p.add_argument("--size", choices=IMAGE_SIZES, help="image size 1K/2K/4K (default from config)")
    p.add_argument("--style", choices=list(STYLE_PRESETS), help="style preset (default from config)")
    p.add_argument("--style-notes", default="", help="free-form style instructions")
    p.add_argument("--rounds", type=int, help="max Critic refinement rounds (default from config)")
    p.add_argument("--top-k", type=int, help="number of reference figures (0 disables retrieval)")
    p.add_argument("--no-stylist", action="store_true", help="skip the Stylist agent")
    p.add_argument("--no-critic", action="store_true", help="skip the Critic loop")
    p.add_argument("--style-refs", action="store_true", help="also pass reference figures to the image model as style references")
    p.add_argument("--out", help="copy the final PNG to this path")
    p.add_argument("--no-save", action="store_true", help="do not write a run folder under outputs/")


def _build_request(args: argparse.Namespace, s: Settings, method_text: str, intent: str, title: str = "") -> GenerationRequest:
    return GenerationRequest(
        method_text=method_text,
        intent=intent,
        figure_type=args.figure_type,
        aspect_ratio=args.aspect or s.default_aspect_ratio,
        image_size=args.size or s.default_image_size,
        style_preset=args.style or s.default_style,
        style_notes=args.style_notes,
        critic_rounds=s.critic_rounds if args.rounds is None else args.rounds,
        retrieval_top_k=s.retrieval_top_k if args.top_k is None else args.top_k,
        use_stylist=not args.no_stylist,
        use_critic=not args.no_critic,
        style_references_to_image_model=args.style_refs,
        text_model=s.text_model,
        image_model=s.image_model,
        save=not args.no_save,
        title=title,
    )


def _run(s: Settings, req: GenerationRequest, out: str | None) -> int:
    pipeline = Pipeline(s)
    result = None
    try:
        for ev in pipeline.run(req):
            if ev.stage == "done":
                result = ev.result
            else:
                print(f"[{ev.stage:9s}] {ev.message}", flush=True)
    except ProviderError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    assert result is not None
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(result.image)
        print(f"Final figure written to {out}")
    print()
    print(result.summary())
    return 0


# -- commands ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> int:
    s = _settings_from_args(args)
    if args.text_file:
        method_text = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text is not None:
        method_text = args.text
    elif not sys.stdin.isatty():
        try:
            method_text = sys.stdin.read()
        except OSError:
            method_text = ""
    else:
        print("error: provide --text, --text-file, or pipe the method text on stdin", file=sys.stderr)
        return 1
    req = _build_request(args, s, method_text, args.intent)
    return _run(s, req, args.out)


def cmd_bench(args: argparse.Namespace) -> int:
    s = _settings_from_args(args)
    cases = load_test_cases(s.data_dir, args.task, args.split)
    if not cases:
        print(f"No cached {args.task}/{args.split} split under {s.data_dir}. Run: paperbanana prepare-bench --tasks {args.task}", file=sys.stderr)
        return 1
    if args.id:
        chosen = [c for c in cases if c.id == args.id]
        if not chosen:
            print(f"id {args.id!r} not found", file=sys.stderr)
            return 1
    elif args.random:
        chosen = random.sample(cases, k=min(args.limit, len(cases)))
    else:
        chosen = cases[args.start : args.start + args.limit]
    rc = 0
    for case in chosen:
        print(f"\n=== {case.id} ({case.category}) ===\n{case.caption_short(200)}")
        args.aspect = args.aspect or (case.rounded_ratio if case.rounded_ratio in ASPECT_RATIOS else None)
        req = _build_request(args, s, case.content, case.visual_intent, title=case.id)
        out = str(Path(args.out_dir) / f"{case.id}.png") if args.out_dir else None
        rc |= _run(s, req, out)
    return rc


def cmd_prepare_bench(args: argparse.Namespace) -> int:
    s = _settings_from_args(args)
    print(f"Downloading {DATASET_ID} -> {s.data_dir}")
    counts = download_bench(s.data_dir, tasks=args.tasks, splits=args.splits, token=s.key("hf_token") or None, progress=print, max_rows=args.max_rows)
    print(json.dumps(counts, indent=2))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    s = _settings_from_args(args)
    print(f"PaperBanana {__version__}")
    print(f"config file : {s.config_path or f'(none, using template defaults; copy configs/model_config.template.yaml to {USER_CONFIG_PATH.name})'}")
    print(f"text model  : {s.text_model}")
    print(f"image model : {s.image_model}")
    print(f"providers with credentials: {', '.join(s.configured_providers()) or 'none'}")
    print(f"data dir    : {s.data_dir}")
    status = bench_status(s.data_dir)
    print(f"benchmark   : {json.dumps(status) if status else 'not downloaded (paperbanana prepare-bench)'}")
    print(f"output dir  : {s.output_dir}")
    try:
        from .providers.router import provider_for

        print(f"routing     : text -> {provider_for(s.text_model)}, image -> {provider_for(s.image_model)}")
    except ProviderError as exc:
        print(f"routing     : {exc}")
        return 1
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .webapp import launch

    launch(config_path=args.config, host=args.host, port=args.port, share=args.share)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paperbanana", description="Turn method text and captions into academic figures.")
    p.add_argument("--version", action="version", version=f"paperbanana {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate one figure from text")
    g.add_argument("--text", help="method paragraph (or data for a plot)")
    g.add_argument("--text-file", help="file containing the method text")
    g.add_argument("--intent", required=True, help="figure caption / visual intent")
    _add_generation_args(g)
    _add_model_args(g)
    g.set_defaults(func=cmd_generate)

    b = sub.add_parser("bench", help="run the pipeline on PaperBananaBench test cases")
    b.add_argument("--task", choices=TASKS, default="diagram")
    b.add_argument("--split", choices=SPLITS, default="test")
    b.add_argument("--id", help="run a single case by id (e.g. test_12)")
    b.add_argument("--start", type=int, default=0)
    b.add_argument("--limit", type=int, default=1)
    b.add_argument("--random", action="store_true", help="sample cases at random")
    b.add_argument("--out-dir", help="write <id>.png files here")
    _add_generation_args(b)
    _add_model_args(b)
    b.set_defaults(func=cmd_bench)

    d = sub.add_parser("prepare-bench", help=f"download {DATASET_ID} from the Hugging Face Hub")
    d.add_argument("--tasks", nargs="+", choices=TASKS, default=["diagram"])
    d.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    d.add_argument("--max-rows", type=int, help="only cache the first N rows per split (for quick tests)")
    _add_model_args(d)
    d.set_defaults(func=cmd_prepare_bench)

    c = sub.add_parser("check", help="print the effective configuration")
    _add_model_args(c)
    c.set_defaults(func=cmd_check)

    w = sub.add_parser("serve", help="launch the Gradio web app")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=7860)
    w.add_argument("--share", action="store_true", help="create a public Gradio share link")
    w.add_argument("--config")
    w.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
