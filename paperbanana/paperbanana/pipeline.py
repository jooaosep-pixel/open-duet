"""The PaperBanana pipeline: Retriever -> Planner -> Stylist -> Visualizer -> Critic (loop).

``Pipeline.run`` is a generator that yields ``StageEvent`` objects so the CLI and the web UI can
stream progress. The last event has ``stage == "done"`` and carries the ``GenerationResult``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .agents import Critic, Planner, PlotCoder, PlotRenderError, Retriever, Stylist, Verdict, Visualizer, run_plot_code
from .agents.stylist import plan_to_prompt
from .config import Settings
from .data.bench import Reference, ReferencePool
from .providers import ImageModel, ProviderError, TextModel, make_image_model, make_text_model
from .styles import style_guidelines
from .utils.text import slugify

STAGES = ("retrieve", "plan", "style", "visualize", "critic", "done")


@dataclass
class GenerationRequest:
    method_text: str
    intent: str
    figure_type: str = "diagram"  # "diagram" | "plot"
    aspect_ratio: str = "16:9"
    image_size: str = "1K"
    style_preset: str = "clean"
    style_notes: str = ""
    critic_rounds: int = 2
    retrieval_top_k: int = 3
    use_stylist: bool = True
    use_critic: bool = True
    style_references_to_image_model: bool = False
    text_model: str | None = None
    image_model: str | None = None
    save: bool = True
    title: str = ""

    def validate(self) -> None:
        if not self.method_text.strip():
            raise ValueError("Please paste the method text (or the data for a plot).")
        if not self.intent.strip():
            raise ValueError("Please provide the figure caption / visual intent.")
        if self.figure_type not in ("diagram", "plot"):
            raise ValueError("figure_type must be 'diagram' or 'plot'")


@dataclass
class RoundRecord:
    index: int
    image_path: str | None
    verdict: dict | None
    prompt: str
    elapsed_s: float


@dataclass
class GenerationResult:
    image: bytes
    image_path: Path | None
    prompt: str
    plan: str
    references: list[Reference]
    rounds: list[RoundRecord]
    final_verdict: Verdict | None
    code: str | None
    run_dir: Path | None
    elapsed_s: float
    figure_type: str
    text_model: str
    image_model: str

    def summary(self) -> str:
        lines = [f"{self.figure_type} generated with {self.image_model} (planned by {self.text_model}) in {self.elapsed_s:.0f}s"]
        if self.references:
            lines.append(f"{len(self.references)} reference figures used: " + ", ".join(r.id for r in self.references))
        if self.final_verdict:
            v = self.final_verdict
            status = "passed" if v.passed else "did not fully pass"
            lines.append(f"Critic {status} after {len(self.rounds)} round(s); mean score {v.mean_score}/10")
        if self.run_dir:
            lines.append(f"Saved to {self.run_dir}")
        return "\n".join(lines)


@dataclass
class StageEvent:
    stage: str
    status: str  # "start" | "done" | "info" | "warning"
    message: str
    image: bytes | None = None
    references: list[Reference] = field(default_factory=list)
    text: str = ""  # plan / prompt / code when relevant
    verdict: Verdict | None = None
    result: GenerationResult | None = None
    elapsed_s: float = 0.0


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        text_model: TextModel | None = None,
        image_model: ImageModel | None = None,
        pool: ReferencePool | None = None,
        load_pool: bool = True,
    ):
        self.settings = settings
        self._text_model = text_model
        self._image_model = image_model
        self._pool = pool
        self._load_pool = load_pool

    # -- lazy model construction -----------------------------------------------------
    def text_model(self, name: str | None = None) -> TextModel:
        if self._text_model is not None and (name is None or name == self.settings.text_model):
            return self._text_model
        return make_text_model(self.settings, name)

    def image_model(self, name: str | None = None) -> ImageModel:
        if self._image_model is not None and (name is None or name == self.settings.image_model):
            return self._image_model
        return make_image_model(self.settings, name)

    def pool(self, figure_type: str) -> ReferencePool | None:
        if self._pool is not None:
            return self._pool
        if not self._load_pool or not ReferencePool.is_available(self.settings.data_dir, figure_type):
            return None
        self._pool = ReferencePool.load(self.settings.data_dir, figure_type)
        return self._pool

    # -- main entry point ------------------------------------------------------------
    def run(self, req: GenerationRequest) -> Iterator[StageEvent]:
        req.validate()
        t0 = time.time()
        text_model = self.text_model(req.text_model)
        text_name = getattr(text_model, "name", req.text_model or self.settings.text_model)
        run_dir = self._make_run_dir(req) if req.save else None
        if run_dir:
            (run_dir / "request.json").write_text(json.dumps(asdict(req), indent=2, ensure_ascii=False), encoding="utf-8")

        # 1. Retrieve -------------------------------------------------------------
        yield StageEvent("retrieve", "start", "Retriever: searching PaperBananaBench for structurally similar figures ...")
        references: list[Reference] = []
        pool = self.pool(req.figure_type) if req.retrieval_top_k > 0 else None
        if pool is None or len(pool) == 0:
            msg = (
                "Retriever: reference pool not available, continuing zero-shot. Run `paperbanana prepare-bench` to enable retrieval."
                if req.retrieval_top_k > 0
                else "Retriever: disabled (top_k = 0)."
            )
            yield StageEvent("retrieve", "warning" if req.retrieval_top_k > 0 else "done", msg)
        else:
            references = Retriever(pool, text_model).retrieve(req.method_text, req.intent, req.retrieval_top_k)
            yield StageEvent(
                "retrieve", "done",
                f"Retriever: selected {len(references)} reference figure(s): " + ", ".join(r.id for r in references),
                references=references,
            )

        if req.figure_type == "plot":
            yield from self._run_plot(req, text_model, text_name, references, run_dir, t0)
            return

        # 2. Plan -----------------------------------------------------------------
        yield StageEvent("plan", "start", "Planner: writing the diagram specification ...")
        plan = Planner(text_model).plan(req.method_text, req.intent, references, aspect_ratio=req.aspect_ratio, notes=req.style_notes)
        self._save(run_dir, "plan.md", plan)
        yield StageEvent("plan", "done", "Planner: plan ready.", text=plan)

        # 3. Style ----------------------------------------------------------------
        guidelines = style_guidelines(req.style_preset, "diagram")
        if req.use_stylist:
            yield StageEvent("style", "start", "Stylist: deriving style guide and composing the image prompt ...")
            prompt = Stylist(text_model).refine(plan, references, style_guidelines=guidelines, extra_notes=req.style_notes, aspect_ratio=req.aspect_ratio)
            yield StageEvent("style", "done", "Stylist: final prompt ready.", text=prompt)
        else:
            prompt = plan_to_prompt(plan, guidelines + (" " + req.style_notes if req.style_notes.strip() else ""), req.aspect_ratio)
            yield StageEvent("style", "done", "Stylist: skipped (using plan directly).", text=prompt)
        self._save(run_dir, "prompt.txt", prompt)

        # 4. Visualize ------------------------------------------------------------
        image_model = self.image_model(req.image_model)
        image_name = getattr(image_model, "name", req.image_model or self.settings.image_model)
        visualizer = Visualizer(image_model)
        yield StageEvent("visualize", "start", f"Visualizer: generating the figure with {image_name} ({req.aspect_ratio}, {req.image_size}) ...")
        t_round = time.time()
        style_refs = [b for b in (r.image_bytes() for r in references) if b] if req.style_references_to_image_model else []
        image = visualizer.render(prompt, aspect_ratio=req.aspect_ratio, image_size=req.image_size, style_references=style_refs).data
        rounds: list[RoundRecord] = []
        path = self._save_image(run_dir, "round_1.png", image)
        rounds.append(RoundRecord(1, str(path) if path else None, None, prompt, time.time() - t_round))
        yield StageEvent("visualize", "done", "Visualizer: first draft generated.", image=image)

        # 5. Critic loop ----------------------------------------------------------
        verdict: Verdict | None = None
        if req.use_critic and req.critic_rounds > 0:
            critic = Critic(text_model, figure_type="diagram")
            for rnd in range(1, req.critic_rounds + 1):
                yield StageEvent("critic", "start", f"Critic: reviewing draft {rnd} ...")
                verdict = critic.review(image, method_text=req.method_text, intent=req.intent, prompt=prompt, round_index=rnd)
                rounds[-1].verdict = verdict.to_dict()
                score_str = ", ".join(f"{k[:5]} {v}" for k, v in verdict.scores.items())
                if verdict.passed:
                    yield StageEvent("critic", "done", f"Critic: draft {rnd} passed ({score_str}).", verdict=verdict)
                    break
                issues = "; ".join(verdict.issues[:4]) or "see revision instructions"
                yield StageEvent("critic", "done", f"Critic: draft {rnd} needs work ({score_str}). Issues: {issues}", verdict=verdict)
                if rnd == req.critic_rounds:
                    yield StageEvent("critic", "warning", "Critic: maximum rounds reached, keeping the latest draft.")
                    break
                yield StageEvent("visualize", "start", f"Visualizer: applying the Critic's revisions (round {rnd + 1}) ...")
                t_round = time.time()
                try:
                    image = visualizer.revise(image, verdict.revision_instructions, prompt, aspect_ratio=req.aspect_ratio, image_size=req.image_size).data
                except ProviderError as exc:
                    yield StageEvent("visualize", "warning", f"Visualizer: revision failed ({exc}); keeping the previous draft.")
                    break
                path = self._save_image(run_dir, f"round_{rnd + 1}.png", image)
                rounds.append(RoundRecord(rnd + 1, str(path) if path else None, None, verdict.revision_instructions, time.time() - t_round))
                yield StageEvent("visualize", "done", f"Visualizer: draft {rnd + 1} generated.", image=image)
        else:
            yield StageEvent("critic", "done", "Critic: skipped.")

        result = self._finish(req, image, prompt, plan, references, rounds, verdict, None, run_dir, t0, text_name, image_name)
        yield StageEvent("done", "done", result.summary(), image=image, result=result, elapsed_s=result.elapsed_s)

    # -- plot path -------------------------------------------------------------------
    def _run_plot(self, req, text_model, text_name, references, run_dir, t0) -> Iterator[StageEvent]:
        guidelines = style_guidelines(req.style_preset, "plot")
        coder = PlotCoder(text_model)
        yield StageEvent("plan", "start", "Planner/Visualizer: writing matplotlib code for the plot ...")
        code = coder.write_code(req.method_text, req.intent, aspect_ratio=req.aspect_ratio, style_guidelines=guidelines, notes=req.style_notes)
        yield StageEvent("plan", "done", "Code written.", text=code)
        yield StageEvent("style", "done", f"Stylist: style preset '{req.style_preset}' applied inside the code.", text=guidelines)

        rounds: list[RoundRecord] = []
        image: bytes | None = None
        verdict: Verdict | None = None
        max_rounds = max(1, req.critic_rounds + 1 if req.use_critic else 1)
        last_error: str | None = None
        for rnd in range(1, max_rounds + 1):
            yield StageEvent("visualize", "start", f"Visualizer: rendering plot (attempt {rnd}) ...")
            t_round = time.time()
            try:
                image = run_plot_code(code)
                last_error = None
            except PlotRenderError as exc:
                last_error = str(exc)
                yield StageEvent("visualize", "warning", f"Visualizer: code failed to run: {last_error[-400:]}")
                if rnd == max_rounds:
                    raise ProviderError(f"Plot code failed after {rnd} attempt(s): {last_error[-600:]}")
                code = coder.fix_code(code, "Make the script run without errors and keep the intended plot.", error=last_error)
                yield StageEvent("plan", "done", "Code repaired.", text=code)
                continue
            self._save(run_dir, f"round_{rnd}.py", code)
            path = self._save_image(run_dir, f"round_{rnd}.png", image)
            rounds.append(RoundRecord(rnd, str(path) if path else None, None, code, time.time() - t_round))
            yield StageEvent("visualize", "done", f"Visualizer: plot rendered (attempt {rnd}).", image=image)
            if not req.use_critic or rnd == max_rounds:
                if req.use_critic and rnd == max_rounds and rnd > 1:
                    yield StageEvent("critic", "warning", "Critic: maximum rounds reached, keeping the latest plot.")
                break
            yield StageEvent("critic", "start", f"Critic: reviewing plot {rnd} ...")
            verdict = Critic(text_model, figure_type="plot").review(image, method_text=req.method_text, intent=req.intent, prompt=code, round_index=rnd)
            rounds[-1].verdict = verdict.to_dict()
            score_str = ", ".join(f"{k[:5]} {v}" for k, v in verdict.scores.items())
            if verdict.passed:
                yield StageEvent("critic", "done", f"Critic: plot {rnd} passed ({score_str}).", verdict=verdict)
                break
            yield StageEvent("critic", "done", f"Critic: plot {rnd} needs work ({score_str}). " + "; ".join(verdict.issues[:4]), verdict=verdict)
            code = coder.fix_code(code, verdict.revision_instructions)
            yield StageEvent("plan", "done", "Code revised by the Critic's instructions.", text=code)

        assert image is not None
        result = self._finish(req, image, code, code, references, rounds, verdict, code, run_dir, t0, text_name, "matplotlib")
        yield StageEvent("done", "done", result.summary(), image=image, result=result, elapsed_s=result.elapsed_s)

    # -- persistence -----------------------------------------------------------------
    def _make_run_dir(self, req: GenerationRequest) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = slugify(req.title or req.intent.split(".")[0][:60])
        run_dir = self.settings.output_dir / f"{stamp}-{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _save(run_dir: Path | None, name: str, text: str) -> Path | None:
        if not run_dir:
            return None
        p = run_dir / name
        p.write_text(text, encoding="utf-8")
        return p

    @staticmethod
    def _save_image(run_dir: Path | None, name: str, data: bytes) -> Path | None:
        if not run_dir:
            return None
        p = run_dir / name
        p.write_bytes(data)
        return p

    def _finish(self, req, image, prompt, plan, references, rounds, verdict, code, run_dir, t0, text_name, image_name) -> GenerationResult:
        final_path = self._save_image(run_dir, "figure.png", image)
        if run_dir:
            meta = {
                "request": asdict(req),
                "text_model": text_name,
                "image_model": image_name,
                "references": [{"id": r.id, "caption": r.visual_intent, "score": r.score} for r in references],
                "rounds": [asdict(r) for r in rounds],
                "final_verdict": verdict.to_dict() if verdict else None,
                "elapsed_s": round(time.time() - t0, 1),
            }
            (run_dir / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return GenerationResult(
            image=image, image_path=final_path, prompt=prompt, plan=plan, references=references, rounds=rounds,
            final_verdict=verdict, code=code, run_dir=run_dir, elapsed_s=time.time() - t0, figure_type=req.figure_type,
            text_model=text_name, image_model=image_name,
        )


def generate(settings: Settings, req: GenerationRequest, *, log=print) -> GenerationResult:
    """Convenience wrapper: run the pipeline to completion, logging progress messages."""
    result: GenerationResult | None = None
    for ev in Pipeline(settings).run(req):
        if log and ev.stage != "done":
            log(f"[{ev.stage:9s}] {ev.message}")
        if ev.result is not None:
            result = ev.result
    assert result is not None
    return result
