"""Critic: scores a generated figure and writes revision instructions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..prompts import CRITIC_SYSTEM, PLOT_CRITIC_SYSTEM
from ..providers.base import ImagePart, Message, TextModel
from ..utils.images import downscale
from ..utils.text import extract_json, truncate

AXES = ("faithfulness", "conciseness", "readability", "aesthetics")
PASS_THRESHOLD = 7


@dataclass
class Verdict:
    passed: bool
    scores: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    revision_instructions: str = ""
    raw: str = ""

    @property
    def mean_score(self) -> float:
        vals = [v for v in self.scores.values() if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mean_score"] = self.mean_score
        return d

    @classmethod
    def from_raw(cls, raw: str) -> "Verdict":
        data = extract_json(raw)
        scores = {}
        for axis in AXES:
            try:
                scores[axis] = int(round(float((data.get("scores") or {}).get(axis, 0))))
            except (TypeError, ValueError):
                scores[axis] = 0
        passed = bool(data.get("passed", False)) and all(v >= PASS_THRESHOLD for v in scores.values())
        issues = [str(i) for i in (data.get("issues") or [])]
        return cls(passed=passed, scores=scores, issues=issues, revision_instructions=str(data.get("revision_instructions") or ""), raw=raw)


class Critic:
    def __init__(self, text_model: TextModel, *, figure_type: str = "diagram"):
        self.text_model = text_model
        self.system = CRITIC_SYSTEM if figure_type == "diagram" else PLOT_CRITIC_SYSTEM

    def review(self, image: bytes, *, method_text: str, intent: str, prompt: str, round_index: int = 1) -> Verdict:
        data, mime = downscale(image, max_side=1536, fmt="PNG")
        text = (
            f"REVIEW ROUND {round_index}\n\nINTENDED CAPTION:\n{intent.strip()}\n\n"
            f"METHOD TEXT / DATA:\n{truncate(method_text.strip(), 8000)}\n\n"
            f"GENERATION PROMPT / CODE USED:\n{truncate(prompt.strip(), 5000)}\n\n"
            "The generated figure is attached. Review it and return the JSON verdict."
        )
        raw = self.text_model.complete([Message.user(text, ImagePart(data, mime))], system=self.system, json_mode=True, temperature=0.2)
        return Verdict.from_raw(raw)
