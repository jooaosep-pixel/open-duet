"""Planner: method text + caption (+ reference examples) -> detailed textual diagram plan."""

from __future__ import annotations

from ..data.bench import Reference
from ..prompts import PLANNER_SYSTEM
from ..providers.base import ImagePart, Message, TextModel
from ..utils.images import downscale
from ..utils.text import truncate

MAX_METHOD_CHARS = 12000


class Planner:
    def __init__(self, text_model: TextModel, *, include_reference_images: bool = True):
        self.text_model = text_model
        self.include_reference_images = include_reference_images

    def plan(self, method_text: str, intent: str, references: list[Reference], *, aspect_ratio: str, notes: str = "") -> str:
        parts: list = []
        if references:
            parts.append(
                "Here are reference figures from published papers. Study how they structure a methodology diagram "
                "(layout, grouping, label density). Each is given as caption then image."
            )
            for i, ref in enumerate(references, 1):
                parts.append(f"Reference {i} caption: {ref.caption_short(600)}")
                img = ref.image_bytes()
                if img and self.include_reference_images:
                    data, mime = downscale(img, max_side=1024)
                    parts.append(ImagePart(data, mime))
        parts.append(
            "Now plan the TARGET figure.\n\n"
            f"TARGET CAPTION / VISUAL INTENT:\n{intent.strip()}\n\n"
            f"TARGET METHOD TEXT:\n{truncate(method_text.strip(), MAX_METHOD_CHARS)}\n\n"
            f"Canvas aspect ratio: {aspect_ratio} (width:height)."
            + (f"\nAdditional author notes: {notes.strip()}" if notes.strip() else "")
            + "\n\nWrite the plan now."
        )
        return self.text_model.complete([Message.user(*parts)], system=PLANNER_SYSTEM, temperature=0.4).strip()
