"""Stylist: plan + style preset + references -> final image-generation prompt."""

from __future__ import annotations

from ..data.bench import Reference
from ..prompts import STYLIST_SYSTEM
from ..providers.base import ImagePart, Message, TextModel
from ..utils.images import downscale


class Stylist:
    def __init__(self, text_model: TextModel, *, include_reference_images: bool = True):
        self.text_model = text_model
        self.include_reference_images = include_reference_images

    def refine(self, plan: str, references: list[Reference], *, style_guidelines: str, extra_notes: str = "", aspect_ratio: str) -> str:
        parts: list = []
        if references and self.include_reference_images:
            parts.append("Reference figures (for style only, do not copy their content):")
            for ref in references[:3]:
                img = ref.image_bytes()
                if img:
                    data, mime = downscale(img, max_side=768)
                    parts.append(ImagePart(data, mime))
        parts.append(
            f"REQUESTED STYLE PRESET:\n{style_guidelines}\n"
            + (f"\nAUTHOR STYLE NOTES:\n{extra_notes.strip()}\n" if extra_notes.strip() else "")
            + f"\nCANVAS: aspect ratio {aspect_ratio}.\n\nDIAGRAM PLAN:\n{plan}\n\nProduce the final image prompt."
        )
        return self.text_model.complete([Message.user(*parts)], system=STYLIST_SYSTEM, temperature=0.3).strip()


def plan_to_prompt(plan: str, style_guidelines: str, aspect_ratio: str) -> str:
    """Fallback used when the Stylist is disabled: wrap the plan into a usable image prompt."""
    return (
        "Draw a publication-quality methodology diagram for a machine learning paper, following this plan exactly.\n\n"
        f"{plan}\n\nStyle: {style_guidelines}\n"
        f"Constraints: flat vector illustration, aspect ratio {aspect_ratio}, white background, crisp legible sans-serif text "
        "with no misspellings, no watermark, no photographic textures, nothing beyond the plan."
    )
