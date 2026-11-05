"""Visualizer: prompt -> image, and revision of a previous draft."""

from __future__ import annotations

from typing import Sequence

from ..providers.base import ImageModel, ImagePart, ImageResult
from ..utils.images import downscale, ensure_png


class Visualizer:
    def __init__(self, image_model: ImageModel):
        self.image_model = image_model

    def render(self, prompt: str, *, aspect_ratio: str, image_size: str, style_references: Sequence[bytes] = ()) -> ImageResult:
        images = []
        if style_references:
            prompt = (
                "The attached images are style references from published papers: match their visual style "
                "(palette, line weight, typography, layout density) but draw ONLY the content described below.\n\n" + prompt
            )
            for ref in style_references[:3]:
                data, mime = downscale(ref, max_side=1024)
                images.append(ImagePart(data, mime))
        result = self.image_model.generate(prompt, aspect_ratio=aspect_ratio, image_size=image_size, images=images)
        return ImageResult(ensure_png(result.data), "image/png", result.text)

    def revise(self, previous: bytes, instructions: str, original_prompt: str, *, aspect_ratio: str, image_size: str) -> ImageResult:
        prompt = (
            "Edit the attached academic figure. Apply ONLY the following revisions and keep everything else identical "
            "(layout, colours, labels, style):\n\n"
            f"{instructions.strip()}\n\n"
            "For context, the figure was originally specified as:\n"
            f"{original_prompt.strip()}\n\n"
            "Output the corrected figure with crisp, correctly spelled text on a white background."
        )
        result = self.image_model.generate(
            prompt, aspect_ratio=aspect_ratio, image_size=image_size, images=[ImagePart(previous, "image/png")]
        )
        return ImageResult(ensure_png(result.data), "image/png", result.text)
