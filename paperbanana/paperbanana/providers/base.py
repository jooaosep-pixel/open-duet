"""Provider-agnostic interfaces for text (multimodal) and image models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, Union


class ProviderError(RuntimeError):
    """Raised for configuration problems or unrecoverable API failures."""


@dataclass
class ImagePart:
    data: bytes
    mime: str = "image/png"


Part = Union[str, ImagePart]


@dataclass
class Message:
    role: str  # "user" | "assistant"
    parts: list[Part] = field(default_factory=list)

    @classmethod
    def user(cls, *parts: Part) -> "Message":
        return cls("user", list(parts))

    @classmethod
    def assistant(cls, *parts: Part) -> "Message":
        return cls("assistant", list(parts))

    def text(self) -> str:
        return "\n".join(p for p in self.parts if isinstance(p, str))


@dataclass
class ImageResult:
    data: bytes
    mime: str = "image/png"
    text: str = ""  # any accompanying text the model returned


class TextModel(Protocol):
    name: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str: ...


class ImageModel(Protocol):
    name: str

    def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: str = "16:9",
        image_size: str = "1K",
        images: Sequence[ImagePart] = (),
    ) -> ImageResult:
        """Generate an image. ``images`` are optional inputs (references or a previous draft to edit)."""
        ...
