"""OpenAI and OpenAI-compatible providers (OpenAI, OpenRouter, local servers, Ollama)."""

from __future__ import annotations

import base64
from typing import Sequence

from ..utils.images import data_uri, decode_data_uri
from .base import ImagePart, ImageResult, Message, ProviderError

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


def _client(api_key: str, base_url: str | None, timeout: int, label: str):
    if OpenAI is None:
        raise ProviderError("openai is not installed. Run: pip install openai")
    if not api_key:
        raise ProviderError(f"No API key configured for {label}. Add it to configs/model_config.yaml, the environment, or the UI.")
    kwargs = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _to_chat_messages(messages: Sequence[Message], system: str | None):
    out = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        content = []
        for p in m.parts:
            if isinstance(p, str):
                content.append({"type": "text", "text": p})
            else:
                content.append({"type": "image_url", "image_url": {"url": data_uri(p.data, p.mime)}})
        if len(content) == 1 and content[0]["type"] == "text":
            out.append({"role": m.role, "content": content[0]["text"]})
        else:
            out.append({"role": m.role, "content": content})
    return out


class OpenAICompatText:
    def __init__(self, name: str, api_key: str, base_url: str | None = None, timeout: int = 180, label: str = "OpenAI"):
        self.name = name
        self.label = label
        self._client = _client(api_key, base_url, timeout, label)

    def complete(self, messages, *, system=None, json_mode=False, temperature=None) -> str:
        kwargs = {"model": self.name, "messages": _to_chat_messages(messages, system)}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if json_mode and "response_format" in str(exc):
                kwargs.pop("response_format")
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception as exc2:  # noqa: BLE001
                    raise ProviderError(f"{self.label} text request failed ({self.name}): {exc2}") from exc2
            else:
                raise ProviderError(f"{self.label} text request failed ({self.name}): {exc}") from exc
        try:
            text = resp.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise ProviderError(f"{self.label} returned an unexpected response for {self.name}") from exc
        if not text:
            raise ProviderError(f"{self.label} returned empty text for {self.name}")
        return text


_OPENAI_SIZE = {  # aspect ratio -> closest supported gpt-image size
    "1:1": "1024x1024",
    "16:9": "1536x1024",
    "21:9": "1536x1024",
    "3:2": "1536x1024",
    "4:3": "1536x1024",
    "3:4": "1024x1536",
    "9:16": "1024x1536",
}
_OPENAI_QUALITY = {"1K": "medium", "2K": "high", "4K": "high"}


class OpenAIImage:
    """OpenAI Images API (gpt-image-2, gpt-image-1, ...). Uses /edits when input images are supplied."""

    def __init__(self, name: str, api_key: str, base_url: str | None = None, timeout: int = 180):
        self.name = name
        self._client = _client(api_key, base_url, timeout, "OpenAI")

    def generate(self, prompt, *, aspect_ratio="16:9", image_size="1K", images: Sequence[ImagePart] = ()) -> ImageResult:
        size = _OPENAI_SIZE.get(aspect_ratio, "1536x1024")
        kwargs = {"model": self.name, "prompt": prompt, "size": size, "n": 1}
        if self.name.startswith("gpt-image"):
            kwargs["quality"] = _OPENAI_QUALITY.get(image_size, "medium")
        try:
            if images:
                files = [(f"ref{i}.png", im.data, im.mime) for i, im in enumerate(images)]
                resp = self._client.images.edit(image=files if len(files) > 1 else files[0], **kwargs)
            else:
                resp = self._client.images.generate(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenAI image request failed ({self.name}): {exc}") from exc
        item = resp.data[0]
        if getattr(item, "b64_json", None):
            return ImageResult(data=base64.b64decode(item.b64_json), mime="image/png", text=getattr(item, "revised_prompt", "") or "")
        raise ProviderError("OpenAI image response contained no base64 image data")


class OpenRouterImage:
    """Image output through OpenRouter chat completions (``modalities: ["image", "text"]``)."""

    def __init__(self, name: str, api_key: str, base_url: str, timeout: int = 180):
        self.name = name
        self._client = _client(api_key, base_url, timeout, "OpenRouter")

    def generate(self, prompt, *, aspect_ratio="16:9", image_size="1K", images: Sequence[ImagePart] = ()) -> ImageResult:
        messages = [Message.user(*images, f"{prompt}\n\nAspect ratio: {aspect_ratio}. Resolution: {image_size}.")]
        try:
            resp = self._client.chat.completions.create(
                model=self.name,
                messages=_to_chat_messages(messages, None),
                extra_body={"modalities": ["image", "text"], "image_config": {"aspect_ratio": aspect_ratio, "image_size": image_size}},
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenRouter image request failed ({self.name}): {exc}") from exc
        msg = resp.choices[0].message
        imgs = getattr(msg, "images", None) or (msg.model_extra or {}).get("images") or []
        for im in imgs:
            url = im.get("image_url", {}).get("url") if isinstance(im, dict) else getattr(getattr(im, "image_url", None), "url", None)
            if url and url.startswith("data:"):
                mime = url[5:].split(";", 1)[0] or "image/png"
                return ImageResult(data=decode_data_uri(url), mime=mime, text=msg.content or "")
        raise ProviderError(f"OpenRouter model {self.name} returned no image. Response text: {(msg.content or '')[:200]}")
