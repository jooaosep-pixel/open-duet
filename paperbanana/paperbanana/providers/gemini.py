"""Google Gemini provider (google-genai SDK)."""

from __future__ import annotations

from typing import Sequence

from .base import ImagePart, ImageResult, Message, ProviderError

try:  # imported lazily so the package works without the SDK installed
    from google import genai
    from google.genai import types as gtypes
except ImportError:  # pragma: no cover
    genai = None
    gtypes = None


def _client(api_key: str, timeout: int):
    if genai is None:
        raise ProviderError("google-genai is not installed. Run: pip install google-genai")
    if not api_key:
        raise ProviderError("No Google API key. Set google_api_key in configs/model_config.yaml, GOOGLE_API_KEY, or enter it in the UI.")
    return genai.Client(api_key=api_key, http_options=gtypes.HttpOptions(timeout=timeout * 1000))


def _to_contents(messages: Sequence[Message]):
    contents = []
    for m in messages:
        parts = []
        for p in m.parts:
            if isinstance(p, str):
                parts.append(gtypes.Part.from_text(text=p))
            else:
                parts.append(gtypes.Part.from_bytes(data=p.data, mime_type=p.mime))
        contents.append(gtypes.Content(role="user" if m.role == "user" else "model", parts=parts))
    return contents


class GeminiText:
    def __init__(self, name: str, api_key: str, timeout: int = 180):
        self.name = name
        self._client = _client(api_key, timeout)

    def complete(self, messages, *, system=None, json_mode=False, temperature=None) -> str:
        config = gtypes.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        try:
            resp = self._client.models.generate_content(model=self.name, contents=_to_contents(messages), config=config)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Gemini text request failed ({self.name}): {exc}") from exc
        text = getattr(resp, "text", None)
        if not text:
            raise ProviderError(f"Gemini returned no text ({self.name}). Finish reason: {_finish_reason(resp)}")
        return text


class GeminiImage:
    def __init__(self, name: str, api_key: str, timeout: int = 180):
        self.name = name
        self._client = _client(api_key, timeout)

    def generate(self, prompt, *, aspect_ratio="16:9", image_size="1K", images: Sequence[ImagePart] = ()) -> ImageResult:
        parts = [gtypes.Part.from_bytes(data=im.data, mime_type=im.mime) for im in images]
        parts.append(gtypes.Part.from_text(text=prompt))
        contents = [gtypes.Content(role="user", parts=parts)]

        attempts = [
            gtypes.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size),
            gtypes.ImageConfig(aspect_ratio=aspect_ratio),  # older models reject image_size
        ]
        last_exc: Exception | None = None
        for image_config in attempts:
            config = gtypes.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"], image_config=image_config)
            try:
                resp = self._client.models.generate_content(model=self.name, contents=contents, config=config)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if "image_size" in str(exc).lower() or "imagesize" in str(exc).lower():
                    continue
                raise ProviderError(f"Gemini image request failed ({self.name}): {exc}") from exc
            return _extract_image(resp, self.name)
        raise ProviderError(f"Gemini image request failed ({self.name}): {last_exc}")


def _finish_reason(resp) -> str:
    try:
        return str(resp.candidates[0].finish_reason)
    except Exception:  # noqa: BLE001
        return "unknown"


def _extract_image(resp, model_name: str) -> ImageResult:
    text_chunks: list[str] = []
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return ImageResult(data=inline.data, mime=inline.mime_type or "image/png", text="\n".join(text_chunks))
            if getattr(part, "text", None):
                text_chunks.append(part.text)
    detail = " ".join(text_chunks).strip()
    raise ProviderError(
        f"Gemini ({model_name}) returned no image. Finish reason: {_finish_reason(resp)}."
        + (f" Model said: {detail[:300]}" if detail else "")
    )
