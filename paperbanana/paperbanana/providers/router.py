"""Pick a provider implementation from a model name prefix."""

from __future__ import annotations

from ..config import Settings
from .base import ImageModel, ProviderError, TextModel

OPENROUTER_DEFAULT_URL = "https://openrouter.ai/api/v1"
OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"


def provider_for(model_name: str) -> str:
    """Return one of: gemini, openai, openrouter, local, ollama."""
    n = (model_name or "").strip().lower()
    if n.startswith("openrouter/"):
        return "openrouter"
    if n.startswith("local/"):
        return "local"
    if n.startswith("ollama/"):
        return "ollama"
    if n.startswith(("gemini", "google/")):
        return "gemini"
    if n.startswith(("openai/", "gpt-", "o1", "o3", "o4", "chatgpt", "dall-e")):
        return "openai"
    raise ProviderError(
        f"Cannot route model '{model_name}'. Use a gemini-* model, a gpt-* model, or a prefix such as "
        "openrouter/<model>, local/<model>, ollama/<model>."
    )


def _strip_prefix(name: str) -> str:
    return name.split("/", 1)[1] if "/" in name else name


def make_text_model(settings: Settings, model_name: str | None = None) -> TextModel:
    from .gemini import GeminiText
    from .openai_compat import OpenAICompatText

    name = model_name or settings.text_model
    provider = provider_for(name)
    t = settings.request_timeout
    if provider == "gemini":
        return GeminiText(_strip_prefix(name), settings.key("google_api_key"), t)
    if provider == "openai":
        return OpenAICompatText(_strip_prefix(name), settings.key("openai_api_key"), settings.url("openai_base_url") or None, t, "OpenAI")
    if provider == "openrouter":
        return OpenAICompatText(_strip_prefix(name), settings.key("openrouter_api_key"), settings.url("openrouter_base_url") or OPENROUTER_DEFAULT_URL, t, "OpenRouter")
    if provider == "local":
        url = settings.url("local_openai_base_url")
        if not url:
            raise ProviderError("local/<model> needs local_openai_base_url (e.g. http://127.0.0.1:8000/v1)")
        return OpenAICompatText(_strip_prefix(name), settings.key("local_openai_api_key") or "local", url, t, "Local server")
    if provider == "ollama":
        return OpenAICompatText(_strip_prefix(name), settings.key("ollama_api_key") or "ollama", settings.url("ollama_base_url") or OLLAMA_DEFAULT_URL, t, "Ollama")
    raise ProviderError(f"Unsupported text provider: {provider}")


def make_image_model(settings: Settings, model_name: str | None = None) -> ImageModel:
    from .gemini import GeminiImage
    from .openai_compat import OpenAIImage, OpenRouterImage

    name = model_name or settings.image_model
    provider = provider_for(name)
    t = settings.request_timeout
    if provider == "gemini":
        return GeminiImage(_strip_prefix(name), settings.key("google_api_key"), t)
    if provider == "openai":
        return OpenAIImage(_strip_prefix(name), settings.key("openai_api_key"), settings.url("openai_base_url") or None, t)
    if provider == "openrouter":
        return OpenRouterImage(_strip_prefix(name), settings.key("openrouter_api_key"), settings.url("openrouter_base_url") or OPENROUTER_DEFAULT_URL, t)
    raise ProviderError(f"Image generation is not supported for '{provider}' models. Use gemini-*, gpt-image-*, or openrouter/<model>.")
