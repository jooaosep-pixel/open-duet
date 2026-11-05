import pytest

from paperbanana.config import Settings
from paperbanana.providers import ProviderError, make_image_model, make_text_model, provider_for


@pytest.mark.parametrize(
    "name,expected",
    [
        ("gemini-3.1-pro-preview", "gemini"),
        ("gemini-3.1-flash-image", "gemini"),
        ("gpt-image-2", "openai"),
        ("gpt-5", "openai"),
        ("openai/gpt-5-mini", "openai"),
        ("openrouter/google/gemini-3-pro", "openrouter"),
        ("local/llama", "local"),
        ("ollama/qwen3", "ollama"),
    ],
)
def test_provider_for(name, expected):
    assert provider_for(name) == expected


def test_provider_for_unknown():
    with pytest.raises(ProviderError):
        provider_for("mystery-model")


def test_missing_key_is_a_clear_error():
    s = Settings(api_keys={}, base_urls={})
    with pytest.raises(ProviderError, match="Google API key"):
        make_text_model(s, "gemini-3.1-pro-preview")
    with pytest.raises(ProviderError, match="OpenAI"):
        make_image_model(s, "gpt-image-2")
    with pytest.raises(ProviderError, match="local_openai_base_url"):
        make_text_model(s, "local/llama")
    with pytest.raises(ProviderError, match="not supported"):
        make_image_model(s, "ollama/llava")


def test_models_construct_with_keys():
    s = Settings(api_keys={"google_api_key": "k", "openai_api_key": "k", "openrouter_api_key": "k"}, base_urls={})
    assert make_text_model(s, "gemini-3.1-pro-preview").name == "gemini-3.1-pro-preview"
    assert make_image_model(s, "gemini-3.1-flash-image").name == "gemini-3.1-flash-image"
    assert make_text_model(s, "openrouter/anthropic/claude-x").name == "anthropic/claude-x"
    assert make_image_model(s, "gpt-image-2").name == "gpt-image-2"
    assert make_image_model(s, "openrouter/google/gemini-3.1-flash-image").name == "google/gemini-3.1-flash-image"
