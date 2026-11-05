"""Configuration loading: template defaults < configs/model_config.yaml < environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACKAGE_ROOT / "configs"
TEMPLATE_PATH = CONFIG_DIR / "model_config.template.yaml"
USER_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"

# yaml key -> environment variable names (first non-empty wins)
KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "google_api_key": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openai_api_key": ("OPENAI_API_KEY",),
    "openrouter_api_key": ("OPENROUTER_API_KEY",),
    "local_openai_api_key": ("LOCAL_OPENAI_API_KEY",),
    "ollama_api_key": ("OLLAMA_API_KEY",),
    "hf_token": ("HF_TOKEN",),
}
URL_ENV_VARS: dict[str, tuple[str, ...]] = {
    "openrouter_base_url": ("OPENROUTER_BASE_URL",),
    "openai_base_url": ("OPENAI_BASE_URL",),
    "local_openai_base_url": ("LOCAL_OPENAI_BASE_URL",),
    "ollama_base_url": ("OLLAMA_BASE_URL",),
}


@dataclass
class Settings:
    text_model: str = "gemini-3.1-pro-preview"
    image_model: str = "gemini-3.1-flash-image"
    critic_rounds: int = 2
    retrieval_top_k: int = 3
    default_aspect_ratio: str = "16:9"
    default_image_size: str = "1K"
    default_style: str = "clean"
    request_timeout: int = 180
    api_keys: dict[str, str] = field(default_factory=dict)
    base_urls: dict[str, str] = field(default_factory=dict)
    data_dir: Path = PACKAGE_ROOT / "data"
    output_dir: Path = PACKAGE_ROOT / "outputs"
    config_path: Path | None = None

    # -- helpers -------------------------------------------------------------------
    def key(self, name: str) -> str:
        return (self.api_keys.get(name) or "").strip()

    def url(self, name: str) -> str:
        return (self.base_urls.get(name) or "").strip()

    def with_overrides(self, **kwargs: Any) -> "Settings":
        """Return a copy with non-empty overrides applied (used by the UI and CLI)."""
        api_keys = dict(self.api_keys)
        base_urls = dict(self.base_urls)
        plain: dict[str, Any] = {}
        for k, v in kwargs.items():
            if v is None or v == "":
                continue
            if k in KEY_ENV_VARS:
                api_keys[k] = str(v)
            elif k in URL_ENV_VARS:
                base_urls[k] = str(v)
            else:
                plain[k] = v
        return replace(self, api_keys=api_keys, base_urls=base_urls, **plain)

    def configured_providers(self) -> list[str]:
        names = []
        if self.key("google_api_key"):
            names.append("Google Gemini")
        if self.key("openai_api_key"):
            names.append("OpenAI")
        if self.key("openrouter_api_key"):
            names.append("OpenRouter")
        if self.url("local_openai_base_url"):
            names.append("Local OpenAI-compatible")
        if self.url("ollama_base_url"):
            names.append("Ollama")
        return names


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out


def _env(*names: str) -> str:
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return ""


def load_settings(config_path: str | Path | None = None, *, use_env: bool = True) -> Settings:
    """Load settings. ``config_path`` defaults to configs/model_config.yaml when it exists."""
    raw = _read_yaml(TEMPLATE_PATH)
    chosen: Path | None = None
    if config_path is not None:
        chosen = Path(config_path)
        if not chosen.exists():
            raise FileNotFoundError(f"Config file not found: {chosen}")
    elif USER_CONFIG_PATH.exists():
        chosen = USER_CONFIG_PATH
    if chosen is not None:
        raw = _deep_merge(raw, _read_yaml(chosen))

    defaults = raw.get("defaults", {}) or {}
    api_keys = {k: str(v or "") for k, v in (raw.get("api_keys", {}) or {}).items()}
    base_urls = {k: str(v or "") for k, v in (raw.get("base_urls", {}) or {}).items()}
    paths = raw.get("paths", {}) or {}

    if use_env:
        for key, env_names in KEY_ENV_VARS.items():
            v = _env(*env_names)
            if v:
                api_keys[key] = v
        for key, env_names in URL_ENV_VARS.items():
            v = _env(*env_names)
            if v:
                base_urls[key] = v
        defaults["text_model"] = _env("PAPERBANANA_TEXT_MODEL") or defaults.get("text_model")
        defaults["image_model"] = _env("PAPERBANANA_IMAGE_MODEL") or defaults.get("image_model")
        if _env("PAPERBANANA_CRITIC_ROUNDS"):
            defaults["critic_rounds"] = int(_env("PAPERBANANA_CRITIC_ROUNDS"))
        if _env("PAPERBANANA_RETRIEVAL_TOP_K"):
            defaults["retrieval_top_k"] = int(_env("PAPERBANANA_RETRIEVAL_TOP_K"))
        paths["data_dir"] = _env("PAPERBANANA_DATA_DIR") or paths.get("data_dir")
        paths["output_dir"] = _env("PAPERBANANA_OUTPUT_DIR") or paths.get("output_dir")

    def _path(value: Any, fallback: Path) -> Path:
        if not value:
            return fallback
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else (PACKAGE_ROOT / p)

    return Settings(
        text_model=str(defaults.get("text_model") or Settings.text_model),
        image_model=str(defaults.get("image_model") or Settings.image_model),
        critic_rounds=int(defaults.get("critic_rounds", Settings.critic_rounds)),
        retrieval_top_k=int(defaults.get("retrieval_top_k", Settings.retrieval_top_k)),
        default_aspect_ratio=str(defaults.get("default_aspect_ratio") or Settings.default_aspect_ratio),
        default_image_size=str(defaults.get("default_image_size") or Settings.default_image_size),
        default_style=str(defaults.get("default_style") or Settings.default_style),
        request_timeout=int(defaults.get("request_timeout", Settings.request_timeout)),
        api_keys=api_keys,
        base_urls=base_urls,
        data_dir=_path(paths.get("data_dir"), PACKAGE_ROOT / "data"),
        output_dir=_path(paths.get("output_dir"), PACKAGE_ROOT / "outputs"),
        config_path=chosen,
    )
