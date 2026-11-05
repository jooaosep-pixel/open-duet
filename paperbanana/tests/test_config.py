import textwrap

from paperbanana.config import Settings, load_settings


def test_template_defaults_load(monkeypatch):
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = load_settings(use_env=False) if False else load_settings()
    assert s.text_model
    assert s.image_model
    assert s.critic_rounds >= 0


def test_user_config_and_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(textwrap.dedent("""
        defaults:
          text_model: openrouter/vendor/model-x
          critic_rounds: 3
        api_keys:
          openrouter_api_key: sk-or-test
        paths:
          output_dir: out_here
    """))
    monkeypatch.setenv("GOOGLE_API_KEY", "g-env")
    monkeypatch.setenv("PAPERBANANA_IMAGE_MODEL", "gpt-image-2")
    s = load_settings(cfg)
    assert s.text_model == "openrouter/vendor/model-x"
    assert s.image_model == "gpt-image-2"
    assert s.critic_rounds == 3
    assert s.key("openrouter_api_key") == "sk-or-test"
    assert s.key("google_api_key") == "g-env"
    assert s.output_dir.name == "out_here"
    assert "OpenRouter" in s.configured_providers() and "Google Gemini" in s.configured_providers()


def test_with_overrides_routes_keys_and_fields():
    s = Settings()
    s2 = s.with_overrides(google_api_key="abc", text_model="gemini-x", openai_base_url="http://x", critic_rounds="")
    assert s2.key("google_api_key") == "abc"
    assert s2.text_model == "gemini-x"
    assert s2.url("openai_base_url") == "http://x"
    assert s.key("google_api_key") == ""  # original untouched
