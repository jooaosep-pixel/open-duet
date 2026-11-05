from paperbanana.cli import main


def test_check_runs(capsys, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rc = main(["check"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "text model" in out and "routing" in out


def test_generate_requires_text(capsys):
    rc = main(["generate", "--intent", "x", "--text", ""])
    assert rc in (1, 2)
