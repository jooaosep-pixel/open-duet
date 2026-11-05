import json

import pytest

from paperbanana.agents import Critic, Retriever, Verdict, run_plot_code
from paperbanana.agents.plotter import PlotRenderError
from paperbanana.data.bench import ReferencePool, bench_status
from paperbanana.utils.text import extract_json, strip_code_fences


def test_reference_pool_lexical_search(ref_pool):
    hits = ref_pool.search("a planner issues sub-queries to a retriever and a verifier scores answers over several rounds", top_k=2)
    assert hits[0].id == "ref_1"
    assert hits[0].score > 0
    assert len(hits) == 2


def test_bench_status_reports_cached_splits(ref_pool, tmp_path):
    assert bench_status(tmp_path / "data") == {"diagram": {"ref": 3}}
    assert ReferencePool.is_available(tmp_path / "data", "diagram")
    assert not ReferencePool.is_available(tmp_path / "data", "plot")


def test_retriever_uses_llm_rerank_then_pads(ref_pool, fake_text):
    refs = Retriever(ref_pool, fake_text).retrieve("camera images robot policy", "Robot policy overview", top_k=2)
    # fake model picks candidates [2, 1] of the lexical shortlist
    assert len(refs) == 2
    assert any("Retriever agent" in c for c in fake_text.calls)


def test_retriever_without_pool_returns_empty(fake_text):
    assert Retriever(None, fake_text).retrieve("x", "y", 3) == []


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('text before {"passed": true, "x": [1,2]} after') == {"passed": True, "x": [1, 2]}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_verdict_threshold_overrides_model_claim():
    v = Verdict.from_raw(json.dumps({"scores": {"faithfulness": 5, "conciseness": 9, "readability": 9, "aesthetics": 9}, "passed": True, "issues": []}))
    assert v.passed is False
    assert v.mean_score == 8.0


def test_critic_parses(fake_text):
    from tests.conftest import make_png

    fake_text.critic_pass_on_round = 1
    v = Critic(fake_text).review(make_png(), method_text="m", intent="i", prompt="p")
    assert v.passed and v.scores["faithfulness"] == 9


def test_run_plot_code_renders_png():
    code = strip_code_fences(
        "```python\nimport os, matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot([0, 1], [0, 1])\n"
        "fig.savefig(os.environ['PAPERBANANA_OUTPUT'], dpi=40)\n```"
    )
    png = run_plot_code(code, timeout=60)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_run_plot_code_reports_errors():
    with pytest.raises(PlotRenderError, match="ZeroDivisionError"):
        run_plot_code("x = 1/0", timeout=60)
    with pytest.raises(PlotRenderError, match="did not save"):
        run_plot_code("print('hi')", timeout=60)
