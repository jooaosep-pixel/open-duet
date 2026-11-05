import json

from paperbanana.pipeline import GenerationRequest, Pipeline


def _run(pipeline, req):
    events = list(pipeline.run(req))
    assert events[-1].stage == "done"
    return events, events[-1].result


def test_full_diagram_pipeline_with_critic_loop(settings, fake_text, fake_image, ref_pool):
    p = Pipeline(settings, text_model=fake_text, image_model=fake_image, pool=ref_pool)
    req = GenerationRequest(method_text="planner retriever verifier rounds", intent="Figure 1: planner overview", critic_rounds=3, retrieval_top_k=2)
    events, result = _run(p, req)

    stages = [e.stage for e in events]
    assert stages[:3] == ["retrieve", "retrieve", "plan"]
    assert "style" in stages and "visualize" in stages and "critic" in stages
    # critic failed round 1, visualizer revised, critic passed round 2
    assert fake_text.critic_calls == 2
    assert len(fake_image.calls) == 2
    assert fake_image.calls[1]["n_images"] == 1  # revision passes the previous draft
    assert result.final_verdict.passed
    assert len(result.rounds) == 2 and result.rounds[0].verdict["passed"] is False
    assert len(result.references) == 2
    assert result.image[:8] == b"\x89PNG\r\n\x1a\n"

    # run folder persisted
    assert result.run_dir and (result.run_dir / "figure.png").exists()
    meta = json.loads((result.run_dir / "run.json").read_text())
    assert meta["final_verdict"]["passed"] is True
    assert (result.run_dir / "plan.md").exists() and (result.run_dir / "prompt.txt").exists()


def test_pipeline_without_reference_pool_is_zero_shot(settings, fake_text, fake_image):
    p = Pipeline(settings, text_model=fake_text, image_model=fake_image, load_pool=False)
    req = GenerationRequest(method_text="m", intent="i", critic_rounds=0, use_stylist=False, save=False)
    events, result = _run(p, req)
    retrieve_events = [e for e in events if e.stage == "retrieve"]
    assert retrieve_events[-1].status == "warning"
    assert result.references == []
    assert result.run_dir is None
    assert "Draw a publication-quality methodology diagram" in result.prompt
    assert not any("Stylist agent" in c for c in fake_text.calls)


def test_plot_pipeline_executes_code(settings, fake_text, fake_image):
    fake_text.critic_pass_on_round = 1
    p = Pipeline(settings, text_model=fake_text, image_model=fake_image, load_pool=False)
    req = GenerationRequest(method_text='{"a": 1, "b": 2}', intent="bar chart", figure_type="plot", critic_rounds=1, retrieval_top_k=0)
    events, result = _run(p, req)
    assert result.code and "savefig" in result.code
    assert result.image_model == "matplotlib"
    assert result.final_verdict.passed
    assert fake_image.calls == []  # no image model used for plots


def test_request_validation():
    import pytest

    with pytest.raises(ValueError):
        GenerationRequest(method_text="", intent="x").validate()
    with pytest.raises(ValueError):
        GenerationRequest(method_text="x", intent="y", figure_type="poster").validate()
