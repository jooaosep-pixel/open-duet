import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperbanana.config import Settings  # noqa: E402
from paperbanana.providers.base import ImageResult  # noqa: E402


def make_png(color=(200, 30, 30), size=(64, 32)) -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeTextModel:
    """Scripted text model: answers based on the system prompt it is given."""

    name = "fake-text"

    def __init__(self, critic_pass_on_round: int = 2):
        self.calls: list[str] = []
        self.critic_calls = 0
        self.critic_pass_on_round = critic_pass_on_round

    def complete(self, messages, *, system=None, json_mode=False, temperature=None):
        self.calls.append(system or "")
        sys_ = system or ""
        if "Retriever agent" in sys_:
            return json.dumps({"selected": [2, 1], "reason": "structure"})
        if "Planner agent" in sys_:
            return "## Layout\nleft to right\n## Components\n1. Planner box\n## Connections\n1. Planner -> Retriever"
        if "Stylist agent" in sys_:
            return "Draw a clean diagram with boxes labelled \"Planner\" and \"Retriever\". Constraints: white background."
        if "Critic agent" in sys_:
            self.critic_calls += 1
            passed = self.critic_calls >= self.critic_pass_on_round
            return json.dumps({
                "scores": {"faithfulness": 9 if passed else 6, "conciseness": 8, "readability": 9, "aesthetics": 8},
                "passed": passed,
                "issues": [] if passed else ["label 'Retriver' misspelled"],
                "revision_instructions": "" if passed else "Fix the spelling of Retriever.",
            })
        if "data plots" in sys_ or "revise matplotlib" in sys_:
            return (
                "```python\nimport os\nimport matplotlib.pyplot as plt\nfig, ax = plt.subplots(figsize=(6, 4))\n"
                "ax.bar(['a', 'b'], [1, 2])\nfig.savefig(os.environ['PAPERBANANA_OUTPUT'], dpi=50, bbox_inches='tight')\n```"
            )
        return "ok"


class FakeImageModel:
    name = "fake-image"

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, prompt, *, aspect_ratio="16:9", image_size="1K", images=()):
        self.calls.append({"prompt": prompt, "aspect_ratio": aspect_ratio, "image_size": image_size, "n_images": len(images)})
        shade = 40 * len(self.calls)
        return ImageResult(make_png((shade, shade, 200)), "image/png")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(api_keys={}, base_urls={}, data_dir=tmp_path / "data", output_dir=tmp_path / "outputs")


@pytest.fixture
def fake_text():
    return FakeTextModel()


@pytest.fixture
def fake_image():
    return FakeImageModel()


@pytest.fixture
def ref_pool(tmp_path):
    """A tiny cached 'ref' split so the Retriever has something to search."""
    from paperbanana.data.bench import ReferencePool, task_dir

    d = task_dir(tmp_path / "data", "diagram")
    (d / "images").mkdir(parents=True)
    rows = []
    specs = [
        ("ref_1", "Overview of a retrieval-augmented planner with a verifier loop", "planner retriever verifier rounds question corpus"),
        ("ref_2", "Diffusion training and sampling for molecular graphs", "masking diffusion graph transformer atoms bonds"),
        ("ref_3", "Robot manipulation policy with multi-view cameras", "camera images transformer action chunk policy"),
    ]
    for rid, cap, content in specs:
        (d / "images" / f"{rid}.jpg").write_bytes(make_png((10, 10, 10)))
        rows.append({"id": rid, "split": "ref", "category": "agent_reasoning", "content": content, "visual_intent": cap,
                     "image_file": f"images/{rid}.jpg", "rounded_ratio": "16:9", "task": "diagram"})
    (d / "ref.json").write_text(json.dumps(rows))
    return ReferencePool.load(tmp_path / "data", "diagram")
