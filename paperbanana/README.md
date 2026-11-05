# 🍌 PaperBanana

Turn a paragraph of method text and a figure caption into a clean academic illustration or plot.

PaperBanana re-implements the reference-driven multi-agent workflow from
[*PaperBanana: Automating Academic Illustration for AI Scientists*](https://arxiv.org/abs/2601.23265)
as a small, easy-to-run Python app:

| Agent | What it does |
| --- | --- |
| **Retriever** | Finds structurally similar figures in the `ref` split of [PaperBananaBench](https://huggingface.co/datasets/dwzhu/PaperBananaBench) (TF-IDF recall, LLM rerank). |
| **Planner** | Writes an explicit diagram specification (layout, components, arrows, exact labels) from the method text, caption and reference examples. |
| **Stylist** | Derives a style guide (palette, shapes, typography) from the references and your preset, and composes the final image prompt. |
| **Visualizer** | Generates the figure with an image model (Gemini "Nano Banana" family, OpenAI `gpt-image-*`, or any OpenRouter image model). For plots it writes matplotlib code and renders it locally. |
| **Critic** | Scores faithfulness, conciseness, readability and aesthetics, then sends precise revision instructions back to the Visualizer for up to *N* rounds. |

It ships with a polished Gradio web UI, a CLI, a one-command benchmark downloader and a single YAML
config for model names and API keys.

![pipeline](https://img.shields.io/badge/pipeline-retrieve%20%E2%86%92%20plan%20%E2%86%92%20style%20%E2%86%92%20generate%20%E2%86%92%20critique-f5b400)

## Quick start

```bash
cd paperbanana
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .                                       # or: pip install -r requirements.txt

cp configs/model_config.template.yaml configs/model_config.yaml
# edit configs/model_config.yaml: paste one API key (Gemini is enough for everything)

paperbanana prepare-bench            # optional but recommended: downloads PaperBananaBench (~0.5 GB) for retrieval
paperbanana serve                    # open http://127.0.0.1:7860
```

Prefer not to create a config file? Export a key instead (`GOOGLE_API_KEY`, `OPENAI_API_KEY` or
`OPENROUTER_API_KEY`), or type it into the **Models & API key** panel of the web UI. Keys typed in the UI
are used for that request only and never written to disk.

`python app.py` also launches the UI.

## Web interface

1. Paste a paragraph from your method section (or a table / JSON of numbers for a plot).
2. Write the caption you intend to use. It drives what the figure focuses on.
3. Pick **Diagram** or **Plot**, an aspect ratio (21:9 to 9:16), a resolution (1K/2K/4K) and a style preset. Add free-form style notes if you like.
4. Click **Generate figure**. The stage pills and log stream progress; the result appears as soon as the first draft exists and is replaced by each revised draft.
5. Inspect the **Critic** scores, the **Plan**, the final **Image prompt / code**, the **References** the planner saw and every **Draft**. Download the PNG from the image toolbar.

The **Benchmark** tab downloads PaperBananaBench from the Hugging Face Hub, and the **Benchmark case** button fills
the form with a random held-out test case so you can compare against the ground-truth figure.

Every run is saved under `outputs/<timestamp>-<title>/` with `figure.png`, `plan.md`, `prompt.txt`, each draft
(`round_N.png`), the critic verdicts and a `run.json` manifest.

## Command line

```bash
# one figure from a text file
paperbanana generate --text-file method.md \
  --intent "Figure 1: Overview of our framework. ..." \
  --aspect 16:9 --size 2K --style clean --rounds 2 --out figure.png

# pipe text from stdin, no critic loop, zero-shot (no retrieval)
cat method.txt | paperbanana generate --intent "..." --no-critic --top-k 0

# a plot from data (matplotlib code is generated, executed locally and reviewed by the Critic)
paperbanana generate --type plot --text '{"x": [1,2,3], "acc": [71.2, 78.9, 84.3]}' \
  --intent "Line chart of accuracy vs. number of retrieval rounds, with markers"

# run PaperBananaBench test cases (needs `paperbanana prepare-bench`)
paperbanana bench --task diagram --limit 3 --out-dir runs/bench
paperbanana bench --id test_12 --rounds 3

# utilities
paperbanana prepare-bench --tasks diagram plot      # download / refresh the dataset cache
paperbanana check                                   # show the effective config, credentials and routing
paperbanana serve --host 0.0.0.0 --port 7860        # web UI (add --share for a public Gradio link)
```

Models and keys can be overridden per call: `--text-model`, `--image-model`, `--api-key`, `--key-provider`.

## Configuration

`configs/model_config.yaml` (copied from the template, git-ignored):

```yaml
defaults:
  text_model: "gemini-3.1-pro-preview"      # Retriever / Planner / Stylist / Critic
  image_model: "gemini-3.1-flash-image"     # Visualizer
  critic_rounds: 2
  retrieval_top_k: 3
api_keys:
  google_api_key: "..."
```

Routing is decided by the model-name prefix, so mixing vendors is just a matter of naming:

| Prefix | Provider | Text | Image |
| --- | --- | --- | --- |
| `gemini-*` | Google Gemini API | ✓ | ✓ (`gemini-3.1-flash-image`, `gemini-3-pro-image`, `gemini-2.5-flash-image`) |
| `gpt-*`, `o*`, `openai/*`, `gpt-image-*` | OpenAI | ✓ | ✓ (`gpt-image-2`, `gpt-image-1`) |
| `openrouter/<vendor>/<model>` | OpenRouter | ✓ | ✓ (models with image output) |
| `local/<model>` | any OpenAI-compatible server (`local_openai_base_url`) | ✓ | – |
| `ollama/<model>` | Ollama | ✓ | – |

Precedence: template defaults → `configs/model_config.yaml` → environment variables
(`GOOGLE_API_KEY`/`GEMINI_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `PAPERBANANA_TEXT_MODEL`,
`PAPERBANANA_IMAGE_MODEL`, `PAPERBANANA_DATA_DIR`, `PAPERBANANA_OUTPUT_DIR`, …) → UI / CLI overrides.
Run `paperbanana check` to see what is in effect.

Notes on image models (from the provider docs at the time of writing):

- Gemini image models take `aspect_ratio` and `image_size` (`1K`/`2K`/`4K`) directly; 4K requires `gemini-3-pro-image`. Older models that reject `image_size` are retried without it.
- OpenAI `gpt-image-*` models expose fixed sizes; aspect ratios are mapped to the closest one (`1536x1024`, `1024x1024`, `1024x1536`) and resolution to `quality`. Revisions use the `/images/edits` endpoint with the previous draft as input.
- Model ids change; if a default stops working, change one line in the config.

## The benchmark

[`dwzhu/PaperBananaBench`](https://huggingface.co/datasets/dwzhu/PaperBananaBench) has two subsets,
`diagram` (NeurIPS 2025 methodology figures) and `plot` (ChartMimic-derived charts), each with a `test` split
and a disjoint `ref` split. PaperBanana only ever retrieves from `ref`, so you can evaluate on `test` honestly.

```bash
paperbanana prepare-bench --tasks diagram            # cache under data/PaperBananaBench/diagram/{ref,test}.json + images/
```

or programmatically:

```python
from paperbanana.data import download_bench, ReferencePool, load_test_cases
download_bench(Path("data"), tasks=["diagram"])
pool = ReferencePool.load(Path("data"), "diagram")
pool.search("planner, retriever and verifier loop", top_k=3)
cases = load_test_cases(Path("data"), "diagram", "test")
```

The pipeline runs without the dataset (zero-shot planning), but retrieval noticeably improves layouts.

## Using it from Python

```python
from paperbanana.config import load_settings
from paperbanana.pipeline import GenerationRequest, generate

settings = load_settings()
result = generate(settings, GenerationRequest(method_text=open("method.md").read(), intent="Figure 1: ...", aspect_ratio="16:9"))
open("figure.png", "wb").write(result.image)
print(result.final_verdict.scores, result.run_dir)
```

`Pipeline(settings).run(request)` is a generator of `StageEvent`s if you want to stream progress yourself.

## Development

```bash
pip install -e ".[dev]"
pytest -q          # 29 tests, all offline (fake providers), ~5 s
```

Project layout:

```
paperbanana/
├── app.py                      # launch the web UI
├── configs/model_config.template.yaml
├── scripts/prepare_bench.py
├── paperbanana/
│   ├── cli.py / webapp.py      # entry points
│   ├── pipeline.py             # orchestration + streaming events
│   ├── agents/                 # retriever, planner, stylist, visualizer, critic, plotter
│   ├── providers/              # gemini, openai-compatible, router
│   ├── data/bench.py           # Hugging Face download, local cache, reference pool
│   ├── prompts.py / styles.py / examples.py
│   └── utils/
└── tests/
```

## Caveats

- **Plot mode executes model-generated matplotlib code** in a subprocess with a timeout and a scrubbed environment. It is not a security sandbox; use it with trusted models.
- Reference figures are copyrighted by their original authors; they are only used as in-context examples and are never passed to the image model unless you tick *Pass references to the image model*.
- Generated figures can still contain errors. The Critic catches many, but always check the final figure against your method.

Licensed under Apache-2.0, like the original PaperBanana code and dataset.
