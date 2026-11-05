"""Run PaperBanana on figure requests extracted from the PENSI EEG deck."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paperbanana.config import load_settings
from paperbanana.pipeline import GenerationRequest, Pipeline
from paperbanana.providers import ProviderError

here = Path(__file__).parent
out = here / "out"; out.mkdir(exist_ok=True)
settings = load_settings().with_overrides(output_dir=out)
reqs = json.loads((here / (sys.argv[1] if sys.argv[1:] and sys.argv[1].endswith(".json") else "requests.json")).read_text())
only = [a for a in sys.argv[1:] if not a.endswith(".json")] or [r["id"] for r in reqs]
for r in reqs:
    if r["id"] not in only: continue
    print(f"\n=== {r['id']} ===")
    req = GenerationRequest(method_text=r["method_text"], intent=r["intent"], aspect_ratio=r["aspect"],
                            style_preset=r["style"], critic_rounds=2, retrieval_top_k=3, title=r["id"],
                            style_notes="Rótulos em português. Paleta: roxo #6A4BB8, azul #33639A, laranja #C44A2F, grafite #1B1F3A, fundo branco.")
    try:
        for ev in Pipeline(settings).run(req):
            print(f"[{ev.stage:9s}] {ev.message}")
            if ev.result: (out / f"{r['id']}.png").write_bytes(ev.result.image)
    except ProviderError as e:
        print("ERRO:", e); break
