"""Plot path: the Visualizer writes matplotlib code, which is executed in a subprocess to render the figure."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ..prompts import PLOT_CODER_SYSTEM, PLOT_FIX_SYSTEM
from ..providers.base import Message, TextModel
from ..utils.text import strip_code_fences, truncate

_RATIO_TO_FIGSIZE = {"21:9": (14, 6), "16:9": (12, 6.75), "3:2": (9, 6), "4:3": (8, 6), "1:1": (6, 6), "3:4": (6, 8), "9:16": (6, 10.5)}


class PlotRenderError(RuntimeError):
    pass


def figsize_for(aspect_ratio: str) -> tuple[float, float]:
    return _RATIO_TO_FIGSIZE.get(aspect_ratio, (12, 6.75))


class PlotCoder:
    def __init__(self, text_model: TextModel):
        self.text_model = text_model

    def write_code(self, data_text: str, intent: str, *, aspect_ratio: str, style_guidelines: str, notes: str = "") -> str:
        w, h = figsize_for(aspect_ratio)
        prompt = (
            f"PLOT INTENT / CAPTION:\n{intent.strip()}\n\nDATA (table, JSON or prose):\n{truncate(data_text.strip(), 12000)}\n\n"
            f"FIGURE SIZE: figsize=({w}, {h}) inches.\nSTYLE GUIDELINES: {style_guidelines}\n"
            + (f"AUTHOR NOTES: {notes.strip()}\n" if notes.strip() else "")
            + "\nWrite the complete script."
        )
        raw = self.text_model.complete([Message.user(prompt)], system=PLOT_CODER_SYSTEM, temperature=0.2)
        return strip_code_fences(raw)

    def fix_code(self, code: str, instructions: str, error: str | None = None) -> str:
        prompt = f"CURRENT CODE:\n```python\n{code}\n```\n\nREVIEWER INSTRUCTIONS:\n{instructions.strip()}\n"
        if error:
            prompt += f"\nThe previous run FAILED with this error, fix it as well:\n{truncate(error, 3000)}\n"
        raw = self.text_model.complete([Message.user(prompt)], system=PLOT_FIX_SYSTEM, temperature=0.2)
        return strip_code_fences(raw)


def run_plot_code(code: str, *, timeout: int = 60) -> bytes:
    """Execute generated matplotlib code in a clean subprocess and return the PNG bytes.

    The code is model-generated: it runs with the current interpreter in an isolated temporary
    directory, with network-free matplotlib (Agg backend) and a hard timeout. Only run plots from
    sources you trust.
    """
    with tempfile.TemporaryDirectory(prefix="paperbanana_plot_") as tmp:
        tmp_path = Path(tmp)
        script = tmp_path / "plot.py"
        out = tmp_path / "figure.png"
        script.write_text("import matplotlib\nmatplotlib.use('Agg')\n" + code, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PAPERBANANA_OUTPUT": str(out),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
            "HOME": str(tmp_path),
            "PYTHONIOENCODING": "utf-8",
        }
        for var in ("SYSTEMROOT", "TEMP", "TMP", "VIRTUAL_ENV", "PYTHONPATH"):
            if var in os.environ:
                env[var] = os.environ[var]
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)], cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise PlotRenderError(f"plot code timed out after {timeout}s") from exc
        if proc.returncode != 0:
            raise PlotRenderError(proc.stderr.strip()[-3000:] or f"exit code {proc.returncode}")
        if not out.exists():
            raise PlotRenderError("the script finished but did not save PAPERBANANA_OUTPUT")
        return out.read_bytes()
