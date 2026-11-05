"""Figure style presets and size options shared by the CLI and the web UI."""

from __future__ import annotations

ASPECT_RATIOS: list[str] = ["21:9", "16:9", "3:2", "4:3", "1:1", "3:4", "9:16"]
IMAGE_SIZES: list[str] = ["1K", "2K", "4K"]

STYLE_PRESETS: dict[str, dict[str, str]] = {
    "clean": {
        "label": "Clean academic",
        "guidelines": (
            "Flat vector look, white background, thin dark-grey outlines, rounded rectangles for modules, "
            "a restrained palette of 3-4 muted colours (soft blue, warm orange, sage green, light grey), "
            "sans-serif labels, generous whitespace, clear left-to-right or top-to-bottom flow with simple arrows."
        ),
    },
    "minimal": {
        "label": "Minimal line art",
        "guidelines": (
            "Black-and-white line drawing with a single accent colour, no fills except very light grey, "
            "hairline arrows, compact layout, typographic labels only, nothing decorative."
        ),
    },
    "colorful": {
        "label": "Colourful block diagram",
        "guidelines": (
            "Bold pastel fills for every module, each stage in its own hue, strong rounded borders, bold sans-serif titles, "
            "icons inside boxes where helpful, high contrast between text and fill, still white background."
        ),
    },
    "schematic": {
        "label": "Technical schematic",
        "guidelines": (
            "Precise engineering schematic feel: grid-aligned rectangles, orthogonal connectors with square corners, "
            "monochrome blue-grey palette, small caps labels, dashed boxes for grouped components, legend in a corner."
        ),
    },
    "sketch": {
        "label": "Whiteboard sketch",
        "guidelines": (
            "Hand-drawn marker style on a white board, slightly irregular strokes, two marker colours plus black, "
            "readable handwritten-style labels, still neatly organised and uncluttered."
        ),
    },
    "grayscale": {
        "label": "Print-safe grayscale",
        "guidelines": (
            "Only shades of grey and black on white, differentiate elements by hatching, line weight and shape, "
            "high legibility when printed, formal journal look."
        ),
    },
}

PLOT_STYLE_PRESETS: dict[str, dict[str, str]] = {
    "clean": {
        "label": "Clean academic",
        "guidelines": "matplotlib default fonts, no top/right spines, light dotted grid, colourblind-safe palette (tab10 or viridis), legend without frame.",
    },
    "minimal": {
        "label": "Minimal",
        "guidelines": "No grid, thin spines, grey tick labels, single accent colour with grey for context series, direct labels instead of a legend when possible.",
    },
    "colorful": {
        "label": "Colourful",
        "guidelines": "Saturated but harmonious palette (Set2 / Dark2), slightly thicker lines, filled markers, framed legend.",
    },
    "schematic": {"label": "Technical", "guidelines": "Full box spines, major and minor ticks, serif fonts (Times-like), monochrome blue palette with distinct markers."},
    "sketch": {"label": "Sketch (xkcd-like)", "guidelines": "Use plt.xkcd()-like loose aesthetics but keep text perfectly legible."},
    "grayscale": {"label": "Print-safe grayscale", "guidelines": "Grayscale only, distinguish series with line styles and hatching, high contrast."},
}


def style_choices(figure_type: str = "diagram") -> list[tuple[str, str]]:
    presets = STYLE_PRESETS if figure_type == "diagram" else PLOT_STYLE_PRESETS
    return [(v["label"], k) for k, v in presets.items()]


def style_guidelines(preset: str, figure_type: str = "diagram") -> str:
    presets = STYLE_PRESETS if figure_type == "diagram" else PLOT_STYLE_PRESETS
    return presets.get(preset, presets["clean"])["guidelines"]
