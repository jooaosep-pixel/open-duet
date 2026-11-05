"""Small image helpers (PIL based)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

MIME_BY_FORMAT = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def open_image(data: bytes | Path | str | Image.Image) -> Image.Image:
    if isinstance(data, Image.Image):
        return data
    if isinstance(data, (str, Path)):
        return Image.open(data)
    return Image.open(io.BytesIO(data))


def to_bytes(img: Image.Image, fmt: str = "PNG", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=quality) if fmt.upper() == "JPEG" else img.save(buf, format=fmt)
    return buf.getvalue()


def downscale(data: bytes | Image.Image, max_side: int = 1024, fmt: str = "JPEG") -> tuple[bytes, str]:
    """Return (bytes, mime) of the image resized so its longest side is <= max_side."""
    img = open_image(data)
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return to_bytes(img, fmt), MIME_BY_FORMAT[fmt.upper()]


def sniff_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def data_uri(data: bytes, mime: str | None = None) -> str:
    mime = mime or sniff_mime(data)
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def decode_data_uri(uri: str) -> bytes:
    _, _, payload = uri.partition(",")
    return base64.b64decode(payload)


def ensure_png(data: bytes) -> bytes:
    if sniff_mime(data) == "image/png":
        return data
    return to_bytes(open_image(data).convert("RGBA"), "PNG")


def aspect_ratio_of(data: bytes | Image.Image) -> float:
    w, h = open_image(data).size
    return w / float(h)
