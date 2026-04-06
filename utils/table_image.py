from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates += [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT_TITLE = _font(36, bold=True)
FONT_SUB = _font(20, bold=False)
FONT_HEAD = _font(20, bold=True)
FONT_BODY = _font(18, bold=False)
FONT_NOTE = _font(16, bold=False)


PALETTE = {
    "bg": (250, 251, 253),
    "card": (255, 255, 255),
    "grid": (228, 232, 238),
    "head_bg": (36, 48, 66),
    "head_fg": (255, 255, 255),
    "text": (24, 32, 44),
    "sub": (88, 99, 112),
    "accent": (24, 94, 168),
    "row_alt": (247, 249, 252),
}


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    draw.text(xy, text, font=font, fill=fill)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def _truncate(draw: ImageDraw.ImageDraw, text: str, max_width: int, font: ImageFont.ImageFont) -> str:
    text = str(text)
    if _text_width(draw, text, font) <= max_width:
        return text
    ell = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        trial = text[:mid] + ell
        if _text_width(draw, trial, font) <= max_width:
            lo = mid + 1
        else:
            hi = mid
    return text[: max(0, lo - 1)] + ell


RIGHT_HINTS = {"score", "rr", "r", "risk", "risk%", "entry", "sl", "tp", "tp1", "tp2", "avg", "last", "pnl", "pnl%"}
CENTER_HINTS = {"ticker", "setup", "lane", "flag", "status", "sector"}


def _is_right(col: str) -> bool:
    c = str(col).lower()
    return any(h in c for h in RIGHT_HINTS)


def _is_center(col: str) -> bool:
    c = str(col).lower()
    return c in CENTER_HINTS or c.endswith("_type")


def _stringify(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "-"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 100:
            return f"{value:,.1f}"
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _column_widths(df: pd.DataFrame, draw: ImageDraw.ImageDraw, content_width: int) -> dict[str, int]:
    cols = list(df.columns)
    raw = {}
    for col in cols:
        base = max(len(str(col)) * 1.1, 6)
        vals = df[col].head(12).tolist()
        cell = max((len(_stringify(v)) for v in vals), default=6)
        if _is_center(col):
            score = min(max(base, cell * 0.75), 16)
        elif _is_right(col):
            score = min(max(base, cell * 0.85), 18)
        else:
            score = min(max(base, cell), 34)
        raw[col] = score
    total = sum(raw.values())
    if total <= 0:
        return {col: content_width // max(1, len(cols)) for col in cols}
    widths = {col: max(70, int(content_width * raw[col] / total)) for col in cols}
    delta = content_width - sum(widths.values())
    if cols:
        widths[cols[-1]] += delta
    return widths


def save_table_image(
    df: pd.DataFrame,
    path: str | Path,
    title: str,
    subtitle: str = "",
    notes: Iterable[str] | None = None,
    empty_text: str = "候補なし",
    max_rows: int = 12,
) -> str:
    notes = list(notes or [])
    frame = df.copy() if df is not None else pd.DataFrame()
    if frame is None or frame.empty:
        frame = pd.DataFrame({"Status": [empty_text]})
    frame = frame.head(max_rows).copy()
    frame.columns = [str(c) for c in frame.columns]
    for col in frame.columns:
        frame[col] = frame[col].map(_stringify)

    width = 1260
    margin = 32
    content_width = width - 2 * margin
    header_h = 92
    subtitle_h = 34 if subtitle else 0
    note_h = 28 * len(notes) + (12 if notes else 0)
    row_h = 48
    title_h = header_h + subtitle_h + note_h
    table_h = row_h * (len(frame) + 1)
    height = title_h + table_h + 32

    img = Image.new("RGB", (width, height), PALETTE["bg"])
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((14, 14, width - 14, height - 14), radius=22, fill=PALETTE["card"], outline=PALETTE["grid"], width=1)
    _text(draw, (margin, 28), title, FONT_TITLE, PALETTE["text"])
    cursor_y = 74
    if subtitle:
        _text(draw, (margin, cursor_y), subtitle, FONT_SUB, PALETTE["sub"])
        cursor_y += 28
    for note in notes:
        _text(draw, (margin, cursor_y), f"• {note}", FONT_NOTE, PALETTE["accent"])
        cursor_y += 24
    cursor_y += 12

    col_widths = _column_widths(frame, draw, content_width)
    x_positions: dict[str, tuple[int, int]] = {}
    x = margin
    for col in frame.columns:
        w = col_widths[col]
        x_positions[col] = (x, x + w)
        x += w

    draw.rounded_rectangle((margin, cursor_y, width - margin, cursor_y + row_h), radius=14, fill=PALETTE["head_bg"])
    for col in frame.columns:
        x0, x1 = x_positions[col]
        label = _truncate(draw, col, x1 - x0 - 16, FONT_HEAD)
        if _is_right(col):
            tw = _text_width(draw, label, FONT_HEAD)
            _text(draw, (x1 - tw - 10, cursor_y + 12), label, FONT_HEAD, PALETTE["head_fg"])
        elif _is_center(col):
            tw = _text_width(draw, label, FONT_HEAD)
            _text(draw, ((x0 + x1 - tw) // 2, cursor_y + 12), label, FONT_HEAD, PALETTE["head_fg"])
        else:
            _text(draw, (x0 + 8, cursor_y + 12), label, FONT_HEAD, PALETTE["head_fg"])
    cursor_y += row_h

    for ridx, (_, row) in enumerate(frame.iterrows()):
        bg = PALETTE["row_alt"] if ridx % 2 else PALETTE["card"]
        draw.rectangle((margin, cursor_y, width - margin, cursor_y + row_h), fill=bg)
        draw.line((margin, cursor_y + row_h, width - margin, cursor_y + row_h), fill=PALETTE["grid"], width=1)
        for col in frame.columns:
            x0, x1 = x_positions[col]
            text = _truncate(draw, row[col], x1 - x0 - 16, FONT_BODY)
            if _is_right(col):
                tw = _text_width(draw, text, FONT_BODY)
                _text(draw, (x1 - tw - 10, cursor_y + 13), text, FONT_BODY, PALETTE["text"])
            elif _is_center(col):
                tw = _text_width(draw, text, FONT_BODY)
                _text(draw, ((x0 + x1 - tw) // 2, cursor_y + 13), text, FONT_BODY, PALETTE["text"])
            else:
                _text(draw, (x0 + 8, cursor_y + 13), text, FONT_BODY, PALETTE["text"])
        cursor_y += row_h

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return str(out)
