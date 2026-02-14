from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

# Pillow is optional in some environments.
# We keep this module importable even if PIL is missing so the caller can
# degrade gracefully (e.g., export CSV instead).
try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    _HAVE_PIL = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore
    _HAVE_PIL = False


@dataclass
class TableImageStyle:
    """Styling knobs for PNG table rendering."""

    font_size: int = 22
    title_font_size: int = 28
    pad_x: int = 14
    pad_y: int = 10
    margin: int = 18
    line_width: int = 2
    # Soft caps to avoid absurdly wide images if a cell string explodes
    max_col_px: int = 520
    max_total_px: int = 1600


def _first_existing(paths: Sequence[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _load_font(size: int, bold: bool = False):
    """Load a Japanese-capable font.

    Prefer Noto Sans CJK on Linux if present.
    """

    if not _HAVE_PIL:
        raise RuntimeError("Pillow (PIL) is not installed")

    reg = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    bld = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Bold.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    path = _first_existing(bld if bold else reg) or _first_existing(reg)
    if not path:
        # Fall back to Pillow's default bitmap font (ASCII-centric). This keeps
        # the pipeline alive even if Japanese glyphs may not render ideally.
        try:
            return ImageFont.load_default()  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError("No usable font found for table image rendering") from e

    # For .ttc, Pillow can take an index; default 0 is fine for NotoSansCJK.
    try:
        return ImageFont.truetype(path, size=size, index=0)  # type: ignore[arg-type]
    except TypeError:
        # Some Pillow builds don't accept index for non-ttc fonts.
        return ImageFont.truetype(path, size=size)  # type: ignore[arg-type]


def _text_bbox(draw, text: str, font) -> Tuple[int, int]:
    """Return (w, h) for a single-line text."""

    s = "" if text is None else str(text)
    x0, y0, x1, y1 = draw.textbbox((0, 0), s, font=font)
    return int(x1 - x0), int(y1 - y0)


def _truncate_to_px(draw, text: str, font, max_px: int, ellipsis: str = "…") -> str:
    s = "" if text is None else str(text)
    if max_px <= 0:
        return s
    w, _ = _text_bbox(draw, s, font)
    if w <= max_px:
        return s

    # Binary search for best prefix.
    lo, hi = 0, len(s)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = s[:mid] + ellipsis
        w2, _ = _text_bbox(draw, cand, font)
        if w2 <= max_px:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best or ellipsis


def render_table_png(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    out_path: str,
    style: TableImageStyle | None = None,
) -> str:
    """Render a simple table to a PNG image."""

    if not _HAVE_PIL:
        raise RuntimeError("Pillow (PIL) is not installed; cannot render PNG")

    style = style or TableImageStyle()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    title_font = _load_font(style.title_font_size, bold=True)
    font = _load_font(style.font_size, bold=False)
    header_font = _load_font(style.font_size, bold=True)

    # First pass: measure content.
    tmp = Image.new("RGB", (10, 10), "white")  # type: ignore[attr-defined]
    draw = ImageDraw.Draw(tmp)  # type: ignore[attr-defined]

    headers_s = ["" if h is None else str(h) for h in headers]
    rows_s = [["" if c is None else str(c) for c in r] for r in rows]

    n_cols = len(headers_s)
    if n_cols == 0:
        raise ValueError("headers must not be empty")

    # Normalize row lengths
    norm_rows: List[List[str]] = []
    for r in rows_s:
        rr = list(r[:n_cols]) + [""] * max(0, n_cols - len(r))
        norm_rows.append(rr)

    # Compute max width per column, with truncation caps.
    col_px: List[int] = []
    for j in range(n_cols):
        w_h, _ = _text_bbox(draw, headers_s[j], header_font)
        mx = w_h
        for r in norm_rows:
            w, _ = _text_bbox(draw, r[j], font)
            mx = max(mx, w)
        mx = min(mx, style.max_col_px)
        col_px.append(mx + style.pad_x * 2)

    # If total is too wide, shrink by truncation.
    total_w = sum(col_px) + style.margin * 2
    if total_w > style.max_total_px and n_cols > 0:
        over = total_w - style.max_total_px
        reducible = [max(0, w - 160) for w in col_px]
        while over > 0 and sum(reducible) > 0:
            for j in range(n_cols):
                if over <= 0:
                    break
                if reducible[j] <= 0:
                    continue
                step = min(12, reducible[j], over)
                col_px[j] -= step
                reducible[j] -= step
                over -= step
        total_w = sum(col_px) + style.margin * 2

    # Truncate texts to fit in each column (after width adjustments).
    headers_fit: List[str] = []
    for j in range(n_cols):
        max_txt_px = max(10, col_px[j] - style.pad_x * 2)
        headers_fit.append(_truncate_to_px(draw, headers_s[j], header_font, max_txt_px))

    rows_fit: List[List[str]] = []
    for r in norm_rows:
        rr: List[str] = []
        for j in range(n_cols):
            max_txt_px = max(10, col_px[j] - style.pad_x * 2)
            rr.append(_truncate_to_px(draw, r[j], font, max_txt_px))
        rows_fit.append(rr)

    # Heights
    _title_w, title_h = _text_bbox(draw, title, title_font)
    row_h = int(max(_text_bbox(draw, "A", font)[1], _text_bbox(draw, "あ", font)[1]) + style.pad_y * 2)
    head_h = int(
        max(_text_bbox(draw, "A", header_font)[1], _text_bbox(draw, "あ", header_font)[1]) + style.pad_y * 2
    )

    table_h = head_h + row_h * max(1, len(rows_fit))
    img_h = style.margin * 2 + title_h + 12 + table_h
    img_w = int(total_w)

    img = Image.new("RGB", (img_w, img_h), "white")  # type: ignore[attr-defined]
    d = ImageDraw.Draw(img)  # type: ignore[attr-defined]

    # Title
    tx = style.margin
    ty = style.margin
    d.text((tx, ty), title, font=title_font, fill="black")

    # Table origin
    y0 = ty + title_h + 12
    x0 = style.margin

    # Header background
    d.rectangle([x0, y0, img_w - style.margin, y0 + head_h], fill="#F0F0F0")

    # Grid lines
    x = x0
    d.line([x, y0, x, y0 + head_h + row_h * max(1, len(rows_fit))], fill="#202020", width=style.line_width)
    for w in col_px:
        x += w
        d.line([x, y0, x, y0 + head_h + row_h * max(1, len(rows_fit))], fill="#202020", width=style.line_width)

    y = y0
    d.line([x0, y, img_w - style.margin, y], fill="#202020", width=style.line_width)
    y += head_h
    d.line([x0, y, img_w - style.margin, y], fill="#202020", width=style.line_width)
    for _ in range(max(1, len(rows_fit))):
        y += row_h
        d.line([x0, y, img_w - style.margin, y], fill="#202020", width=style.line_width)

    # Header text
    x = x0
    for j, h in enumerate(headers_fit):
        d.text((x + style.pad_x, y0 + style.pad_y), h, font=header_font, fill="black")
        x += col_px[j]

    # Body
    for i in range(max(1, len(rows_fit))):
        rr = rows_fit[i] if i < len(rows_fit) else [""] * n_cols
        x = x0
        y = y0 + head_h + i * row_h
        for j, cell in enumerate(rr):
            d.text((x + style.pad_x, y + style.pad_y), cell, font=font, fill="black")
            x += col_px[j]

    img.save(out_path)
    return out_path


def render_table_csv(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    out_path: str,
) -> str:
    """Export the table as UTF-8 CSV (Excel-friendly with BOM)."""

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if title:
            w.writerow([title])
            w.writerow([])
        w.writerow(list(headers))
        for r in rows:
            w.writerow(list(r))
    return out_path
