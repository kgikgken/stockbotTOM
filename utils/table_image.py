from __future__ import annotations

import csv
import os
import shutil
import subprocess
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
    """Styling knobs for table rendering (PNG/SVG)."""

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


def _find_japanese_font_path() -> Optional[str]:
    """Best-effort find a Japanese-capable font path.

    - Prefers Noto Sans CJK if present.
    - Falls back to DejaVu Sans.

    This is used by both PIL and matplotlib renderers.
    """

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
    return _first_existing(bld) or _first_existing(reg)


def _load_font(size: int, bold: bool = False):
    """Load a Japanese-capable font with Pillow."""

    if not _HAVE_PIL:
        raise RuntimeError("Pillow (PIL) is not installed")

    # Prefer bold font path when requested.
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


def _mpl_modules():
    """Lazy-import matplotlib (optional dependency).

    Returns (plt, font_manager) or None if matplotlib is unavailable.
    """

    try:
        import matplotlib

        # Use a non-GUI backend.
        try:
            matplotlib.use("Agg", force=True)  # type: ignore[attr-defined]
        except Exception:
            pass

        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib import font_manager  # type: ignore

        return plt, font_manager
    except Exception:  # pragma: no cover
        return None


def _which(cmd: str) -> Optional[str]:
    """shutil.which wrapper (safe)."""

    try:
        return shutil.which(cmd)
    except Exception:  # pragma: no cover
        return None


def _run_quiet(cmd: Sequence[str]) -> bool:
    """Run a command quietly, returning True if it succeeded."""

    try:
        subprocess.run(
            list(cmd),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _svg_to_png(svg_path: str, png_path: str) -> bool:
    """Convert SVG to PNG via optional Python deps or external tools.

    This allows the bot to output a PNG even when Pillow/matplotlib are missing.
    Best-effort: returns False if conversion is not possible.
    """

    # 1) CairoSVG (if installed)
    try:
        import cairosvg  # type: ignore

        cairosvg.svg2png(url=svg_path, write_to=png_path)
        return os.path.exists(png_path) and os.path.getsize(png_path) > 0
    except Exception:
        pass

    # 2) Inkscape (preferred)
    if _which("inkscape"):
        # Inkscape 1.0+ syntax
        if _run_quiet(["inkscape", svg_path, "--export-type=png", f"--export-filename={png_path}"]):
            return os.path.exists(png_path) and os.path.getsize(png_path) > 0
        # Older syntax
        if _run_quiet(["inkscape", svg_path, "--export-png", png_path]):
            return os.path.exists(png_path) and os.path.getsize(png_path) > 0

    # 3) librsvg
    if _which("rsvg-convert"):
        if _run_quiet(["rsvg-convert", "-o", png_path, svg_path]):
            return os.path.exists(png_path) and os.path.getsize(png_path) > 0

    # 4) resvg
    if _which("resvg"):
        if _run_quiet(["resvg", svg_path, png_path]):
            return os.path.exists(png_path) and os.path.getsize(png_path) > 0

    # 5) ImageMagick
    if _which("magick"):
        if _run_quiet(["magick", svg_path, png_path]):
            return os.path.exists(png_path) and os.path.getsize(png_path) > 0
    if _which("convert"):
        if _run_quiet(["convert", svg_path, png_path]):
            return os.path.exists(png_path) and os.path.getsize(png_path) > 0

    return False


def _render_table_png_pil(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    out_path: str,
    style: TableImageStyle,
) -> str:
    """Render a simple table to a PNG image using Pillow."""

    if not _HAVE_PIL:
        raise RuntimeError("Pillow (PIL) is not installed; cannot render PNG")

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


def _render_table_png_mpl(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    out_path: str,
    style: TableImageStyle,
) -> str:
    """Render table as PNG using matplotlib (no Pillow required).

    This is a fallback for environments where Pillow isn't installed.
    """

    mods = _mpl_modules()
    if mods is None:
        raise RuntimeError("matplotlib is not installed; cannot render PNG")

    plt, font_manager = mods

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Normalize table cells to strings.
    headers_s = ["" if h is None else str(h) for h in headers]
    n_cols = len(headers_s)
    if n_cols == 0:
        raise ValueError("headers must not be empty")

    norm_rows: List[List[str]] = []
    for r in rows:
        rr = [("" if c is None else str(c)) for c in r]
        rr = list(rr[:n_cols]) + [""] * max(0, n_cols - len(rr))
        norm_rows.append(rr)

    # Rough truncation to keep images reasonably sized.
    # Pixel-accurate truncation requires PIL, so we go by character count.
    # (Font size ~22 -> ~12px per ASCII char; Japanese tends to be wider.)
    est_char_px = max(8.0, style.font_size * 0.55)
    max_chars = max(8, int(style.max_col_px / est_char_px))

    def _trunc(s: str, n: int) -> str:
        if n <= 0:
            return s
        if len(s) <= n:
            return s
        if n <= 1:
            return "…"
        return s[: n - 1] + "…"

    # Per-column caps
    col_caps: List[int] = []
    for j in range(n_cols):
        mx = len(headers_s[j])
        for r in norm_rows:
            mx = max(mx, len(r[j]))
        col_caps.append(min(mx, max_chars))

    headers_fit = [_trunc(headers_s[j], col_caps[j]) for j in range(n_cols)]
    rows_fit = [[_trunc(r[j], col_caps[j]) for j in range(n_cols)] for r in norm_rows]

    # Figure sizing (approx; DPI-based cap).
    dpi = 150
    px_est_w = (
        sum((c + 2) for c in col_caps) * est_char_px
        + n_cols * style.pad_x * 2
        + style.margin * 2
    )
    fig_w = max(6.0, min(style.max_total_px / dpi, px_est_w / dpi))

    row_px = style.font_size * 1.8 + style.pad_y * 2
    px_est_h = (len(rows_fit) + 2) * row_px + style.title_font_size * 2.0 + style.margin * 2
    fig_h = max(2.8, min(40.0, px_est_h / dpi))

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis("off")

    # Font (best-effort)
    fp = None
    font_path = _find_japanese_font_path()
    if font_path:
        try:
            fp = font_manager.FontProperties(fname=font_path)
        except Exception:
            fp = None

    # Title
    if fp is not None:
        ax.set_title(title, fontproperties=fp, fontsize=style.title_font_size, loc="left", pad=12)
    else:
        ax.set_title(title, fontsize=style.title_font_size, loc="left", pad=12)

    tbl = ax.table(cellText=rows_fit, colLabels=headers_fit, cellLoc="left", loc="upper left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(style.font_size)

    # Mild vertical scaling for readability
    tbl.scale(1.0, 1.35)

    # Apply per-cell styling
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#F0F0F0")
            cell.get_text().set_weight("bold")
        if fp is not None:
            cell.get_text().set_fontproperties(fp)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out_path


def render_table_png(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    out_path: str,
    style: TableImageStyle | None = None,
) -> str:
    """Render a simple table to a PNG image.

    Priority:
      1) Pillow (best typography / pixel truncation)
      2) matplotlib fallback (works even when Pillow isn't installed)

    If neither is available, try to render SVG and convert it to PNG via
    external tools (inkscape / rsvg-convert / resvg / ImageMagick) or
    optional CairoSVG.

    If neither is available, raises RuntimeError.
    """

    style = style or TableImageStyle()

    if _HAVE_PIL:
        return _render_table_png_pil(title, headers, rows, out_path, style)

    # Fallback: matplotlib (if installed)
    try:
        return _render_table_png_mpl(title, headers, rows, out_path, style)
    except Exception:
        pass

    # Fallback: SVG -> PNG (via external tools)
    svg_path = os.path.splitext(out_path)[0] + ".svg"
    try:
        render_table_svg(title, headers, rows, svg_path, style)
        if _svg_to_png(svg_path, out_path):
            return out_path
    except Exception:
        pass

    raise RuntimeError(
        "matplotlib/Pillowが無くPNGを作れません。pip install pillow または pip install matplotlib、"
        "もしくは inkscape / rsvg-convert / resvg / ImageMagick(convert) のいずれかを用意してください。"
        "（SVGは out/report_table_*.svg として出力可能です）"
    )


def render_table_svg(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    out_path: str,
    style: TableImageStyle | None = None,
) -> str:
    """Render a table as SVG (no external dependencies).

    This is a fallback when PNG rendering isn't possible.
    """

    from xml.sax.saxutils import escape

    style = style or TableImageStyle()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    headers_s = ["" if h is None else str(h) for h in headers]
    n_cols = len(headers_s)
    if n_cols == 0:
        raise ValueError("headers must not be empty")

    norm_rows: List[List[str]] = []
    for r in rows:
        rr = [("" if c is None else str(c)) for c in r]
        rr = list(rr[:n_cols]) + [""] * max(0, n_cols - len(rr))
        norm_rows.append(rr)

    # Approximate text metrics
    char_px = max(8.0, style.font_size * 0.60)

    # Column widths
    col_px: List[int] = []
    for j in range(n_cols):
        mx = len(headers_s[j])
        for r in norm_rows:
            mx = max(mx, len(r[j]))
        w = int(min(style.max_col_px, mx * char_px + style.pad_x * 2))
        col_px.append(w)

    total_w = int(min(style.max_total_px, sum(col_px) + style.margin * 2))

    # Heights
    title_h = int(style.title_font_size * 1.6)
    head_h = int(style.font_size * 1.8 + style.pad_y * 2)
    row_h = int(style.font_size * 1.8 + style.pad_y * 2)

    n_rows = max(1, len(norm_rows))
    table_h = head_h + row_h * n_rows
    total_h = int(style.margin * 2 + title_h + 12 + table_h)

    # If total_w is capped, proportionally shrink columns.
    raw_w = sum(col_px) + style.margin * 2
    if raw_w > total_w and sum(col_px) > 0:
        scale = (total_w - style.margin * 2) / float(sum(col_px))
        col_px = [max(60, int(w * scale)) for w in col_px]

    # SVG build
    # Font stack: prefer Noto (if installed) but allow system fallback.
    font_stack = "'Noto Sans CJK JP','Noto Sans CJK','Noto Sans','DejaVu Sans',sans-serif"

    def _rect(x, y, w, h, fill="white", stroke="#202020", sw=1):
        return f"<rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"{sw}\" />"

    def _text(x, y, s, size, weight="normal"):
        s2 = escape(s)
        return (
            f"<text x=\"{x}\" y=\"{y}\" font-family=\"{font_stack}\" font-size=\"{size}\" font-weight=\"{weight}\" "
            f"dominant-baseline=\"hanging\" fill=\"#000\">{s2}</text>"
        )

    parts: List[str] = []
    parts.append(f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{total_w}\" height=\"{total_h}\" viewBox=\"0 0 {total_w} {total_h}\">")
    parts.append(_rect(0, 0, total_w, total_h, fill="white", stroke="none", sw=0))

    # Title
    tx = style.margin
    ty = style.margin
    parts.append(_text(tx, ty, title, style.title_font_size, weight="bold"))

    # Table origin
    x0 = style.margin
    y0 = ty + title_h + 12

    # Header background
    parts.append(_rect(x0, y0, total_w - style.margin * 2, head_h, fill="#F0F0F0", stroke="#202020", sw=1))

    # Outer table border
    parts.append(_rect(x0, y0, total_w - style.margin * 2, table_h, fill="none", stroke="#202020", sw=1))

    # Vertical lines + header text
    x = x0
    for j in range(n_cols):
        w = col_px[j]
        # Header text
        parts.append(_text(x + style.pad_x, y0 + style.pad_y, headers_s[j], style.font_size, weight="bold"))
        x += w
        # Vertical grid
        parts.append(f"<line x1=\"{x}\" y1=\"{y0}\" x2=\"{x}\" y2=\"{y0 + table_h}\" stroke=\"#202020\" stroke-width=\"1\" />")

    # Horizontal lines
    parts.append(f"<line x1=\"{x0}\" y1=\"{y0 + head_h}\" x2=\"{x0 + sum(col_px)}\" y2=\"{y0 + head_h}\" stroke=\"#202020\" stroke-width=\"1\" />")
    for i in range(n_rows):
        y = y0 + head_h + i * row_h
        parts.append(f"<line x1=\"{x0}\" y1=\"{y + row_h}\" x2=\"{x0 + sum(col_px)}\" y2=\"{y + row_h}\" stroke=\"#202020\" stroke-width=\"1\" />")

    # Body cells text
    for i in range(n_rows):
        rr = norm_rows[i] if i < len(norm_rows) else [""] * n_cols
        y = y0 + head_h + i * row_h
        x = x0
        for j in range(n_cols):
            parts.append(_text(x + style.pad_x, y + style.pad_y, rr[j], style.font_size, weight="normal"))
            x += col_px[j]

    parts.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

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
