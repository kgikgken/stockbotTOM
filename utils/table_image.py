from __future__ import annotations
import csv
import os
import shutil
import subprocess
from pathlib import Path
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
    """Styling knobs for the PNG/SVG table renderer.
    Notes
    -----
    This bot is mainly consumed on mobile (LINE chat). If the image becomes too wide,
    LINE scales it down to fit the chat bubble and the text becomes unreadable.
    So we intentionally cap the width and prefer wrapping (taller image) over truncation.
    """
    # Target width for mobile (LINE chat). Keep this modest so text stays large in the preview.
    max_total_px: int = 960
    # Per-column cap (before global scaling). Larger values help long text columns.
    max_col_px: int = 520
    margin: int = 20
    pad_x: int = 16
    pad_y: int = 12
    font_size: int = 28
    title_font_size: int = 40
    section_font_size: int = 30
    line_width: int = 2
    line_spacing: int = 6  # between multiline lines
    # Colors
    header_bg: str = "#F2F2F2"
    zebra_bg: str = "#F7F7F7"
    section_bg: str = "#E8F0FE"
    text_color: str = "#101010"
    grid_color: str = "#202020"
    # Text wrapping
    wrap_cells: bool = True
    max_lines: int = 5



def _first_existing(paths: Sequence[str]) -> Optional[str]:
    for p in paths:
        try:
            if p and Path(p).exists():
                return str(p)
        except Exception:
            continue
    return None


def _load_pil_font(size: int, *, bold: bool = False):
    """Load a Japanese-capable font with Pillow.

    GitHub Actions (ubuntu) typically has Noto CJK installed.
    """

    if not _HAVE_PIL:
        raise RuntimeError("Pillow (PIL) is not installed")

    from PIL import ImageFont

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

    chosen = _first_existing(bld if bold else reg) or _first_existing(reg)
    if not chosen:
        return ImageFont.load_default()

    try:
        return ImageFont.truetype(chosen, size=size, index=0)
    except TypeError:
        return ImageFont.truetype(chosen, size=size)

def _text_bbox(draw, text: str, font, spacing: int = 0) -> Tuple[int, int]:
    """Return (w, h) for a possibly-multiline text."""
    s = "" if text is None else str(text)
    try:
        x0, y0, x1, y1 = draw.multiline_textbbox((0, 0), s, font=font, spacing=spacing)
    except Exception:
        # Fallback: treat as single line.
        x0, y0, x1, y1 = draw.textbbox((0, 0), s.replace("\n", " "), font=font)
    return int(x1 - x0), int(y1 - y0)
def _truncate_to_px(draw, text: str, font, max_px: int, ellipsis: str = "…") -> str:
    s = "" if text is None else str(text)
    # If multi-line, truncate each line independently.
    if "\n" in s:
        return "\n".join(_truncate_to_px(draw, line, font, max_px, ellipsis=ellipsis) for line in s.splitlines())
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
def _wrap_to_px(draw, text: str, font, max_px: int, max_lines: int = 0) -> str:
    """Wrap text so each rendered line fits within *max_px* (pixels).
    - Existing newlines are preserved and wrapped per line.
    - For Japanese (no spaces), this wraps per character.
    - If *max_lines* > 0, the text is truncated to that many lines (with ellipsis).
    """
    s = "" if text is None else str(text)
    if max_px <= 0 or not s:
        return s
    out: list[str] = []
    for raw in s.splitlines() or [""]:
        line = raw
        if line == "":
            out.append("")
            continue
        while line:
            w, _ = _text_bbox(draw, line, font)
            if w <= max_px:
                out.append(line)
                line = ""
                break
            # Binary search for the longest prefix that fits.
            lo, hi = 1, len(line)
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                part = line[:mid]
                w2, _ = _text_bbox(draw, part, font)
                if w2 <= max_px:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            out.append(line[:best])
            line = line[best:]
    if max_lines and len(out) > max_lines:
        ell = "…"
        out = out[:max_lines]
        last = out[-1]
        while last and _text_bbox(draw, last + ell, font)[0] > max_px:
            last = last[:-1]
        out[-1] = (last + ell) if last else ell
    return "\n".join(out)
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
    """Render a table image using Pillow.
    The output is tuned for mobile (LINE): keep width modest and prefer wrapping.
    """
    # Pillow is an optional dependency (installed in GitHub Actions).
    from PIL import Image, ImageDraw, ImageFont
    headers_s = ["" if h is None else str(h) for h in headers]
    n_cols = len(headers_s)
    # ---- Fonts
    # Title / section headers: bold makes the image easier to scan on mobile.
    title_font = _load_pil_font(style.title_font_size, bold=True)
    header_font = _load_pil_font(style.font_size, bold=True)
    body_font = _load_pil_font(style.font_size)
    section_font = _load_pil_font(style.section_font_size, bold=True)
    # ---- Measurement helpers (use a tiny dummy canvas)
    dummy = Image.new("RGB", (4, 4), "white")
    mdraw = ImageDraw.Draw(dummy)
    def fit_cell(text: str, font: ImageFont.ImageFont, max_w: int, *, max_lines: int) -> str:
        if style.wrap_cells:
            return _wrap_to_px(mdraw, text, font, max_w, max_lines=max_lines)
        return _truncate_to_px(mdraw, text, font, max_w)
    # ---- Normalize rows
    # We support a *section row* syntax: a row with length==1 spans all columns.
    items: list[tuple[str, list[str]]] = []
    for r in rows:
        if r is None:
            continue
        cells = list(r)
        if n_cols > 1 and len(cells) == 1:
            items.append(("section", ["" if cells[0] is None else str(cells[0])]))
            continue
        norm = [("" if c is None else str(c)) for c in cells[:n_cols]]
        if len(norm) < n_cols:
            norm += [""] * (n_cols - len(norm))
        items.append(("data", norm))
    # ---- Decide alignment (right-align common numeric columns)
    right_cols: set[int] = set()
    for j, h in enumerate(headers_s):
        if h.strip() in {"#", "SL", "TP1", "TP1/リム", "Risk", "R"}:
            right_cols.add(j)
    # ---- Compute column widths (content-aware, then clamp)
    col_content_px: list[int] = [0] * n_cols
    for j in range(n_cols):
        w_h, _ = _text_bbox(mdraw, headers_s[j], header_font, spacing=style.line_spacing)
        max_w = w_h
        for kind, cells in items:
            if kind != "data":
                continue
            w, _ = _text_bbox(mdraw, cells[j], body_font, spacing=style.line_spacing)
            if w > max_w:
                max_w = w
        max_w = min(max_w, style.max_col_px)
        col_content_px[j] = max_w
    col_px = [w + style.pad_x * 2 for w in col_content_px]
    table_inner_w = sum(col_px)
    max_inner_w = max(200, style.max_total_px - style.margin * 2)
    if table_inner_w > max_inner_w and table_inner_w > 0:
        scale = max_inner_w / table_inner_w
        # keep columns usable even after scaling
        min_col = max(64, int(max_inner_w / max(1, n_cols) * 0.45))
        col_px = [max(min_col, int(w * scale)) for w in col_px]
        # Fix rounding drift to match max_inner_w exactly.
        diff = max_inner_w - sum(col_px)
        j = 0
        while diff != 0 and n_cols > 0:
            if diff > 0:
                col_px[j] += 1
                diff -= 1
            else:
                if col_px[j] > min_col:
                    col_px[j] -= 1
                    diff += 1
            j = (j + 1) % n_cols
    table_inner_w = sum(col_px)
    # ---- Wrap/truncate text to the *actual* column widths
    headers_fit: list[str] = []
    for j, h in enumerate(headers_s):
        max_w = max(10, col_px[j] - style.pad_x * 2)
        headers_fit.append(fit_cell(h, header_font, max_w, max_lines=2))
    rows_fit: list[tuple[str, list[str]]] = []
    for kind, cells in items:
        if kind == "section":
            max_w = max(10, table_inner_w - style.pad_x * 2)
            rows_fit.append(("section", [fit_cell(cells[0], section_font, max_w, max_lines=2)]))
            continue
        fitted: list[str] = []
        for j, c in enumerate(cells):
            max_w = max(10, col_px[j] - style.pad_x * 2)
            fitted.append(fit_cell(c, body_font, max_w, max_lines=style.max_lines))
        rows_fit.append(("data", fitted))
    # ---- Heights
    # Wrap the title/subtitle to the table width so it never gets clipped.
    title_fit = _wrap_to_px(mdraw, title, title_font, max_px=table_inner_w)
    title_w, title_h = _text_bbox(mdraw, title_fit, title_font, spacing=style.line_spacing)
    head_h = 0
    for h in headers_fit:
        _, h_px = _text_bbox(mdraw, h, header_font, spacing=style.line_spacing)
        head_h = max(head_h, h_px)
    head_h = head_h + style.pad_y * 2
    row_heights: list[int] = []
    for kind, cells in rows_fit:
        if kind == "section":
            _, h_px = _text_bbox(mdraw, cells[0], section_font, spacing=style.line_spacing)
            row_heights.append(h_px + style.pad_y * 2)
        else:
            max_h = 0
            for c in cells:
                _, h_px = _text_bbox(mdraw, c, body_font, spacing=style.line_spacing)
                max_h = max(max_h, h_px)
            row_heights.append(max_h + style.pad_y * 2)
    table_h = head_h + sum(row_heights)
    img_w = style.margin * 2 + table_inner_w
    img_h = style.margin * 2 + title_h + 12 + table_h
    # ---- Render
    img = Image.new("RGB", (int(img_w), int(img_h)), "white")
    draw = ImageDraw.Draw(img)
    # Title
    draw.multiline_text(
        (style.margin, style.margin),
        title_fit,
        fill=style.text_color,
        font=title_font,
        spacing=style.line_spacing,
    )
    x0 = style.margin
    y0 = style.margin + title_h + 12
    x1 = x0 + table_inner_w
    # Helper: draw aligned multiline text inside a cell
    def draw_cell(
        x: int,
        y: int,
        w: int,
        h: int,
        text: str,
        font: ImageFont.ImageFont,
        *,
        align: str,
        fill: str,
    ) -> None:
        lines = (text or "").split("\n")
        # total text height
        _, th = _text_bbox(mdraw, text or "", font, spacing=style.line_spacing)
        ty = y + (h - th) / 2
        # estimate line step from actual bbox per-line
        cy = ty
        for i, line in enumerate(lines):
            lw, lh = _text_bbox(mdraw, line, font, spacing=style.line_spacing)
            if align == "center":
                tx = x + (w - lw) / 2
            elif align == "right":
                tx = x + w - style.pad_x - lw
            else:
                tx = x + style.pad_x
            draw.text((tx, cy), line, font=font, fill=fill)
            cy += lh + style.line_spacing
    # Header background
    draw.rectangle([x0, y0, x1, y0 + head_h], fill=style.header_bg)
    # Header text
    cx = x0
    for j, htxt in enumerate(headers_fit):
        draw_cell(int(cx), int(y0), int(col_px[j]), int(head_h), htxt, header_font, align="center", fill=style.text_color)
        cx += col_px[j]
    # Body rows: backgrounds + text
    y = y0 + head_h
    data_row_index = 0
    for i, (kind, cells) in enumerate(rows_fit):
        rh = row_heights[i]
        if kind == "section":
            draw.rectangle([x0, y, x1, y + rh], fill=style.section_bg)
            draw_cell(int(x0), int(y), int(table_inner_w), int(rh), cells[0], section_font, align="left", fill=style.text_color)
            y += rh
            continue
        if data_row_index % 2 == 1:
            draw.rectangle([x0, y, x1, y + rh], fill=style.zebra_bg)
        cx = x0
        for j, c in enumerate(cells):
            align = "right" if j in right_cols else "left"
            draw_cell(int(cx), int(y), int(col_px[j]), int(rh), c, body_font, align=align, fill=style.text_color)
            cx += col_px[j]
        data_row_index += 1
        y += rh
    # ---- Grid lines (draw last so they sit on top)
    xs: list[int] = [int(x0)]
    cur = x0
    for w in col_px:
        cur += w
        xs.append(int(cur))
    # Horizontal lines
    y_line = y0
    draw.line([x0, y_line, x1, y_line], fill=style.grid_color, width=style.line_width)
    y_line = y0 + head_h
    draw.line([x0, y_line, x1, y_line], fill=style.grid_color, width=style.line_width)
    y_cursor = y0 + head_h
    for rh in row_heights:
        y_cursor += rh
        draw.line([x0, y_cursor, x1, y_cursor], fill=style.grid_color, width=style.line_width)
    # Vertical lines: header uses full grid; body respects section rows (span)
    for x in xs:
        draw.line([x, y0, x, y0 + head_h], fill=style.grid_color, width=style.line_width)
    y_cursor = y0 + head_h
    for i, (kind, _cells) in enumerate(rows_fit):
        rh = row_heights[i]
        y_next = y_cursor + rh
        if kind == "section":
            for x in (x0, x1):
                draw.line([x, y_cursor, x, y_next], fill=style.grid_color, width=style.line_width)
        else:
            for x in xs:
                draw.line([x, y_cursor, x, y_next], fill=style.grid_color, width=style.line_width)
        y_cursor = y_next
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
