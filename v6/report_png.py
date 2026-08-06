"""v6 — PNGレポート。v5の視覚的な枠組み(配色・マスコット合成・カードデザイン)を踏襲し、
中身をv6のデータ構造(絞り込みファネル・週足/日足ステージ・RS・セットアップ詳細・
リスク計画・保有出口シグナル)に合わせて再構成する。

v5と同じく、素材(assets/hero.*・assets/mascot.*)があれば背景・マスコットとして合成し、
無ければ図形描画のクマで代替する。
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .filters import STAGE_D_NAMES
from .setups import SETUP_VCP, SETUP_PULL, SETUP_PIVOT

# --- 配色(v5と同じ、温かいクリーム/ブラウン系) ---
BG        = (243, 234, 216)
CARD      = (252, 247, 238)
CARD_ALT  = (248, 241, 227)
BORDER    = (216, 199, 168)
HEADER    = (74, 55, 40)
CREAM     = (250, 245, 235)
INK       = (59, 47, 35)
SUB       = (122, 106, 85)
GOLD      = (176, 130, 30)
RED       = (183, 48, 42)
GREEN     = (43, 110, 78)
BLUE      = (44, 95, 138)
PAW       = (214, 196, 165)

TAG_COLORS = {SETUP_VCP: (44, 95, 138), SETUP_PULL: (191, 144, 0), SETUP_PIVOT: (110, 80, 150)}

W, MARGIN = 1200, 30
CARD_W = W - 2 * MARGIN

STAGE_W_NAMES = {1: "底固め", 2: "上昇期", 3: "天井圏", 4: "下降期"}


# ★2026-08: 日本語フォントが見つからないと、PILがCJK非対応の代替フォントに落ちて
# 日本語だけ「豆腐(□)」になる事故があった(2026-08-06、本番で発覚)。英数字は
# 正常に見えるため気づきにくい。cfg.font_pathが見つからない場合に備え、Ubuntu/Debian
# でよく使われる複数のインストール先を順に試す保険を入れる。
_FALLBACK_REG = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto-cjk/NotoSansCJK-Regular.ttc",
]
_FALLBACK_BOLD = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Bold.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto-cjk/NotoSansCJK-Bold.ttc",
]


class F:
    def __init__(self, cfg):
        self._cache = {}
        self.reg = cfg.font_path
        self.bold = cfg.font_path_bold
        self.warned = False

    def _load(self, path: str, size: int):
        candidates = [path] + (_FALLBACK_BOLD if path == self.bold else _FALLBACK_REG)
        for p in candidates:
            try:
                return ImageFont.truetype(p, size, index=0), p
            except Exception:
                continue
        return None, None

    def __call__(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self._cache:
            path = self.bold if bold else self.reg
            font, used = self._load(path, size)
            if font is None:
                if not self.warned:
                    print("[warn] 日本語フォントが1つも見つからない。"
                          "PNGの日本語が文字化けする可能性がある。"
                          "GitHub Actionsなら fonts-noto-cjk のインストールを確認すること。")
                    self.warned = True
                font = ImageFont.load_default()
            self._cache[key] = font
        return self._cache[key]


def _wrap(d, text, font, maxw):
    out, cur = [], ""
    for ch in text:
        if d.textbbox((0, 0), cur + ch, font=font)[2] > maxw:
            out.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _paw(d, cx, cy, s=1.0, color=PAW):
    r = 11 * s
    d.ellipse([cx - r * 1.15, cy - r * 0.55, cx + r * 1.15, cy + r * 1.25], fill=color)
    for dx, dy, rr in ((-1.25, -1.35, 0.52), (-0.42, -1.75, 0.52),
                       (0.42, -1.75, 0.52), (1.25, -1.35, 0.52)):
        d.ellipse([cx + dx * r - rr * r, cy + dy * r - rr * r,
                   cx + dx * r + rr * r, cy + dy * r + rr * r], fill=color)


FUR, FUR_DARK, FUR_IN, NOSE, BAND = (198, 160, 106), (172, 133, 82), (233, 212, 178), (74, 55, 40), (250, 245, 235)


def _bear(d, cx, cy, r, band: bool = True):
    """クマの顔を図形で描く(マスコット画像が無いときの代替)。"""
    cx, cy = int(round(cx)), int(round(cy))
    I = lambda v: int(round(v * r))
    eo, er, ir = I(0.78), I(0.36), I(0.19)
    ey = cy - I(0.74)
    for sx in (-1, 1):
        ex = cx + sx * eo
        d.ellipse([ex - er, ey - er, ex + er, ey + er], fill=FUR_DARK)
        d.ellipse([ex - ir, ey - ir, ex + ir, ey + ir], fill=FUR_IN)
    R = I(1.0)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=FUR)
    if band:
        t0, t1 = -0.52, -0.26
        hw0 = int(round(r * math.sqrt(max(0.0, 1 - t0 * t0))))
        hw1 = int(round(r * math.sqrt(max(0.0, 1 - t1 * t1))))
        y0, y1 = cy + I(t0), cy + I(t1)
        d.polygon([(cx - hw0, y0), (cx + hw0, y0), (cx + hw1, y1), (cx - hw1, y1)], fill=BAND)
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=FUR_DARK, width=max(1, I(0.05)))
    xo, ew = I(0.40), I(0.20)
    ey0, ey1 = cy - I(0.14), cy + I(0.20)
    for sx in (-1, 1):
        exc = cx + sx * xo
        d.arc([exc - ew, ey0, exc + ew, ey1], start=200, end=340, fill=NOSE, width=max(2, I(0.085)))
    mw, my0 = I(0.46), cy + I(0.10)
    d.ellipse([cx - mw, my0, cx + mw, my0 + 2 * I(0.32)], fill=FUR_IN)
    nw, nh, ny = I(0.16), I(0.115), cy + I(0.22)
    d.ellipse([cx - nw, ny - nh, cx + nw, ny + nh], fill=NOSE)
    lw = max(2, I(0.06))
    a0, a1 = I(0.02), I(0.30)
    my1, my2 = cy + I(0.30), cy + I(0.54)
    for sx in (-1, 1):
        x0 = cx + (a0 if sx > 0 else -a1)
        x1 = cx + (a1 if sx > 0 else -a0)
        d.arc([x0, my1, x1, my2], start=15, end=165, fill=NOSE, width=lw)


def _asset(stem: str, cfg):
    for dpath in (Path("."), Path("assets"), Path(getattr(cfg, "outdir", "."))):
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            p = dpath / (stem + ext)
            try:
                if p.exists():
                    return Image.open(p).convert("RGBA")
            except Exception:
                pass
    return None


def _mascot(cfg):
    return _asset("mascot", cfg)


def _hero(cfg):
    return _asset("hero", cfg)


def _paste_bg(img, photo, H: int, photo_h: int = 620, wash: float = 0.16, tex_wash: float = 0.62):
    pw, ph = photo.size
    p = photo.convert("RGB")
    tex = p.crop((0, int(ph * 0.72), pw, ph)).resize((W, max(1, H)), Image.LANCZOS)
    tex = tex.filter(ImageFilter.GaussianBlur(26))
    tex = Image.blend(tex, Image.new("RGB", tex.size, BG), tex_wash)
    img.paste(tex, (0, 0))
    sw = max(1, int(pw * photo_h / ph))
    r = p.resize((sw, photo_h), Image.LANCZOS)
    r = Image.blend(r, Image.new("RGB", r.size, BG), wash)
    x0 = max(0, W - sw)
    mask = Image.new("L", (sw, photo_h), 255)
    md = ImageDraw.Draw(mask)
    fx = int(sw * 0.16)
    for i in range(fx):
        md.line([(i, 0), (i, photo_h)], fill=int(255 * i / fx))
    fy = int(photo_h * 0.26)
    for i in range(fy):
        v = int(255 * (1 - i / fy))
        yy = photo_h - fy + i
        for x in range(0, sw, 1):
            cur = mask.getpixel((x, yy))
            if cur > v:
                mask.putpixel((x, yy), v)
    img.paste(r, (x0, 0), mask)


def _header(d, img, f, mascot_at, title_main: str, title_sub: str, today: str, TOPW: int):
    y = MARGIN + 14
    d.rounded_rectangle([MARGIN + 62, y, MARGIN + TOPW, y + 76], 20, fill=HEADER)
    t1, t2 = title_main, title_sub
    x_title, avail = MARGIN + 118, (MARGIN + TOPW) - (MARGIN + 118) - 46
    s1, s2 = 30, 26
    while s1 > 15:
        tw = d.textbbox((0, 0), t1, font=f(s1, True))[2]
        tw2 = d.textbbox((0, 0), t2, font=f(s2, True))[2]
        if tw + 4 + tw2 <= avail:
            break
        s1 -= 1; s2 -= 1
    d.text((x_title, y + 18 + (30 - s1) // 2), t1, font=f(s1, True), fill=(226, 186, 118))
    d.text((x_title + tw + 4, y + 22 + (26 - s2) // 2), t2, font=f(s2, True), fill=CREAM)
    _paw(d, x_title + tw + 4 + tw2 + 24, y + 40, 0.72, (196, 172, 132))
    mascot_at(MARGIN - 4, y - 14, 92)
    y += 76 + 12
    d.rounded_rectangle([MARGIN, y, MARGIN + 176, y + 36], 12, fill=CARD_ALT, outline=BORDER, width=2)
    d.text((MARGIN + 18, y + 6), today, font=f(20, True), fill=SUB)
    return y + 36 + 12


def render_png(outpath: str, today: str, res: dict, exits: list, cfg) -> str:
    f = F(cfg)
    counts = res["counts"]
    dd = res["dd"]
    picked = res["picked"]

    exit_notable = [e for e in exits if e.get("top")]
    est_h = (900 + len(picked) * 260 + len(exit_notable) * 110 + 400)
    img = Image.new("RGB", (W, est_h), BG)
    hero = _hero(cfg)
    if hero is not None:
        _paste_bg(img, hero, est_h)
    d = ImageDraw.Draw(img)
    m = _mascot(cfg)
    HAS_PHOTO = hero is not None

    def mascot_at(x, y, size):
        if m is not None:
            mm = m.resize((size, size), Image.LANCZOS)
            img.paste(mm, (int(x), int(y)), mm)
        else:
            _bear(d, x + size / 2, y + size / 2, size / 2 * 0.86)

    def fit(txt, maxw, size, bold=True, minsize=14):
        while size > minsize and d.textbbox((0, 0), txt, font=f(size, bold))[2] > maxw:
            size -= 1
        return size

    TOPW = int(CARD_W * 0.52) if HAS_PHOTO else CARD_W

    def card(y, h, fill=CARD, outline=BORDER, width=2, r=16, w=None, x=MARGIN):
        d.rounded_rectangle([x, y, x + (w if w else CARD_W), y + h], r, fill=fill, outline=outline, width=width)

    y = _header(d, img, f, mascot_at, "AIトム", "のスクリーニング v6", today, TOPW)

    # ================= 絞り込みファネル =================
    funnel = (f"{counts['universe']} → 流動性{counts['layer0']} → 週足St2 {counts['layer1']}"
              f" → TT+RS {counts['layer2']} → 日足St {counts['layer3']}"
              f" → 点灯{counts['layer4']} → 通知{counts['layer5']}")
    lines = [(f"絞り込み: {funnel}", 18, INK, True)]
    # 点灯したのに通知に至らなかった理由の内訳(設計不具合の早期発見のため)
    bs = counts.get("by_setup") or {}
    if bs:
        lines.append(("点灯内訳: " + " / ".join(f"{k}{v}" for k, v in bs.items()), 15, SUB, False))
    drops = res.get("drops") or []
    if drops:
        by_reason = {}
        for dr in drops:
            k = dr.get("layer", "不明")
            by_reason[k] = by_reason.get(k, 0) + 1
        lines.append(("脱落: " + " / ".join(f"{k} {v}件" for k, v in sorted(by_reason.items())),
                      15, SUB, False))
    h = 20 + sum(len(_wrap(d, s_, f(sz, b_), TOPW - 40)) * int(sz * 1.5) for s_, sz, _, b_ in lines)
    card(y, h, w=TOPW)
    yy = y + 12
    for s_, sz, c_, b_ in lines:
        for ln in _wrap(d, s_, f(sz, b_), TOPW - 40):
            d.text((MARGIN + 20, yy), ln, font=f(sz, b_), fill=c_); yy += int(sz * 1.5)
    y += h + 12

    # ================= 資金状態バナー =================
    heat_ok = res["can_new"]
    col = GREEN if heat_ok else RED
    bg = (234, 244, 236) if heat_ok else (250, 233, 229)
    ban_h = 96
    card(y, ban_h, fill=bg, outline=col, width=3, w=TOPW)
    mascot_at(MARGIN + 14, y + 20, 60)
    headline = f"ヒート{res['heat']*100:.1f}%(上限{cfg.portfolio_heat_max*100:.0f}%) / DD{dd['dd']*100:.1f}%"
    d.text((MARGIN + 88, y + 14), headline, font=f(fit(headline, TOPW - 104, 24), True), fill=col)
    yy = y + 46
    if not heat_ok:
        for ln in _wrap(d, res["gate_reason"], f(16, True), TOPW - 104):
            d.text((MARGIN + 88, yy), ln, font=f(16, True), fill=RED); yy += 22
    else:
        d.text((MARGIN + 88, yy), "新規発注: 可", font=f(16), fill=SUB); yy += 22
    y += ban_h + 12

    if not picked:
        card(y, 58, w=TOPW if False else CARD_W)
        d.text((MARGIN + 20, y + 16), "本日の新規候補 — 該当なし(0件は正常な結果)",
               font=f(20, True), fill=SUB)
        y += 58 + 14

    # ================= 候補カード =================
    BOXW = 182
    rx = MARGIN + 26 + BOXW * 3 + 20 + 22
    rw = W - MARGIN - 22 - rx

    for i, c in enumerate(picked, 1):
        setup_label = {SETUP_VCP: SETUP_VCP, SETUP_PULL: SETUP_PULL, SETUP_PIVOT: SETUP_PIVOT}.get(c.setup, c.setup)
        detail_line = ""
        if c.setup == SETUP_VCP:
            dtl = c.detail
            detail_line = f"収縮 {dtl.get('n_contractions','-')}回 → {dtl.get('final_depth',0)*100:.1f}% / 出来高比{dtl.get('vol_ratio','-')}"
        elif c.setup == SETUP_PULL:
            dtl = c.detail
            detail_line = f"押し {dtl.get('depth',0)*100:.1f}% / 高値から{dtl.get('days_since_high','-')}日"
        elif c.setup == SETUP_PIVOT:
            dtl = c.detail
            detail_line = f"上抜け +{dtl.get('extension',0)*100:.1f}% / 出来高比{dtl.get('vol_ratio','-')}"

        stage_line = (f"週足St{c.stage_w}({STAGE_W_NAMES.get(c.stage_w,'-')} 30週線{c.slope_w*100:+.1f}%)"
                      f" RS{c.rs_rating:.0f}  日足St{c.stage_d}(滞在{c.stage_days}日)")

        rlines = [(f"{c.order_type} {c.entry:,.0f}円", True)]
        rlines = [(ln, True) for ln in _wrap(d, rlines[0][0], f(20, True), rw)]
        rlines += [(ln, False) for ln in _wrap(d, f"注文期限 {c.expire_date}まで", f(15), rw)]
        rlines += [(ln, False) for ln in _wrap(d, f"利確目安 +2R = {c.target_2r:,.0f}円", f(15), rw)]
        if c.score_parts:
            rlines += [(ln, False) for ln in _wrap(d, "スコア " + " / ".join(c.score_parts), f(14), rw)]

        trig_lines = _wrap(d, stage_line, f(16), CARD_W - 60) + _wrap(d, detail_line, f(16), CARD_W - 60)
        risk_lines = []
        for fl in c.flags:
            risk_lines.extend(_wrap(d, "⚠ " + fl, f(16), CARD_W - 60))
        if not c.tradable:
            risk_lines.extend(_wrap(d, "🚫 発注不可", f(16, True), CARD_W - 60))
        risk_lines.extend(_wrap(d, "⚠ ニュース内容は未確認。発注前に自分の目で確認すること", f(15), CARD_W - 60))

        mid_h = max(84, len(rlines) * 23 + 6)
        ch = 16 + 46 + 24 + len(trig_lines) * 22 + mid_h + len(risk_lines) * 21 + 18
        card(y, ch)
        _paw(d, W - MARGIN - 36, y + 30, 0.72)

        cx, cy = MARGIN + 26, y + 16
        d.ellipse([cx, cy + 2, cx + 38, cy + 40], fill=CARD_ALT, outline=BORDER, width=2)
        d.text((cx + (12 if i < 10 else 6), cy + 8), str(i), font=f(20, True), fill=SUB)
        mascot_at(cx + 46, cy + 2, 38)
        tagcol = TAG_COLORS.get(c.setup, BLUE)
        tx = cx + 94
        tagw = d.textbbox((0, 0), setup_label, font=f(19, True))[2] + 28
        d.rounded_rectangle([tx, cy + 2, tx + tagw, cy + 40], 10, fill=tagcol)
        d.text((tx + 14, cy + 9), setup_label, font=f(19, True), fill=(255, 252, 246))
        scx = tx + tagw + 10
        d.rounded_rectangle([scx, cy + 2, scx + 96, cy + 40], 10, fill=CARD_ALT, outline=BORDER, width=2)
        d.text((scx + 14, cy + 10), f"score {c.score}", font=f(17, True), fill=SUB)
        d.text((scx + 106, cy), f"{c.code} {c.name}", font=f(fit(f"{c.code} {c.name}", CARD_W - (scx+106-MARGIN) - 30, 26), True), fill=INK)
        cy += 46

        for ln in trig_lines:
            d.text((cx, cy), ln, font=f(16), fill=SUB); cy += 22

        for k, (bx, oc, lbl, v1, v2, c1, c2) in enumerate((
                (cx, GREEN, "指値/逆指値", f"{c.entry:,.0f} 円", c.order_type, INK, SUB),
                (cx + BOXW + 10, RED, "STOP(構造)", f"{c.stop:,.0f} 円", f"{-c.stop_pct*100:.1f}% / {c.stop_kind}", RED, SUB),
                (cx + 2 * (BOXW + 10), BORDER, "株数・リスク",
                 f"{c.shares:,}株", f"{c.risk_amount:,.0f}円({c.risk_pct*100:.1f}%)", INK, GREEN))):
            d.rounded_rectangle([bx, cy, bx + BOXW, cy + 78], 12, fill=(255, 253, 249), outline=oc, width=2)
            d.text((bx + 12, cy + 8), lbl, font=f(15), fill=SUB)
            d.text((bx + 12, cy + 30), v1, font=f(20, True), fill=c1)
            d.text((bx + 12, cy + 56), v2, font=f(14), fill=c2)

        ry = cy + 2
        for ln, bold in rlines:
            d.text((rx, ry), ln, font=f(17, True) if bold else f(15), fill=INK if bold else SUB)
            ry += 23
        cy += mid_h

        for ln in risk_lines:
            d.text((cx, cy), ln, font=f(15), fill=GOLD); cy += 21
        y += ch + 14

    # ================= 保有銘柄シグナル =================
    if exit_notable:
        y = _section_title(d, f, y, "保有銘柄シグナル")
        for e in exit_notable:
            top = e["top"]
            tagcol = {1: RED, 2: RED, 3: GOLD, 4: GOLD, 5: GREEN, 6: GREEN, 7: SUB}.get(top[0], SUB)
            lns = _wrap(d, top[2], f(16), CARD_W - 240)
            bh2 = 14 + 24 + len(lns) * 21 + 10
            card(y, bh2, fill=CARD_ALT, r=12)
            d.text((MARGIN + 20, y + 10), f"[優先{top[0]}] {e['code']} {e['name']}: {top[1]}",
                   font=f(18, True), fill=tagcol)
            yy = y + 36
            for ln in lns:
                d.text((MARGIN + 20, yy), ln, font=f(16), fill=SUB); yy += 21
            y += bh2 + 8
        y += 6

    # ================= フッター =================
    for s_ in ["テクニカル分析に基づく抽出であり投資助言ではない。通知は「調べる価値があるかもしれない」という意味に留まる。",
               "発注可否の最終判断と結果の責任は運用者にある。"]:
        for ln in _wrap(d, s_, f(16), CARD_W):
            d.text((MARGIN + 4, y), ln, font=f(16), fill=SUB); y += 24
    y += 8
    sig = "stockbotTOM v6"
    sw = d.textbbox((0, 0), sig, font=f(19, True))[2]
    _paw(d, W / 2 - sw / 2 - 22, y + 12, 0.62, (150, 124, 90))
    d.text((W / 2 - sw / 2, y), sig, font=f(19, True), fill=(110, 88, 62))
    y += 34

    img = img.crop((0, 0, W, min(est_h, y + MARGIN)))
    img.save(outpath)
    return outpath


def _section_title(d, f, y, text):
    d.text((MARGIN + 4, y + 4), text, font=f(21, True), fill=INK)
    return y + 4 + 30


def render_summary_png(outpath: str, today: str, res: dict, cfg) -> str:
    """まとめ(2枚目) — ファネル・資金状態・ランキングだけを1枚に。"""
    f = F(cfg)
    counts = res["counts"]
    dd = res["dd"]
    picked = res["picked"]

    est_h = 900 + len(picked) * 86
    img = Image.new("RGB", (W, est_h), BG)
    hero = _hero(cfg)
    if hero is not None:
        _paste_bg(img, hero, est_h)
    d = ImageDraw.Draw(img)
    m = _mascot(cfg)
    HAS_PHOTO = hero is not None

    def mascot_at(x, y, size):
        if m is not None:
            mm = m.resize((size, size), Image.LANCZOS)
            img.paste(mm, (int(x), int(y)), mm)
        else:
            _bear(d, x + size / 2, y + size / 2, size / 2 * 0.86)

    def fit(txt, maxw, size, bold=True, minsize=14):
        while size > minsize and d.textbbox((0, 0), txt, font=f(size, bold))[2] > maxw:
            size -= 1
        return size

    TOPW = int(CARD_W * 0.52) if HAS_PHOTO else CARD_W

    def card(y, h, fill=CARD, outline=BORDER, width=2, r=16, w=None, x=MARGIN):
        d.rounded_rectangle([x, y, x + (w if w else CARD_W), y + h], r, fill=fill, outline=outline, width=width)

    y = _header(d, img, f, mascot_at, "AIトム", "のまとめ v6", today, TOPW)

    funnel = (f"{counts['universe']} → 流動性{counts['layer0']} → 週足St2 {counts['layer1']}"
              f" → TT+RS {counts['layer2']} → 日足St {counts['layer3']}"
              f" → 点灯{counts['layer4']} → 通知{counts['layer5']}")
    heat_ok = res["can_new"]
    col = GREEN if heat_ok else RED
    bg = (234, 244, 236) if heat_ok else (250, 233, 229)
    lines2 = _wrap(d, f"絞り込み: {funnel}", f(16), TOPW - 104)
    ban_h = 60 + len(lines2) * 22
    card(y, ban_h, fill=bg, outline=col, width=3, w=TOPW)
    mascot_at(MARGIN + 14, y + 20, 60)
    headline = f"ヒート{res['heat']*100:.1f}% / DD{dd['dd']*100:.1f}%"
    d.text((MARGIN + 88, y + 14), headline, font=f(fit(headline, TOPW - 104, 24), True), fill=col)
    yy = y + 46
    for ln in lines2:
        d.text((MARGIN + 88, yy), ln, font=f(16), fill=SUB); yy += 22
    y += ban_h + 14

    head = f"本日の候補 {len(picked)}件"
    rows_h = max(1, len(picked)) * 86
    rk_h = 58 + rows_h + 16
    card(y, rk_h)
    d.text((MARGIN + 22, y + 16), head, font=f(fit(head, CARD_W - 60, 21), True), fill=INK)
    _paw(d, W - MARGIN - 36, y + 30, 0.72)
    ry = y + 58
    if not picked:
        d.text((MARGIN + 26, ry + 16), "該当なし — 0件は正常な結果", font=f(20, True), fill=SUB)
    for i, c in enumerate(picked, 1):
        d.rounded_rectangle([MARGIN + 16, ry, W - MARGIN - 16, ry + 74], 12, fill=CARD_ALT, outline=BORDER, width=2)
        d.ellipse([MARGIN + 30, ry + 17, MARGIN + 70, ry + 57], fill=CARD, outline=BORDER, width=2)
        label = str(i)
        rw_ = d.textbbox((0, 0), label, font=f(22, True))[2]
        d.text((MARGIN + 50 - rw_ / 2, ry + 25), label, font=f(22, True), fill=SUB)
        mascot_at(MARGIN + 82, ry + 17, 40)
        tagcol = TAG_COLORS.get(c.setup, BLUE)
        d.rounded_rectangle([MARGIN + 138, ry + 16, MARGIN + 138 + 80, ry + 44], 8, fill=tagcol)
        d.text((MARGIN + 148, ry + 20), c.setup, font=f(14, True), fill=(255, 252, 246))
        name = f"{c.code}  {c.name}"
        d.text((MARGIN + 232, ry + 16),
               name, font=f(fit(name, CARD_W - 300, 24), True), fill=INK)
        d.text((MARGIN + 232, ry + 46), f"score {c.score} / RS{c.rs_rating:.0f}", font=f(15), fill=SUB)
        ry += 86
    y += rk_h + 14

    for s_ in ["詳細(指値・ストップ・リスク金額)は1枚目を参照。",
               "投資助言ではない。最終判断と結果の責任は運用者にある。"]:
        for ln in _wrap(d, s_, f(16), CARD_W):
            d.text((MARGIN + 4, y), ln, font=f(16), fill=SUB); y += 24
    y += 8
    sig = "stockbotTOM v6"
    sw = d.textbbox((0, 0), sig, font=f(19, True))[2]
    _paw(d, W / 2 - sw / 2 - 22, y + 12, 0.62, (150, 124, 90))
    d.text((W / 2 - sw / 2, y), sig, font=f(19, True), fill=(110, 88, 62))
    y += 34

    img = img.crop((0, 0, W, min(est_h, y + MARGIN)))
    img.save(outpath)
    return outpath
