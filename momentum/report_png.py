"""モメンタム・スクリーニング PNGインフォグラフィック — PILで直接描画."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W = 1080
MARGIN = 36
INK = (28, 32, 38)
SUB = (95, 103, 115)
LINE = (225, 228, 233)
RED = (200, 38, 60)
GREEN = (63, 106, 82)
GOLD = (169, 126, 31)
BLUE = (23, 92, 211)
BG_CARD = (248, 249, 251)
BG_ATTACK = (226, 235, 227)
BG_DEFENSE = (245, 227, 225)


class F:
    def __init__(self, cfg):
        self._cache = {}
        self.reg = cfg.font_path
        self.bold = cfg.font_path_bold

    def __call__(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self._cache:
            path = self.bold if bold else self.reg
            try:
                self._cache[key] = ImageFont.truetype(path, size, index=0)
            except Exception:
                self._cache[key] = ImageFont.load_default()
        return self._cache[key]


def _wrap(d, text, font, maxw):
    out, cur = [], ""
    for ch in text:
        b = d.textbbox((0, 0), cur + ch, font=font)
        if (b[2] - b[0]) > maxw:
            out.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def render_png(outpath: str, today: str, meta: dict, regime: dict, res: dict,
               pos_note: str, cfg) -> str:
    f = F(cfg)
    est_h = 520 + len(res["picked"]) * 420 + 360
    img = Image.new("RGB", (W, est_h), "white")
    d = ImageDraw.Draw(img)
    y = MARGIN

    def text(x, yy, s, size=22, color=INK, bold=False):
        d.text((x, yy), s, font=f(size, bold), fill=color)
        return yy + int(size * 1.5)

    def hline(yy):
        d.line([(MARGIN, yy), (W - MARGIN, yy)], fill=LINE, width=2)
        return yy + 14

    y = text(MARGIN, y, "モメンタム・スクリーニング", 29, GREEN, True)
    y = text(MARGIN, y, today, 21, SUB)
    y += 6

    attack = regime.get("attack", False)
    d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 92], 14,
                        fill=BG_ATTACK if attack else BG_DEFENSE,
                        outline=GREEN if attack else RED, width=3)
    text(MARGIN + 20, y + 8, regime.get("mode", "-"), 28, INK, True)
    for ln in _wrap(d, regime.get("detail", regime.get("reason", "")), f(18), W - 2 * MARGIN - 40):
        text(MARGIN + 20, y + 50, ln, 18, SUB)
        break
    y += 110

    cov = meta.get("data_coverage", 0.0)
    y = text(MARGIN, y, f"データ {meta.get('data_ok',0)}/{meta.get('data_total',0)}({cov*100:.0f}%) "
             f"{meta.get('source','')}", 18, SUB)
    y = text(MARGIN, y, "単一ソース(yfinance)。本命は全件仮点灯 — 確定はiSPEED照合後。", 18, GOLD)
    st = res["stats"]
    sc = st.get("state_count", {})
    y = text(MARGIN, y, f"候補プール{st.get('pool_size',0)}銘柄 → 状態A{sc.get('A',0)}/B{sc.get('B',0)}/C{sc.get('C',0)} "
             f"→ アクション候補{st.get('picked',0)}件(上限{cfg.max_positions}銘柄・1業種1銘柄)",
             18, SUB)
    y = hline(y + 4)

    state_col = {"A": GREEN, "B": GOLD}
    card_textw = W - 2 * MARGIN - 44
    for i, c in enumerate(res["picked"], 1):
        flag_lines = []
        for fl in c.flags:
            flag_lines.extend(_wrap(d, "⚠ " + fl, f(18, True), card_textw))
        check_lines = _wrap(d, "確認 → " + " / ".join(c.checks[:2]), f(18), card_textw)
        fixed_h = 16 + 52 + 34 + 92 + 28 + 28  # header+meta+price boxes+risk line+trail note
        ch = fixed_h + len(flag_lines) * 25 + len(check_lines) * 24 + 16

        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + ch], 16, fill=BG_CARD, outline=LINE, width=2)
        cx, cy = MARGIN + 22, y + 16
        col = state_col.get(c.state, GREEN)
        d.rounded_rectangle([cx, cy, cx + 110, cy + 42], 10, fill=col)
        d.text((cx + 14, cy + 6), f"状態{c.state}", font=f(22, True), fill="white")
        d.text((cx + 124, cy - 2), f"{i}. {c.code} {c.name}", font=f(26, True), fill=INK)
        cy += 52
        d.text((cx, cy), f"{c.sector} / {c.market} / スコア{c.score:.1f}", font=f(18), fill=SUB)
        cy += 34

        cols = [("エントリー", c.entry, INK), ("初期ストップ", c.stop, RED), ("シャンデリア", c.chandelier, col)]
        bw = (W - 2 * MARGIN - 44 - 2 * 12) // 3
        for j, (lab, val, vc) in enumerate(cols):
            x0 = cx + j * (bw + 12)
            d.rounded_rectangle([x0, cy, x0 + bw, cy + 76], 10, fill="white", outline=LINE, width=2)
            d.text((x0 + 12, cy + 8), lab, font=f(17), fill=SUB)
            d.text((x0 + 12, cy + 32), f"{val:,.0f}円", font=f(24, True), fill=vc)
        cy += 92

        lot = f"リスク {c.risk_pct:.2f}%" + (f" ≈{c.shares:,}株" if c.shares else "")
        d.text((cx, cy), f"{lot}  リスク幅{c.risk_w/c.entry*100:.1f}%", font=f(19, True), fill=INK)
        cy += 28
        d.text((cx, cy), "固定利確なし。シャンデリア水準を割るまで保有(トレーリング)。",
               font=f(18), fill=SUB)
        cy += 28
        for ln in flag_lines:
            d.text((cx, cy), ln, font=f(18, True), fill=GOLD); cy += 25
        for ln in check_lines:
            d.text((cx, cy), ln, font=f(18), fill=BLUE); cy += 24
        y += ch + 16

    if not res["picked"]:
        reason = st.get("blocked_reason") or "該当なし — ゼロ件はゼロ件。"
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 90], 14, fill=BG_CARD, outline=LINE, width=2)
        for ln in _wrap(d, reason, f(20, True), W - 2 * MARGIN - 40):
            text(MARGIN + 20, y + 26, ln, 20, SUB, True)
            break
        y += 106

    y = hline(y + 2)
    y = text(MARGIN, y, f"総リスク: 新規計 {st['total_risk']:.2f}% / 上限 {st['risk_cap']:.1f}%   |   {pos_note}", 19, INK)
    y = hline(y + 4)
    for s, col, bold in [
        ("出口: 固定利確なし。シャンデリア水準(直近22日高値-3×ATR)割れで手仕舞い。", SUB, False),
        ("行動注意: 防御モードの日は無理をしない。候補が出ても必ず取引する必要はない。", GOLD, True),
        ("免責: 投資助言ではない。モメンタムクラッシュのリスクは対策後も完全には消えない。", SUB, False),
        ("最終判断と結果責任はユーザーにある。数値は必ず自分で再確認。", SUB, False),
    ]:
        for ln in _wrap(d, s, f(18, bold), W - 2 * MARGIN):
            y = text(MARGIN, y, ln, 18, col, bold)

    img = img.crop((0, 0, W, min(est_h, y + MARGIN)))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    img.save(outpath, "PNG")
    return outpath
