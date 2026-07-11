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
               pos_note: str, cfg, position_alerts: list[dict] | None = None,
               swing2w_res: dict | None = None, swing2w_alerts: list | None = None,
               swing2w_pos_note: str = "", swing2w_cfg=None) -> str:
    f = F(cfg)
    alerts = [a for a in (position_alerts or [])
             if a.get("state_c") or a.get("score_drop") or a.get("tob_jump") or a.get("state_c") is None]
    est_h = 600 + 70 + 90 + len(alerts) * 130 + len(res["picked"]) * 460 + len(res.get("watch", [])) * 120 + 360
    if swing2w_res is not None:
        from swing2w.report_png import estimate_height as _swing2w_h
        est_h += _swing2w_h(swing2w_res, swing2w_alerts)
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
    detail_lines = _wrap(d, regime.get("detail", regime.get("reason", "")), f(18), W - 2 * MARGIN - 40)
    box_h = 60 + len(detail_lines) * 25
    d.rounded_rectangle([MARGIN, y, W - MARGIN, y + box_h], 14,
                        fill=BG_ATTACK if attack else BG_DEFENSE,
                        outline=GREEN if attack else RED, width=3)
    text(MARGIN + 20, y + 8, regime.get("mode", "-"), 28, INK, True)
    yy = y + 50
    for ln in detail_lines:
        text(MARGIN + 20, yy, ln, 18, SUB)
        yy += 25
    y += box_h + 18

    if alerts:
        y = text(MARGIN, y, "保有銘柄アラート", 22, INK, True)
        for a in alerts:
            if a.get("state_c") or (a.get("tob_jump") and a.get("tob_stage") == "confirmed"):
                tag_col = RED
                tag = "状態C" if a.get("state_c") else "TOB疑い(要確認)"
            elif a.get("tob_jump") and a.get("tob_stage") == "day0":
                tag_col = GOLD
                tag = "急騰検知(参考)"
            elif a.get("score_drop"):
                tag_col = GOLD
                tag = "スコア劣化"
            else:
                tag_col = GOLD
                tag = "要確認"
            lines = _wrap(d, f"⚠{tag} {a['code']} {a['name']}: {a['note']}", f(17, True), W - 2 * MARGIN - 32)
            box_h = 16 + len(lines) * 23 + 10
            d.rounded_rectangle([MARGIN, y, W - MARGIN, y + box_h], 10, fill=(255, 246, 240), outline=tag_col, width=2)
            yy = y + 12
            for ln in lines:
                text(MARGIN + 14, yy, ln, 17, tag_col, True)
                yy += 23
            y += box_h + 8
        y += 4

    cov = meta.get("data_coverage", 0.0)
    recov_note = f" (2巡目で{meta['recovered_2nd_pass']}件回収)" if meta.get("recovered_2nd_pass") else ""
    y = text(MARGIN, y, f"データ {meta.get('data_ok',0)}/{meta.get('data_total',0)}({cov*100:.0f}%) "
             f"{meta.get('source','')}{recov_note}", 18, SUB)
    y = text(MARGIN, y, "単一ソース(yfinance)。本命は全件仮点灯 — 確定はiSPEED照合後。", 18, GOLD)
    if meta.get("data_warn"):
        y = text(MARGIN, y, f"⚠ データ被覆率不足(<{cfg.data_coverage_min*100:.0f}%) — プール精度に影響の可能性", 20, RED, True)
    st = res["stats"]
    sc = st.get("state_count", {})
    tob_note = f" (TOB疑い{st['tob_excluded']}件除外済)" if st.get("tob_excluded") else ""
    y = text(MARGIN, y, f"候補プール{st.get('pool_size',0)}銘柄{tob_note} → 状態A{sc.get('A',0)}/B{sc.get('B',0)}/C{sc.get('C',0)} "
             f"→ アクション候補{st.get('picked',0)}件(上限{cfg.max_positions}銘柄・1業種1銘柄)",
             18, SUB)
    if st.get("top_sectors"):
        sec_txt = " / ".join(f"{s}{n}" for s, n in st["top_sectors"])
        y = text(MARGIN, y, f"(指示⑨診断) プール業種上位{cfg.sector_diag_top_n}: {sec_txt}", 16, SUB)
    y = hline(y + 4)

    if res["picked"] or res.get("watch"):
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 52], 10, fill=(255, 240, 240), outline=RED, width=2)
        text(MARGIN + 16, y + 12, "[要確認] iSPEEDで適時開示(TOB/M&A/大量保有報告等)を確認するまで発注不可",
             19, RED, True)
        y += 64

    if st.get("regime_caution") and res["picked"]:
        cau_lines = _wrap(d, f"⚠相場全体が防御モード。個別シグナルの期待値はレジーム条件付き(Hanauer 2014等)。"
                          f"銘柄選定は通常どおりだが通常より慎重に。", f(18, True), W - 2 * MARGIN - 32)
        cau_h = 12 + len(cau_lines) * 24 + 8
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + cau_h], 10, fill=(255, 246, 225), outline=GOLD, width=2)
        yy = y + 12
        for ln in cau_lines:
            text(MARGIN + 14, yy, ln, 18, GOLD, True)
            yy += 24
        y += cau_h + 12

    state_col = {"A": GREEN, "B": GOLD}
    card_textw = W - 2 * MARGIN - 44
    for i, c in enumerate(res["picked"], 1):
        flag_lines = []
        for fl in c.flags:
            flag_lines.extend(_wrap(d, "⚠ " + fl, f(18, True), card_textw))
        check_lines = _wrap(d, "確認 → " + " / ".join(c.checks[:3]), f(18), card_textw)
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
        reason = "該当なし — ゼロ件はゼロ件。"
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 90], 14, fill=BG_CARD, outline=LINE, width=2)
        for ln in _wrap(d, reason, f(20, True), W - 2 * MARGIN - 40):
            text(MARGIN + 20, y + 26, ln, 20, SUB, True)
            break
        y += 106

    if res.get("watch"):
        BG_WATCH = (245, 247, 250)
        y = text(MARGIN, y, "参考層(本命の次点)", 22, INK, True)
        for c in res["watch"]:
            d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 96], 12, fill=BG_WATCH, outline=LINE, width=2)
            d.text((MARGIN + 16, y + 10), f"{c.code} {c.name}  [状態{c.state}]", font=f(20, True), fill=INK)
            d.text((MARGIN + 16, y + 42), f"{c.sector} / スコア{c.score:.1f}", font=f(17), fill=SUB)
            d.text((MARGIN + 16, y + 68), "監視のみ・直接エントリーは規律違反(3銘柄枠or同業種枠が空けば昇格)",
                   font=f(17, True), fill=GOLD)
            y += 108

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

    if swing2w_res is not None:
        from swing2w.report_png import render_section as _swing2w_render
        y = _swing2w_render(d, f, y, W, MARGIN, swing2w_res, swing2w_alerts, swing2w_pos_note, swing2w_cfg)

    img = img.crop((0, 0, W, min(est_h, y + MARGIN)))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    img.save(outpath, "PNG")
    return outpath
