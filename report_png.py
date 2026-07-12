"""swing2w — レポートのPNGセクション生成(momentum側の統合キャンバスに描画される前提)."""

from __future__ import annotations

from momentum.report_png import _wrap, INK, SUB, LINE, RED, GREEN, GOLD, BLUE, BG_CARD


def estimate_height(res: dict, position_alerts: list) -> int:
    notable_pos = [a for a in (position_alerts or [])
                  if a.get("hit") or a.get("hit") is None or a.get("breakeven_due")]
    return (120 + len(notable_pos) * 90 + len(res["picked"]) * 300
            + len(res.get("watch", [])) * 80 + 140)


def render_section(d, f, y: int, W: int, MARGIN: int, res: dict, position_alerts: list,
                   pos_note: str, cfg) -> int:
    def text(x, yy, s, size=22, color=INK, bold=False):
        d.text((x, yy), s, font=f(size, bold), fill=color)
        return yy + int(size * 1.5)

    def hline(yy):
        d.line([(MARGIN, yy), (W - MARGIN, yy)], fill=LINE, width=2)
        return yy + 14

    y = hline(y + 6)
    y = text(MARGIN, y, "2週間スイング(回転率二分・固定利確+時間ストップ)", 26, GREEN, True)

    notable_pos = [a for a in (position_alerts or [])
                  if a.get("hit") or a.get("hit") is None or a.get("breakeven_due")]
    if notable_pos:
        for a in notable_pos:
            if a.get("hit"):
                tag_col = RED if a.get("hit") in ("stop", "time") else GREEN
                tag = {"target": "利確到達", "stop": "ストップ到達", "time": "時間ストップ"}.get(a.get("hit"), "要確認")
            elif a.get("breakeven_due"):
                tag_col = GREEN
                tag = "建値移動検討"
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

    st = res["stats"]
    y = text(MARGIN, y, f"母集団{st.get('universe_considered',0)}銘柄(TOB疑い{st.get('tob_excluded',0)}件除外済)", 18, SUB)
    y = text(MARGIN, y, f"低回転率{st.get('low_turnover_n',0)}銘柄(R対象) / 高回転率{st.get('high_turnover_n',0)}銘柄(M対象) "
             f"→ 点灯R{st.get('fired_r',0)}/M{st.get('fired_m',0)} → 採用{st.get('picked',0)}件"
             f"(別枠上限{cfg.max_positions}・1業種1銘柄)", 18, SUB)
    y += 4

    if res["picked"] or res.get("watch"):
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 52], 10, fill=(255, 240, 240), outline=RED, width=2)
        text(MARGIN + 16, y + 12, "[要確認] iSPEEDで適時開示を確認するまで発注不可", 19, RED, True)
        y += 64

    eng_col = {"R": GREEN, "M": GOLD}
    card_textw = W - 2 * MARGIN - 44
    for i, c in enumerate(res["picked"], 1):
        flag_lines = []
        for fl in c.flags:
            flag_lines.extend(_wrap(d, "⚠ " + fl, f(17, True), card_textw))
        trig_lines = _wrap(d, f"トリガー: {c.trigger}", f(17), card_textw)
        fixed_h = 16 + 52 + 92 + 28 + 28
        ch = fixed_h + len(trig_lines) * 22 + len(flag_lines) * 23 + 16

        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + ch], 16, fill=BG_CARD, outline=LINE, width=2)
        cx, cy = MARGIN + 22, y + 16
        col = eng_col.get(c.engine, GREEN)
        eng_label = "R反転" if c.engine == "R" else "M入口"
        d.rounded_rectangle([cx, cy, cx + 110, cy + 42], 10, fill=col)
        d.text((cx + 14, cy + 6), eng_label, font=f(20, True), fill="white")
        d.text((cx + 124, cy - 2), f"{i}. {c.code} {c.name}", font=f(26, True), fill=INK)
        cy += 52
        for ln in trig_lines:
            d.text((cx, cy), ln, font=f(17), fill=SUB); cy += 22

        cols = [("エントリー", c.entry, INK), ("初期ストップ", c.stop, RED), ("固定利確目標", c.target, GREEN)]
        bw = (W - 2 * MARGIN - 44 - 2 * 12) // 3
        for j, (lab, val, vc) in enumerate(cols):
            x0 = cx + j * (bw + 12)
            d.rounded_rectangle([x0, cy, x0 + bw, cy + 76], 10, fill="white", outline=LINE, width=2)
            d.text((x0 + 12, cy + 8), lab, font=f(16), fill=SUB)
            d.text((x0 + 12, cy + 32), f"{val:,.0f}円", font=f(23, True), fill=vc)
        cy += 92

        lot = f"リスク {c.risk_pct:.2f}%" + (f" ≈{c.shares:,}株" if c.shares else "")
        d.text((cx, cy), f"{lot}  リスク幅{c.risk_w/c.entry*100:.1f}% / 時間ストップ{cfg.time_stop_days}営業日",
               font=f(19, True), fill=INK); cy += 28
        for ln in flag_lines:
            d.text((cx, cy), ln, font=f(17, True), fill=GOLD); cy += 23
        y += ch + 16

    if not res["picked"]:
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 60], 14, fill=BG_CARD, outline=LINE, width=2)
        text(MARGIN + 20, y + 16, "該当なし — ゼロ件はゼロ件。", 20, SUB, True)
        y += 76

    if res.get("watch"):
        BG_WATCH = (245, 247, 250)
        y = text(MARGIN, y, "2週間スイング・参考層", 20, INK, True)
        for c in res["watch"]:
            eng_label = "R" if c.engine == "R" else "M"
            d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 60], 12, fill=BG_WATCH, outline=LINE, width=2)
            d.text((MARGIN + 16, y + 10), f"{c.code} {c.name} [{eng_label}]", font=f(18, True), fill=INK)
            d.text((MARGIN + 16, y + 34), "監視のみ・直接エントリーは規律違反", font=f(15, True), fill=GOLD)
            y += 68

    y = hline(y + 2)
    y = text(MARGIN, y, f"2週間スイング総リスク: 新規計 {st.get('total_risk',0):.2f}% / "
             f"上限 {st.get('risk_cap',0):.1f}%   |   {pos_note}", 18, INK)
    for s in [f"出口: 固定利確(約{cfg.profit_target_r:.1f}R)/初期ストップ/{cfg.time_stop_days}営業日時間ストップ、"
             "いずれか先に到達で手仕舞い(シャンデリアは使わない)。",
             "エントリー時はpositions_swing2w.csvにentry_date(必須)を記録すること。"]:
        for ln in _wrap(d, s, f(16), W - 2 * MARGIN):
            y = text(MARGIN, y, ln, 16, SUB)

    return y
