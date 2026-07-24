"""topdown — PNGレポート v2.0(モックアップ検証済みレイアウト).

3ボックスの構成を変更:
  ①INゾーン(上端〜下端の指値レンジ) ②STOP(構造・どこで入っても同じ) ③1Rの幅(浅い/深い)
固定利確を撤廃したため「利確価格」は出さない。ゾーン内のどこで約定するかで目標値が変わり、
固定の目標価格を出すと誤りになるため、代わりに1Rの幅(円と%)を出して各自で計算できる形にした。
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

INK = (28, 32, 38)
SUB = (95, 103, 115)
LINE = (225, 228, 233)
RED = (200, 38, 60)
GREEN = (63, 106, 82)
GOLD = (169, 126, 31)
BLUE = (23, 92, 211)
BG_CARD = (248, 249, 251)
BLUE_TAG = (41, 98, 165)
GOLD_TAG = (191, 144, 0)

W, MARGIN = 1080, 36


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


def render_png(outpath: str, today: str, meta: dict, sentiment: dict, res: dict,
               position_alerts: list, pending_summary: dict, pending_events: list,
               events: list, cfg) -> str:
    f = F(cfg)
    st = res["stats"]
    sr = res["sector_rank"]
    tc = st["trigger_count"]
    notable = [a for a in (position_alerts or []) if a.get("hit") or a.get("hit") is None]

    est_h = (760 + len(pending_events) * 60 + len(notable) * 90
             + len(res["picked"]) * 430 + len(res.get("watch", [])) * 100
             + len(events) * 40 + 460)
    img = Image.new("RGB", (W, est_h), "white")
    d = ImageDraw.Draw(img)

    def text(x, y, s, size=22, color=INK, bold=False):
        d.text((x, y), s, font=f(size, bold), fill=color)
        return y + int(size * 1.5)

    def hline(y):
        d.line([(MARGIN, y), (W - MARGIN, y)], fill=LINE, width=2)
        return y + 14

    y = MARGIN
    y = text(MARGIN, y, "新スクリーニング", 30, GREEN, True)
    y = text(MARGIN, y, today, 20, SUB)

    # --- 結論バナー ---
    prov = "(暫定)" if sentiment.get("provisional") else ""
    col = GREEN if sentiment["score"] >= 4 else (GOLD if sentiment["score"] == 3 else RED)
    bg = ((240, 248, 242) if sentiment["score"] >= 4
          else ((253, 249, 235) if sentiment["score"] == 3 else (255, 242, 240)))
    d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 108], 14, fill=bg, outline=col, width=3)
    text(MARGIN + 20, y + 12, f"地合い {sentiment['score']}/5{prov} — {sentiment['stance']}"
                              f" / 本日の候補 {st['picked']}件", 26, col, True)
    sub = " / ".join(sentiment.get("reasons", [])[:3])
    if sentiment.get("vi_proxy") is not None:
        sub += f" / VI代理{sentiment['vi_proxy']:.0f}"
    for ln in _wrap(d, sub, f(17), W - 2 * MARGIN - 40)[:1]:
        text(MARGIN + 20, y + 50, ln, 17, SUB)
    text(MARGIN + 20, y + 76, f"保有中 {len(notable)}件 / ゾーン待ち {pending_summary.get('pending', 0)}件",
         17, BLUE, True)
    y += 122

    y = text(MARGIN, y, f"データ {meta.get('data_ok','?')}/{meta.get('data_total','?')}"
             f"({meta.get('data_coverage',0)*100:.0f}%) {meta.get('source','')}", 18, SUB)
    y = text(MARGIN, y, "カタリスト中身・需給は取得不可 — 発注前にiSPEED/TDnetで要確認", 18, RED, True)

    # --- ゾーン待ちの動き ---
    if pending_events:
        y = hline(y + 6)
        y = text(MARGIN, y, "ゾーン待ち候補の動き", 20, INK, True)
        for e in pending_events:
            mark, mc = {"reached": ("到達", GREEN), "expired": ("失効", SUB),
                        "broken": ("下端割れ", RED)}.get(e["event"], ("", SUB))
            lines = _wrap(d, f"[{mark}] {e['code']} {e['name']}: {e['note']}",
                          f(17, True), W - 2 * MARGIN - 32)
            bh = 12 + len(lines) * 22 + 8
            d.rounded_rectangle([MARGIN, y, W - MARGIN, y + bh], 10,
                                fill=(250, 251, 252), outline=LINE, width=2)
            yy = y + 8
            for ln in lines:
                text(MARGIN + 14, yy, ln, 17, mc, True); yy += 22
            y += bh + 6

    # --- 保有銘柄 ---
    if notable:
        y = hline(y + 6)
        y = text(MARGIN, y, "保有中の銘柄", 20, INK, True)
        for a in notable:
            tag, tc2 = {"stop": ("ストップ割れ", RED), "partial": ("+1R到達", GREEN),
                        "time": ("時間ストップ", GOLD)}.get(a.get("hit"), ("要確認", GOLD))
            prog = ""
            if a.get("days_held") is not None and a.get("time_stop"):
                prog = f"{a['days_held']}/{a['time_stop']}営業日"
            lines = _wrap(d, a["note"], f(16), W - 2 * MARGIN - 200)
            bh = 14 + 24 + len(lines) * 21 + 10
            d.rounded_rectangle([MARGIN, y, W - MARGIN, y + bh], 12,
                                fill=(250, 251, 252), outline=LINE, width=2)
            d.text((MARGIN + 16, y + 10), f"[{tag}] {a['code']} {a['name']}",
                   font=f(18, True), fill=tc2)
            if prog:
                d.text((W - MARGIN - 150, y + 12), prog, font=f(17, True), fill=SUB)
            yy = y + 36
            for ln in lines:
                d.text((MARGIN + 16, yy), ln, font=f(16), fill=SUB); yy += 21
            y += bh + 8

    # --- セクター ---
    y = hline(y + 6)
    y = text(MARGIN, y, "セクター(等ウェイト代理・直近5日)", 22, INK, True)
    if sr["top"]:
        y = text(MARGIN, y, "↑上位: " + " / ".join(f"{s}({r:+.1f}%)" for s, r in sr["top"]), 19, GREEN, True)
    if sr["bottom"]:
        y = text(MARGIN, y, "↓回避: " + " / ".join(f"{s}({r:+.1f}%)" for s, r in sr["bottom"]), 19, RED, True)
    y += 4

    # --- 候補 ---
    y = hline(y)
    y = text(MARGIN, y, f"本日の候補 — 点灯 材料反応{tc.get('材料反応',0)}"
                        f"・押し目{tc.get('押し目',0)}・高値ブレイク{tc.get('高値ブレイク',0)}",
             22, INK, True)
    if st.get("concentration"):
        y = text(MARGIN, y, "⚠" + st["concentration"], 18, GOLD, True)
    if res["picked"]:
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 52], 10, fill=(255, 240, 240),
                            outline=RED, width=2)
        text(MARGIN + 16, y + 13, "[要確認] iSPEED/TDnetでカタリストを確認するまで発注不可", 19, RED, True)
        y += 64
    else:
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 60], 14, fill=BG_CARD, outline=LINE, width=2)
        text(MARGIN + 20, y + 16, "該当なし — ゼロ件はゼロ件。無理に格下げ採用しない。", 20, SUB, True)
        y += 76

    CW = W - 2 * MARGIN - 44
    for i, c in enumerate(res["picked"], 1):
        trig_lines = _wrap(d, c.trigger_text, f(17), CW)
        conf_lines = _wrap(d, f"期待度 {c.score:.0f}/10 — {c.score_reason}", f(17, True), CW)
        risk_lines = []
        for r in c.risks[:2]:
            risk_lines.extend(_wrap(d, "⚠ " + r, f(16), CW))
        flag_lines = []
        for fl in c.flags:
            flag_lines.extend(_wrap(d, fl, f(16, True), CW))
        ch = (16 + 52 + 26 + len(trig_lines) * 24 + 96 + 30 + 26
              + len(conf_lines) * 23 + len(risk_lines) * 21 + len(flag_lines) * 21 + 16)

        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + ch], 16, fill=BG_CARD, outline=LINE, width=2)
        cx, cy = MARGIN + 22, y + 16
        tagcol = BLUE_TAG if c.tag == "1ヶ月" else GOLD_TAG
        d.rounded_rectangle([cx, cy, cx + 150, cy + 42], 10, fill=tagcol)
        d.text((cx + 14, cy + 7), c.trigger, font=f(19, True), fill="white")
        d.rounded_rectangle([cx + 160, cy, cx + 272, cy + 42], 10, fill=(238, 240, 244))
        d.text((cx + 172, cy + 8), "めやす" + c.tag, font=f(17, True), fill=SUB)
        d.text((cx + 286, cy - 2), f"{i}. {c.code} {c.name}", font=f(26, True), fill=INK)
        cy += 52
        d.text((cx, cy), f"{c.sector}" + ("(順風)" if c.tailwind else "(逆風)" if c.headwind else ""),
               font=f(17), fill=GREEN if c.tailwind else (RED if c.headwind else SUB))
        cy += 26
        for ln in trig_lines:
            d.text((cx, cy), ln, font=f(17), fill=SUB); cy += 24

        bw = (CW - 24) // 3
        # ①INゾーン
        d.rounded_rectangle([cx, cy, cx + bw, cy + 80], 10, fill="white", outline=GREEN, width=2)
        d.text((cx + 12, cy + 8), "INゾーン(指値)", font=f(15), fill=SUB)
        d.text((cx + 12, cy + 28), f"{c.zone_hi:,.0f} 円", font=f(20, True), fill=INK)
        d.text((cx + 12, cy + 52), f"〜 {c.zone_lo:,.0f} 円", font=f(20, True), fill=INK)
        # ②STOP
        x1 = cx + bw + 12
        d.rounded_rectangle([x1, cy, x1 + bw, cy + 80], 10, fill="white", outline=RED, width=2)
        d.text((x1 + 12, cy + 8), "STOP(構造)", font=f(15), fill=SUB)
        d.text((x1 + 12, cy + 32), f"{c.stop:,.0f} 円", font=f(23, True), fill=RED)
        d.text((x1 + 12, cy + 60), "どこで入っても同じ", font=f(14), fill=SUB)
        # ③1Rの幅
        x2 = cx + 2 * (bw + 12)
        d.rounded_rectangle([x2, cy, x2 + bw, cy + 80], 10, fill="white", outline=LINE, width=2)
        d.text((x2 + 12, cy + 8), "1Rの幅(リスク幅)", font=f(15), fill=SUB)
        d.text((x2 + 12, cy + 30), f"浅く {c.risk_shallow:,.0f}円 ({c.risk_pct_shallow:.1f}%)",
               font=f(16, True), fill=INK)
        d.text((x2 + 12, cy + 54), f"深く {c.risk_deep:,.0f}円 ({c.risk_pct_deep:.1f}%)",
               font=f(16, True), fill=GREEN)
        cy += 96

        d.text((cx, cy), f"失効 {c.expire_date}({cfg.zone_expire_days}営業日) / "
                         f"時間ストップ {c.time_stop}営業日 / 1単元{c.unit_cost/1e4:,.0f}万円",
               font=f(18, True), fill=INK)
        cy += 30
        for ln in conf_lines:
            d.text((cx, cy), ln, font=f(17, True), fill=INK); cy += 23
        d.text((cx, cy), "出口: +1Rで半分利確(2単元以上) → 残玉は構造まで引上げ+トレーリング",
               font=f(16), fill=BLUE)
        cy += 26
        for ln in risk_lines:
            d.text((cx, cy), ln, font=f(16), fill=GOLD); cy += 21
        for ln in flag_lines:
            d.text((cx, cy), ln, font=f(16, True), fill=GOLD); cy += 21
        y += ch + 14

    # --- 次点 ---
    if res.get("watch"):
        y = text(MARGIN, y + 4, "次点(監視)", 20, INK, True)
        for c in res["watch"]:
            lines = _wrap(d, f"{c.code} {c.name} [{c.trigger}] {c.trigger_text}", f(17, True), CW)
            warn = c.flags[0] if c.flags else ""
            wlines = _wrap(d, warn, f(15, True), CW) if warn else []
            bh = 14 + len(lines) * 22 + len(wlines) * 19 + 10
            d.rounded_rectangle([MARGIN, y, W - MARGIN, y + bh], 12,
                                fill=(245, 247, 250), outline=LINE, width=2)
            yy = y + 10
            for ln in lines:
                d.text((MARGIN + 16, yy), ln, font=f(17, True), fill=INK); yy += 22
            for ln in wlines:
                d.text((MARGIN + 16, yy), ln, font=f(15, True), fill=GOLD); yy += 19
            y += bh + 8

    # --- イベント ---
    y = hline(y + 4)
    y = text(MARGIN, y, "今週の重要イベント(events.csv・手動管理)", 20, INK, True)
    if events:
        for e in events:
            y = text(MARGIN + 10, y, f"・{e['date']} {e['label']}", 18, INK)
    else:
        y = text(MARGIN + 10, y, "登録なし — 経済指標カレンダーは自動取得不可(手動確認)", 18, SUB)

    y = hline(y + 4)
    for s in ["共通リスク: カタリストの中身は未確認(価格痕跡のみ)。悪材料の可能性を常に残す。",
              "免責: AI候補提示で投資助言ではない。最終判断と結果責任はユーザーにある。"]:
        for ln in _wrap(d, s, f(16), W - 2 * MARGIN):
            y = text(MARGIN, y, ln, 16, SUB)

    img = img.crop((0, 0, W, min(est_h, y + MARGIN)))
    img.save(outpath)
    return outpath
