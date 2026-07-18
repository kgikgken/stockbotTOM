"""topdown — PNGレポート(一新プロンプトの【出力後】仕様: 白背景・結論上部・番号カード・
IN/STOP/利確3ボックス・保有期間タグ色分け(短期スイング=黄系/スイング=青系)・寄り天/高ボラ警告・
次点セクション・今週のイベントカード)。PIL直描画、momentum系のユーティリティを再利用。"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# ---- 描画ユーティリティ(自立版: 旧momentum/report_png.pyから逐語移植) ----
INK = (28, 32, 38)
SUB = (95, 103, 115)
LINE = (225, 228, 233)
RED = (200, 38, 60)
GREEN = (63, 106, 82)
GOLD = (169, 126, 31)
BLUE = (23, 92, 211)
BG_CARD = (248, 249, 251)


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

W, MARGIN = 1080, 36
YELLOW_TAG = (191, 144, 0)
BLUE_TAG = (41, 98, 165)


def render_png(outpath: str, today: str, meta: dict, sentiment: dict, res: dict,
               position_alerts: list, pos_note: str, events: list, cfg) -> str:
    f = F(cfg)
    st = res["stats"]
    sr = res["sector_rank"]
    notable = [a for a in (position_alerts or []) if a.get("hit") or a.get("hit") is None]

    est_h = (700 + len(notable) * 100 + len(res["picked"]) * 480
             + len(res.get("watch", [])) * 110 + len(events) * 40 + 500)
    img = Image.new("RGB", (W, est_h), "white")
    d = ImageDraw.Draw(img)

    def text(x, y, s, size=22, color=INK, bold=False):
        d.text((x, y), s, font=f(size, bold), fill=color)
        return y + int(size * 1.5)

    def hline(y):
        d.line([(MARGIN, y), (W - MARGIN, y)], fill=LINE, width=2)
        return y + 14

    y = MARGIN
    y = text(MARGIN, y, "新スクリーニング(地合い×セクター×カタリスト痕跡)", 30, GREEN, True)
    y = text(MARGIN, y, today, 20, SUB)

    # --- 結論バナー(最上部・地合い) ---
    prov = "(暫定)" if sentiment.get("provisional") else ""
    col = GREEN if sentiment["score"] >= 4 else (GOLD if sentiment["score"] == 3 else RED)
    d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 96], 14,
                        fill=(240, 248, 242) if sentiment["score"] >= 4 else ((253, 249, 235) if sentiment["score"] == 3 else (255, 242, 240)),
                        outline=col, width=3)
    text(MARGIN + 20, y + 12, f"地合いスコア {sentiment['score']}/5{prov} — {sentiment['stance']} / 本命{st['picked']}件", 26, col, True)
    sub_line = " / ".join(sentiment.get("reasons", [])[:3])
    if sentiment.get("vi_proxy") is not None:
        sub_line += f" / VI代理{sentiment['vi_proxy']:.0f}"
    for ln in _wrap(d, sub_line, f(17), W - 2 * MARGIN - 40)[:2]:
        text(MARGIN + 20, y + 52, ln, 17, SUB)
        break
    y += 110

    # --- データ取得状況 ---
    y = text(MARGIN, y, f"データ {meta.get('data_ok','?')}/{meta.get('data_total','?')}"
             f"({meta.get('data_coverage',0)*100:.0f}%) {meta.get('source','')}", 18, SUB)
    y = text(MARGIN, y, "カタリスト中身・需給・イベント詳細は取得不可 — 発注前にiSPEED/TDnetで要確認", 18, RED, True)

    # --- 保有銘柄アラート ---
    for a in notable:
        tag_col = RED if a.get("hit") in ("stop", "time") else (GREEN if a.get("hit") == "target" else GOLD)
        tag = {"target": "利確到達", "stop": "ストップ到達", "time": "時間ストップ"}.get(a.get("hit"), "要確認")
        lines = _wrap(d, f"⚠{tag} {a['code']} {a['name']}: {a['note']}", f(17, True), W - 2 * MARGIN - 32)
        bh = 16 + len(lines) * 23 + 10
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + bh], 10, fill=(255, 246, 240), outline=tag_col, width=2)
        yy = y + 12
        for ln in lines:
            text(MARGIN + 14, yy, ln, 17, tag_col, True); yy += 23
        y += bh + 8

    # --- セクター見立て ---
    y = hline(y + 6)
    y = text(MARGIN, y, "セクター見立て(等ウェイト代理・直近5日)", 22, INK, True)
    if sr["top"]:
        y = text(MARGIN, y, "↑上位: " + " / ".join(f"{s}({r:+.1f}%)" for s, r in sr["top"]), 19, GREEN, True)
    if sr["bottom"]:
        y = text(MARGIN, y, "↓回避: " + " / ".join(f"{s}({r:+.1f}%)" for s, r in sr["bottom"]), 19, RED, True)
    y += 4

    # --- 本命カード ---
    y = hline(y)
    y = text(MARGIN, y, f"本日の候補(最大{cfg.max_candidates}) — 点灯 GAP{st['trigger_count']['GAP']}"
             f"/BREAK{st['trigger_count']['BREAK']}/PULL{st['trigger_count']['PULL']}", 22, INK, True)
    if res["picked"]:
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 52], 10, fill=(255, 240, 240), outline=RED, width=2)
        text(MARGIN + 16, y + 12, "[要確認] iSPEED/TDnetでカタリストの中身を確認するまで発注不可", 19, RED, True)
        y += 64
    else:
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 60], 14, fill=BG_CARD, outline=LINE, width=2)
        text(MARGIN + 20, y + 16, "該当なし — ゼロ件はゼロ件。無理に格下げ採用しない。", 20, SUB, True)
        y += 76

    card_textw = W - 2 * MARGIN - 44
    for i, c in enumerate(res["picked"], 1):
        trig_lines = _wrap(d, f"トリガー: {c.trigger_text}", f(17), card_textw)
        conf_lines = _wrap(d, f"確信度: {c.confidence} — {c.conf_reason}", f(17, True), card_textw)
        risk_lines = []
        for r in c.risks[:2]:
            risk_lines.extend(_wrap(d, "⚠ " + r, f(16), card_textw))
        flag_lines = []
        for fl in c.flags:
            flag_lines.extend(_wrap(d, fl, f(16, True), card_textw))
        ch = (16 + 52 + len(trig_lines) * 22 + 92 + 30
              + len(conf_lines) * 23 + len(risk_lines) * 21 + len(flag_lines) * 21 + 16)

        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + ch], 16, fill=BG_CARD, outline=LINE, width=2)
        cx, cy = MARGIN + 22, y + 16
        tag_col = YELLOW_TAG if c.tag == "短期スイング" else BLUE_TAG
        d.rounded_rectangle([cx, cy, cx + 168, cy + 42], 10, fill=tag_col)
        d.text((cx + 12, cy + 7), c.tag, font=f(19, True), fill="white")
        d.text((cx + 182, cy - 2), f"{i}. {c.code} {c.name}", font=f(26, True), fill=INK)
        cy += 52
        d.text((cx, cy), f"{c.sector}" + ("(順風)" if c.tailwind else "(逆風)" if c.headwind else ""),
               font=f(17), fill=GREEN if c.tailwind else (RED if c.headwind else SUB))
        cy += 24
        for ln in trig_lines:
            d.text((cx, cy), ln, font=f(17), fill=SUB); cy += 22

        cols = [("IN(前日終値基準)", c.entry, INK), ("STOP", c.stop, RED), (f"利確({cfg.profit_target_r:.1f}R)", c.target, GREEN)]
        bw = (W - 2 * MARGIN - 44 - 2 * 12) // 3
        for j, (lab, val, vc) in enumerate(cols):
            x0 = cx + j * (bw + 12)
            d.rounded_rectangle([x0, cy, x0 + bw, cy + 76], 10, fill="white", outline=LINE, width=2)
            d.text((x0 + 12, cy + 8), lab, font=f(15), fill=SUB)
            d.text((x0 + 12, cy + 32), f"{val:,.0f}円", font=f(23, True), fill=vc)
        cy += 92

        lot = f"リスク {c.risk_pct:.2f}%" + (f" ≈{c.shares:,}株" if c.shares else "")
        d.text((cx, cy), f"{lot} / 時間ストップ{c.time_stop}営業日", font=f(18, True), fill=INK); cy += 28
        for ln in conf_lines:
            d.text((cx, cy), ln, font=f(17, True), fill=INK); cy += 23
        for ln in risk_lines:
            d.text((cx, cy), ln, font=f(16), fill=GOLD); cy += 21
        for ln in flag_lines:
            d.text((cx, cy), ln, font=f(16, True), fill=GOLD); cy += 21
        y += ch + 14

    # --- 次点 ---
    if res.get("watch"):
        y = text(MARGIN, y + 4, "次点(監視)", 20, INK, True)
        for c in res["watch"]:
            lines = _wrap(d, f"{c.code} {c.name} [{c.trigger}] {c.trigger_text}", f(17, True), card_textw)
            warn = c.flags[0] if c.flags else ""
            wlines = _wrap(d, warn, f(15, True), card_textw) if warn else []
            bh = 14 + len(lines) * 22 + len(wlines) * 19 + 10
            d.rounded_rectangle([MARGIN, y, W - MARGIN, y + bh], 12, fill=(245, 247, 250), outline=LINE, width=2)
            yy = y + 10
            for ln in lines:
                d.text((MARGIN + 16, yy), ln, font=f(17, True), fill=INK); yy += 22
            for ln in wlines:
                d.text((MARGIN + 16, yy), ln, font=f(15, True), fill=GOLD); yy += 19
            y += bh + 8

    # --- 今週のイベントカード ---
    y = hline(y + 4)
    y = text(MARGIN, y, "今週の重要イベント(events.csv・手動管理)", 20, INK, True)
    if events:
        for e in events:
            y = text(MARGIN + 10, y, f"・{e['date']} {e['label']}", 18, INK)
    else:
        y = text(MARGIN + 10, y, "登録なし — 経済指標カレンダーは自動取得不可(手動確認)", 18, SUB)

    y = hline(y + 4)
    for s in ["共通リスク: 全候補ともカタリストの中身は未確認(価格痕跡のみ)。悪材料の可能性を常に残す。",
              f"総リスク: 新規計 {st['total_risk']:.2f}% | {pos_note}",
              "免責: AI候補提示で投資助言ではない。数値は単一ソースにつき要再確認。最終判断と結果責任はユーザーにある。"]:
        for ln in _wrap(d, s, f(16), W - 2 * MARGIN):
            y = text(MARGIN, y, ln, 16, SUB)

    img = img.crop((0, 0, W, min(est_h, y + MARGIN)))
    img.save(outpath)
    return outpath
