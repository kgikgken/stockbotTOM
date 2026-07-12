"""v5.0インフォグラフィック(PNG・白背景・Noto Sans CJK JP) — PILで直接描画.

★変更不可要件: 最上部=結論+ロット指示+資金循環マップ(流入→流出矢印・レジームタグ)。
本命=番号カード。参考監視層=薄色ブロック。候補ゼロの日も資金循環マップと棄却カードを描く。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W = 1080
MARGIN = 36
INK = (28, 32, 38)
SUB = (95, 103, 115)
LINE = (225, 228, 233)
BLUE = (23, 92, 211)
RED = (200, 38, 60)       # ロング(日本式: 上昇=赤)
GREEN = (0, 130, 90)      # ショート/流出
AMBER = (176, 108, 0)
BG_CARD = (248, 249, 251)
BG_WATCH = (245, 247, 250)
BG_FLOW_IN = (255, 240, 235)
BG_FLOW_OUT = (232, 245, 240)


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


def _tw(d, text, font):
    b = d.textbbox((0, 0), text, font=font)
    return b[2] - b[0]


def _wrap(d, text, font, maxw):
    out, cur = [], ""
    for ch in text:
        if _tw(d, cur + ch, font) > maxw:
            out.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def render_png(outpath: str, today: str, meta: dict, macro: dict, flow: dict,
               res: dict, pos_note: str, events: list[str], cfg,
               positions: list[dict] | None = None) -> str:
    f = F(cfg)
    positions = positions or []
    est_h = (760 + 150 + len(res["picked"]) * 480
             + len(res["watch"]) * 130 + len(events) * 34 + 460
             + len(positions) * 150)
    img = Image.new("RGB", (W, est_h), "white")
    d = ImageDraw.Draw(img)
    y = MARGIN

    def text(x, yy, s, size=22, color=INK, bold=False):
        d.text((x, yy), s, font=f(size, bold), fill=color)
        return yy + int(size * 1.5)

    def hline(yy):
        d.line([(MARGIN, yy), (W - MARGIN, yy)], fill=LINE, width=2)
        return yy + 14

    # ---------- ヘッダー: 結論とロット指示(★最上部要件) ----------
    y = text(MARGIN, y, "歪み×資金循環スクリーニング v5.0-bot(パス1)", 29, BLUE, True)
    vi_str = "" if macro["vi"] is None else "={:.1f}".format(macro["vi"])
    y = text(MARGIN, y, f"{today}  |  地合い {macro['score']}/5  |  {macro['vi_label']}{vi_str}",
             21, SUB)
    y += 6
    n = len(res["picked"])
    concl = f"本命 {n}件(仮点灯・要iSPEED確認)" if n else "本命 0件 — 本日は見送り"
    d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 92], 14,
                        fill=(255, 246, 240) if n else (240, 247, 255),
                        outline=AMBER if n else BLUE, width=3)
    text(MARGIN + 20, y + 8, concl, 28, INK, True)
    text(MARGIN + 20, y + 50, f"ロット指示: {macro['lot_text']}  /  総リスク上限 {macro['risk_cap']:.1f}%",
         22, RED if macro["half_lot"] else SUB, macro["half_lot"])
    y += 110

    # ---------- 保有ポジション評価(日次再評価・新規追加) ----------
    if positions:
        y = text(MARGIN, y, "保有ポジション評価(日次再評価)", 24, INK, True)
        for p in positions:
            if p.get("error"):
                d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 60], 12, fill=BG_WATCH, outline=LINE, width=2)
                d.text((MARGIN + 14, y + 18), f"{p['code']} {p['name']}: ⚠{p['error']}", font=f(19), fill=AMBER)
                y += 76
                continue
            box_h = 138 if p.get("candidate") or p.get("liquidity_warn") else 108
            d.rounded_rectangle([MARGIN, y, W - MARGIN, y + box_h], 12, fill=BG_CARD, outline=LINE, width=2)
            dcol = RED if p["direction"] == "ロング" else GREEN
            d.text((MARGIN + 16, y + 10), f"{p['code']} {p['name']}", font=f(21, True), fill=INK)
            d.text((MARGIN + 16, y + 40), f"[{p['direction']}] 現在値 {p['close']:,.0f}円", font=f(18), fill=dcol)
            if "entry_price" in p:
                d.text((MARGIN + 290, y + 40),
                       f"建値{p['entry_price']:,.0f}円 含み損益{p['pnl_pct']:+.1f}%",
                       font=f(18), fill=(RED if p["pnl_pct"] >= 0 else GREEN))
            d.text((MARGIN + 16, y + 68),
                   f"本日の構造的ストップ参考: {p['today_stop']:,.0f}円  "
                   f"(25日線{p['today_sma25']:,.0f} / RSI{p['rsi14']:.0f} / 乖離{p['rel_dev_pct']:+.1f}%)",
                   font=f(17), fill=SUB)
            yy = y + 94
            c = p.get("candidate")
            if c:
                d.text((MARGIN + 16, yy),
                       f"{p['status_note']} → 新規評価ならIN{c.entry:,.0f}/STOP{c.stop:,.0f}"
                       f"/TP1{c.tp1:,.0f}/2R{c.ref2r:,.0f}円",
                       font=f(17, True), fill=BLUE)
            else:
                d.text((MARGIN + 16, yy), p["status_note"], font=f(17), fill=SUB)
            if p.get("liquidity_warn"):
                d.text((MARGIN + 16, yy + 24), "⚠" + p["liquidity_warn"], font=f(17, True), fill=AMBER)
            y += box_h + 14
        y = text(MARGIN, y, "(価格・%ベースの参考表示。R倍数は当初stop_price未登録のため非表示)", 16, SUB)
        y += 6

    # ---------- ★資金循環マップ(流入→流出・レジームタグ・毎回必須) ----------
    d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 8], 0, fill=BLUE)
    y += 18
    y = text(MARGIN, y, f"資金循環マップ  レジーム: {flow.get('regime','不明')}", 24, INK, True)
    if flow.get("ok"):
        colw = (W - 2 * MARGIN - 60) // 2
        x_in, x_out = MARGIN, MARGIN + colw + 60
        d.rounded_rectangle([x_in, y, x_in + colw, y + 150], 12, fill=BG_FLOW_IN, outline=RED, width=2)
        d.text((x_in + 14, y + 10), "流入 上位3", font=f(19, True), fill=RED)
        yy = y + 42
        for r in flow["inflow"]:
            stg = flow["sector_stage"].get(r["sector"], "不明")
            for ln in _wrap(d, f"{r['sector']} {r['ret5']:+.1f}%(5d) [{stg}]", f(18), colw - 24):
                d.text((x_in + 14, yy), ln, font=f(18), fill=INK); yy += 24
        d.text((x_in + colw - 44, y + 60), "→", font=f(34, True), fill=RED)

        d.rounded_rectangle([x_out, y, x_out + colw, y + 150], 12, fill=BG_FLOW_OUT, outline=GREEN, width=2)
        d.text((x_out + 14, y + 10), "流出 上位3", font=f(19, True), fill=GREEN)
        yy = y + 42
        for r in flow["outflow"]:
            stg = flow["sector_stage"].get(r["sector"], "不明")
            for ln in _wrap(d, f"{r['sector']} {r['ret5']:+.1f}%(5d) [{stg}]", f(18), colw - 24):
                d.text((x_out + 14, yy), ln, font=f(18), fill=INK); yy += 24
        y += 162
        for ln in _wrap(d, flow.get("note", ""), f(17), W - 2 * MARGIN):
            y = text(MARGIN, y, ln, 17, SUB)
    else:
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 60], 12, fill=BG_CARD, outline=LINE, width=2)
        text(MARGIN + 14, y + 16, f"作成不可: {flow.get('note','データ不足')}", 19, SUB)
        y += 76
    y = hline(y + 4)

    # データ状況
    cov = meta.get("data_coverage", 0.0)
    y = text(MARGIN, y, f"データ {meta.get('data_ok',0)}/{meta.get('data_total',0)}({cov*100:.0f}%) "
             f"{meta.get('source','')}  取得 {meta.get('fetched_at','')}", 18, SUB)
    y = text(MARGIN, y, "単一ソース(yfinance)を許容 — ただし本命は全件仮点灯。確定はiSPEED照合+チャット側ゲート0/3後。",
             18, AMBER)
    if meta.get("data_warn"):
        y = text(MARGIN, y, f"⚠ データ被覆率不足(<{cfg.data_coverage_min*100:.0f}%) — 新規見送り推奨", 20, RED, True)
    st = res["stats"]
    eng = st["by_engine"]
    y = text(MARGIN, y, f"検討 {st['considered']} → 棄却 {st['rejected']} → 本命 {st['picked']} "
             f"(参考層{st['watch']})   エンジン: A{eng.get('A',0)} / S{eng.get('S',0)} / B疑い{eng.get('B疑い',0)}",
             18, SUB)
    y = hline(y + 4)

    # ---------- 本命カード ----------
    dir_names = {"ロング": ("ロング", RED), "ショート": ("ショート", GREEN)}
    for i, c in enumerate(res["picked"], 1):
        ch = 470 + 26 * max(0, len(c.flags) - 1)
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + ch], 16, fill=BG_CARD, outline=LINE, width=2)
        cx, cy = MARGIN + 22, y + 16
        dlabel, dcol = dir_names[c.direction]
        d.rounded_rectangle([cx, cy, cx + 118, cy + 42], 10, fill=dcol)
        d.text((cx + 12, cy + 6), dlabel, font=f(22, True), fill="white")
        d.rounded_rectangle([cx + 130, cy, cx + 234, cy + 42], 10, outline=dcol, width=3)
        d.text((cx + 144, cy + 6), f"engine {c.engine}", font=f(19, True), fill=dcol)
        tagcol = RED if c.ftag == "順流" else (SUB if c.ftag == "中立" else AMBER)
        d.rounded_rectangle([cx + 246, cy, cx + 340, cy + 42], 10, outline=tagcol, width=3)
        d.text((cx + 258, cy + 6), c.ftag, font=f(19, True), fill=tagcol)
        d.text((cx + 352, cy - 2), f"{i}. {c.code} {c.name}", font=f(26, True), fill=INK)
        cy += 50
        d.text((cx, cy), f"{c.sector} / {c.market} / 確信度 {c.conf} / 根拠 {c.basis} / セクター段階 {c.sec_stage}",
               font=f(18), fill=SUB)
        cy += 32
        for ln in _wrap(d, "①歪み: " + c.nonfund, f(19), W - 2 * MARGIN - 44):
            d.text((cx, cy), ln, font=f(19), fill=INK); cy += 26
        for ln in _wrap(d, c.trigger_text, f(18), W - 2 * MARGIN - 44):
            d.text((cx, cy), ln, font=f(18), fill=SUB); cy += 24
        cy += 6

        cols = [("IN", c.entry, INK), ("STOP", c.stop, RED),
                ("第1利確+1R", c.tp1, BLUE), ("2R参照", c.ref2r, SUB)]
        bw = (W - 2 * MARGIN - 44 - 3 * 12) // 4
        for j, (lab, val, col) in enumerate(cols):
            x0 = cx + j * (bw + 12)
            d.rounded_rectangle([x0, cy, x0 + bw, cy + 76], 10, fill="white", outline=LINE, width=2)
            d.text((x0 + 12, cy + 8), lab, font=f(17), fill=SUB)
            d.text((x0 + 12, cy + 32), f"{val:,.0f}円", font=f(25, True), fill=col)
        cy += 92

        lot = f"リスク {c.risk_pct:.2f}%" + (f" ≈{c.shares:,}株" if c.shares else "")
        d.text((cx, cy), f"{lot}  計画R=2.0  概算ネットR≈{c.net2r:.2f}  到達≈{c.reach_days:.0f}ATR日",
               font=f(19, True), fill=INK); cy += 28
        d.text((cx, cy), f"②資金フロー: {c.ftag} / レジーム{flow.get('regime','不明')}の影響を加味", font=f(18), fill=SUB); cy += 26
        d.text((cx, cy), f"④保有≤{c.hold_days}営業日 / 失効{c.expiry_days}日 / 残玉=直近{c.trail_days}日トレールor時間ストップ",
               font=f(18), fill=SUB); cy += 26
        d.text((cx, cy), "第1利確で半分→残り建値±数ティック / IFD-OCO同時発注 / 寄りギャップ±リスク幅50%で計画失効",
               font=f(18), fill=SUB); cy += 28
        for fl in c.flags:
            d.text((cx, cy), "⚠ " + fl, font=f(18, True), fill=AMBER); cy += 25
        d.text((cx, cy), "iSPEED確認 → " + " / ".join(c.checks[:2]), font=f(18), fill=BLUE); cy += 25
        d.text((cx, cy), "ゲート0(市場が正しい可能性)・ゲート3(反証+プレモータム)は未実施 → チャットで実施",
               font=f(18), fill=RED)
        y += ch + 16

    if not res["picked"]:
        # 候補ゼロの日も棄却カードを必ず描く
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 130], 14, fill=BG_CARD, outline=LINE, width=2)
        text(MARGIN + 20, y + 16, "該当なし — ゼロ件はゼロ件。無理に格下げ採用しない。", 23, SUB, True)
        by_stage = st.get("by_stage", {})
        stg_txt = " / ".join(f"{k}:{v}" for k, v in by_stage.items()) if by_stage else "(棄却理由内訳なし)"
        text(MARGIN + 20, y + 56, f"棄却内訳: {stg_txt}", 19, SUB)
        text(MARGIN + 20, y + 90, f"検討{st['considered']}件 → 全て不採用", 19, SUB)
        y += 146

    # ---------- 参考監視層(薄色ブロック) ----------
    if res["watch"]:
        y = hline(y + 2)
        y = text(MARGIN, y, "参考監視層(本命関連・構造的非実行性あり)", 22, INK, True)
        for c in res["watch"]:
            d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 106], 12, fill=BG_WATCH, outline=LINE, width=2)
            d.text((MARGIN + 16, y + 10), f"{c.code} {c.name}  [{c.direction}/engine {c.engine}]",
                   font=f(20, True), fill=INK)
            d.text((MARGIN + 16, y + 42), f"{c.sector} / {c.ftag} / 理由: {c.watch_reason}",
                   font=f(17), fill=SUB)
            d.text((MARGIN + 16, y + 70), "監視のみ・直接エントリーは規律違反(昇格には後日エンジン点灯+4ゲート通過が必要)",
                   font=f(17, True), fill=AMBER)
            y += 118

    # ---------- 総リスク / イベント / 免責 ----------
    y = hline(y + 2)
    y = text(MARGIN, y, f"総オープンリスク: 新規計 {st['total_risk']:.2f}% / 上限 {st['risk_cap']:.1f}%   |   {pos_note}",
             19, INK)
    if events:
        y = text(MARGIN, y, "今週の重要イベント: " + " / ".join(events), 19, SUB)

    y = hline(y + 4)
    for s, col, bold in [
        ("行動注意: 候補が出た日に必ず取引する必要はない。ゼロ件の日に格下げ採用しない。", AMBER, True),
        ("IN/STOP/利確は前日確定値の目安 — 寄り後に実勢価格で再計算してから発注。", SUB, False),
        ("免責: AIの候補提示で投資助言ではない。ハルシネーション可能性/市場が正しい可能性が常にある。", SUB, False),
        ("数値は必ず自分で再確認。利益には申告分離課税 約20%。最終判断と結果責任はユーザーにある。", SUB, False),
    ]:
        for ln in _wrap(d, s, f(18, bold), W - 2 * MARGIN):
            y = text(MARGIN, y, ln, 18, col, bold)

    img = img.crop((0, 0, W, min(est_h, y + MARGIN)))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    img.save(outpath, "PNG")
    return outpath
