"""v4.1インフォグラフィック(PNG・白背景・Noto Sans CJK JP) — PILで直接描画."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W = 1080
MARGIN = 36
INK = (28, 32, 38)
SUB = (95, 103, 115)
LINE = (225, 228, 233)
BLUE = (23, 92, 211)
RED = (200, 38, 60)      # ロング(日本式: 上昇=赤)
GREEN = (0, 130, 90)     # ショート
AMBER = (176, 108, 0)
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
                self._cache[key] = ImageFont.truetype(path, size, index=0)  # 0 = CJK JP
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
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def render_png(outpath: str, today: str, meta: dict, macro: dict, res: dict,
               pos_note: str, events: list[str], cfg) -> str:
    f = F(cfg)
    est_h = 640 + len(res["picked"]) * 460 + len(res["runners"]) * 40 + len(events) * 34 + 420
    img = Image.new("RGB", (W, est_h), "white")
    d = ImageDraw.Draw(img)
    y = MARGIN

    def text(x, yy, s, size=22, color=INK, bold=False):
        d.text((x, yy), s, font=f(size, bold), fill=color)
        return yy + int(size * 1.5)

    def hline(yy):
        d.line([(MARGIN, yy), (W - MARGIN, yy)], fill=LINE, width=2)
        return yy + 14

    # ---------- ヘッダー: 結論とロット指示を最上部(★変更不可要件) ----------
    y = text(MARGIN, y, f"歪みスクリーニング v4.1-bot(パス1)", 30, BLUE, True)
    vi_str = "" if macro["vi"] is None else "={:.1f}".format(macro["vi"])
    y = text(MARGIN, y, f"{today}  |  地合い {macro['score']}/5  |  "
             f"{macro['vi_label']}{vi_str}", 22, SUB)
    y += 6
    n = len(res["picked"])
    concl = f"仮点灯 {n}件(要iSPEED確認) / 確定候補 0件" if n else "仮点灯 0件 — 本日は見送り"
    d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 96], 14,
                        fill=(255, 246, 240) if n else (240, 247, 255),
                        outline=AMBER if n else BLUE, width=3)
    text(MARGIN + 20, y + 10, concl, 30, INK, True)
    text(MARGIN + 20, y + 54, f"ロット指示: {macro['lot_text']}  /  総リスク上限 {macro['risk_cap']:.1f}%",
         24, RED if macro["half_lot"] else SUB, macro["half_lot"])
    y += 116

    # データ状況
    cov = meta.get("data_coverage", 0.0)
    y = text(MARGIN, y, f"データ {meta.get('data_ok',0)}/{meta.get('data_total',0)}({cov*100:.0f}%) "
             f"{meta.get('source','')}  取得 {meta.get('fetched_at','')}", 19, SUB)
    y = text(MARGIN, y, "全指標=単一ソース算出につき未確認。確定はiSPEED照合(独立2ソース化)+チャット側ゲート0/3後。",
             19, AMBER)
    if meta.get("data_warn"):
        y = text(MARGIN, y, f"⚠ データ被覆率不足(<{cfg.data_coverage_min*100:.0f}%) — 新規見送り推奨", 21, RED, True)
    st = res["stats"]
    y = text(MARGIN, y, f"検討 {st['considered']} → 棄却 {st['rejected']} → 仮点灯 {st['picked']}"
             + (f"   内訳 " + " / ".join(f"{k}:{v}" for k, v in st["by_stage"].items()) if st["by_stage"] else ""),
             19, SUB)
    y = hline(y + 4)

    # ---------- 銘柄カード ----------
    for i, c in enumerate(res["picked"], 1):
        ch = 430 + 26 * max(0, len(c.flags) - 1)
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + ch], 16, fill=BG_CARD, outline=LINE, width=2)
        cx = MARGIN + 22
        cy = y + 16
        dir_col = RED if c.direction == "ロング" else GREEN
        d.rounded_rectangle([cx, cy, cx + 132, cy + 44], 10, fill=dir_col)
        d.text((cx + 14, cy + 6), c.direction, font=f(24, True), fill="white")
        d.rounded_rectangle([cx + 146, cy, cx + 262, cy + 44], 10, outline=dir_col, width=3)
        d.text((cx + 162, cy + 6), f"型 {c.mtype}", font=f(22, True), fill=dir_col)
        d.text((cx + 280, cy + 2), f"{i}. {c.code} {c.name}", font=f(28, True), fill=INK)
        d.text((cx + 280, cy + 40), f"{c.sector} / {c.market} / 確信度 {c.conf} / 根拠 {c.basis}",
               font=f(19), fill=SUB)
        cy += 84
        for ln in _wrap(d, "トリガー: " + c.trigger_text, f(20), W - 2 * MARGIN - 44):
            d.text((cx, cy), ln, font=f(20), fill=INK)
            cy += 28

        # 価格ブロック(円数字)
        cy += 8
        cols = [("IN", c.entry, INK), ("STOP", c.stop, RED),
                ("第1利確+1R", c.tp1, BLUE), ("2R参照", c.ref2r, SUB)]
        bw = (W - 2 * MARGIN - 44 - 3 * 12) // 4
        for j, (lab, val, col) in enumerate(cols):
            x0 = cx + j * (bw + 12)
            d.rounded_rectangle([x0, cy, x0 + bw, cy + 78], 10, fill="white", outline=LINE, width=2)
            d.text((x0 + 12, cy + 8), lab, font=f(18), fill=SUB)
            d.text((x0 + 12, cy + 34), f"{val:,.0f}円", font=f(26, True), fill=col)
        cy += 94

        lot = f"リスク {c.risk_pct:.2f}%" + (f" ≈{c.shares:,}株" if c.shares else "")
        d.text((cx, cy), f"{lot}   計画R=2.0(ブレンド参考 +0.5R/+1.5R)   到達≈{c.reach_days:.0f}ATR日",
               font=f(20, True), fill=INK)
        cy += 30
        d.text((cx, cy), f"保有≤{c.hold_days}営業日(時間ストップ) / 未エントリー失効{c.expiry_days}日 / "
               f"残玉=直近{c.trail_days}日トレール or 時間ストップ(固定2R強制手仕舞い無し)",
               font=f(19), fill=SUB)
        cy += 28
        d.text((cx, cy), "第1利確で半分→残り建値±数ティックへ / 逆指値は発注と同時(IFD-OCO) / 寄りギャップ±リスク幅50%で計画失効",
               font=f(19), fill=SUB)
        cy += 30
        for fl in c.flags:
            d.text((cx, cy), "⚠ " + fl, font=f(19, True), fill=AMBER)
            cy += 26
        d.text((cx, cy), "iSPEED確認 → " + " / ".join(c.checks[:3]), font=f(19), fill=BLUE)
        cy += 26
        d.text((cx, cy), "ゲート0(市場が正しい可能性)・ゲート3(反証3+プレモータム)は未実施 → チャットで実施",
               font=f(19), fill=RED)
        y += ch + 16

    if not res["picked"]:
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 90], 14, fill=BG_CARD, outline=LINE, width=2)
        text(MARGIN + 20, y + 26, "該当なし — ゼロ件はゼロ件。無理に格下げ採用しない。", 24, SUB, True)
        y += 106

    # ---------- 次点 / 総リスク / イベント ----------
    y = hline(y + 2)
    y = text(MARGIN, y, f"総オープンリスク: 新規計 {st['total_risk']:.2f}% / 上限 {st['risk_cap']:.1f}%   |   {pos_note}",
             20, INK)
    if res["runners"]:
        y = text(MARGIN, y, "次点(確信度低・リスク超過分): " + " / ".join(
            f"{c.code} {c.name}({c.direction[0]}·{c.conf})" for c in res["runners"]), 20, SUB)
    if events:
        y = text(MARGIN, y, "今週の重要イベント: " + " / ".join(events), 20, SUB)

    y = hline(y + 4)
    for s, col, bold in [
        ("行動注意: 候補が出た日に必ず取引する必要はない。ゼロ件の日に格下げ採用しない。", AMBER, True),
        ("IN/STOP/利確は前日確定値の目安 — 寄り後に実勢価格で再計算してから発注。", SUB, False),
        ("免責: AIの候補提示で投資助言ではない。ハルシネーション可能性/市場が正しい可能性が常にある。", SUB, False),
        ("数値は必ず自分で再確認。利益には申告分離課税 約20%。最終判断と結果責任はユーザーにある。", SUB, False),
    ]:
        for ln in _wrap(d, s, f(19, bold), W - 2 * MARGIN):
            y = text(MARGIN, y, ln, 19, col, bold)

    img = img.crop((0, 0, W, min(est_h, y + MARGIN)))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    img.save(outpath, "PNG")
    return outpath
