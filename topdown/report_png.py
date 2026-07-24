"""topdown — PNGレポート v3(見た目の全面刷新・2026-07-24).

温かみのあるクリーム/木目調の配色、角丸カード、番号丸、肉球モチーフに変更した。
情報量は v2 から一切減らしていない(安全側の警告・出口・リスクも全て残す)。

マスコット画像について:
  写真調のイラストはPILの図形描画では作れないため、画像ファイルがあれば貼り込む方式にした。
  リポジトリ直下に assets/mascot.png (背景透過推奨) を置くとヘッダー右に合成される。
  無い場合は肉球を描いて代替するので、ファイルが無くてもレイアウトは崩れない。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .data import coverage_note
from .market import closing_note

# --- 配色(温かいクリーム/ブラウン系) ---
BG        = (243, 234, 216)   # 全体の背景
CARD      = (252, 247, 238)   # カード
CARD_ALT  = (248, 241, 227)   # 帯・サブカード
BORDER    = (216, 199, 168)   # 枠線
HEADER    = (74, 55, 40)      # ヘッダーバー
CREAM     = (250, 245, 235)   # ヘッダー上の文字
INK       = (59, 47, 35)      # 本文
SUB       = (122, 106, 85)    # 補足
GOLD      = (176, 130, 30)
RED       = (183, 48, 42)
GREEN     = (43, 110, 78)
BLUE      = (44, 95, 138)
BLUE_TAG  = (44, 95, 138)
GOLD_TAG  = (176, 130, 30)
PAW       = (214, 196, 165)

W, MARGIN = 1200, 30
CARD_W = W - 2 * MARGIN


class F:
    def __init__(self, cfg):
        self._cache = {}
        self.reg = cfg.font_path
        self.bold = cfg.font_path_bold

    def __call__(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self._cache:
            try:
                self._cache[key] = ImageFont.truetype(self.bold if bold else self.reg, size, index=0)
            except Exception:
                self._cache[key] = ImageFont.load_default()
        return self._cache[key]


def _wrap(d, text, font, maxw):
    out, cur = [], ""
    for ch in text:
        if (d.textbbox((0, 0), cur + ch, font=font)[2]) > maxw:
            out.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _paw(d, cx, cy, s=1.0, color=PAW):
    """肉球を描く(マスコット画像が無いときの装飾)。"""
    r = 11 * s
    d.ellipse([cx - r * 1.15, cy - r * 0.55, cx + r * 1.15, cy + r * 1.25], fill=color)
    for dx, dy, rr in ((-1.25, -1.35, 0.52), (-0.42, -1.75, 0.52),
                       (0.42, -1.75, 0.52), (1.25, -1.35, 0.52)):
        d.ellipse([cx + dx * r - rr * r, cy + dy * r - rr * r,
                   cx + dx * r + rr * r, cy + dy * r + rr * r], fill=color)


def _mascot(cfg):
    """assets/mascot.png があれば読み込む(無ければ None)。"""
    for p in (Path("assets/mascot.png"), Path(getattr(cfg, "outdir", ".")) / "mascot.png"):
        try:
            if p.exists():
                return Image.open(p).convert("RGBA")
        except Exception:
            pass
    return None


def render_png(outpath: str, today: str, meta: dict, sentiment: dict, res: dict,
               position_alerts: list, pending_summary: dict, pending_events: list,
               events: list, cfg) -> str:
    f = F(cfg)
    st = res["stats"]
    sr = res["sector_rank"]
    tc = st["trigger_count"]
    notable = [a for a in (position_alerts or []) if a.get("hit") or a.get("hit") is None]

    est_h = (1150 + len(pending_events) * 62 + len(notable) * 95
             + len(res["picked"]) * 250 + len(res.get("watch", [])) * 62
             + len(events) * 34 + 520)
    img = Image.new("RGB", (W, est_h), BG)
    d = ImageDraw.Draw(img)

    def text(x, y, s, size=20, color=INK, bold=False):
        d.text((x, y), s, font=f(size, bold), fill=color)
        return y + int(size * 1.5)

    def card(y, h, fill=CARD, outline=BORDER, width=2, r=16):
        d.rounded_rectangle([MARGIN, y, W - MARGIN, y + h], r, fill=fill, outline=outline, width=width)

    # ================= ヘッダーバー =================
    y = MARGIN
    d.rounded_rectangle([MARGIN, y, W - MARGIN, y + 86], 20, fill=HEADER)
    _paw(d, MARGIN + 44, y + 44, 1.15, (196, 172, 132))
    d.text((MARGIN + 86, y + 22), "スイングトレード・スクリーニング",
           font=f(34, True), fill=CREAM)
    m = _mascot(cfg)
    if m is not None:
        mh = 74
        m = m.resize((int(m.width * mh / m.height), mh))
        img.paste(m, (W - MARGIN - m.width - 18, y + 6), m)
    else:
        for i in range(3):
            _paw(d, W - MARGIN - 46 - i * 46, y + 44, 0.85, (150, 124, 90))
    y += 86 + 12

    # 日付チップ
    d.rounded_rectangle([MARGIN, y, MARGIN + 176, y + 36], 12, fill=CARD_ALT, outline=BORDER, width=2)
    d.text((MARGIN + 18, y + 6), today, font=f(20, True), fill=SUB)
    y += 36 + 14

    # ================= 地合いバナー =================
    sc = int(sentiment.get("score", 3))
    prov = "(暫定)" if sentiment.get("provisional") else ""
    col = GREEN if sc >= 4 else (GOLD if sc == 3 else RED)
    bg = (234, 244, 236) if sc >= 4 else ((251, 244, 226) if sc == 3 else (250, 233, 229))
    card(y, 108, fill=bg, outline=col, width=3)
    _paw(d, MARGIN + 34, y + 40, 0.9, col)
    d.text((MARGIN + 66, y + 18),
           f"地合い {sc}/5{prov} — {sentiment.get('stance','')} / 本日の候補 {st['picked']}件",
           font=f(27, True), fill=col)
    sub = " / ".join(sentiment.get("reasons", [])[:3])
    if sentiment.get("vi_proxy") is not None:
        sub += f" / VI代理{sentiment['vi_proxy']:.0f}"
    for ln in _wrap(d, sub, f(17), CARD_W - 90)[:1]:
        d.text((MARGIN + 66, y + 56), ln, font=f(17), fill=SUB)
    d.text((MARGIN + 66, y + 80),
           f"保有中 {len(notable)}件 / ゾーン待ち {pending_summary.get('pending', 0)}件",
           font=f(17, True), fill=BLUE)
    y += 108 + 14

    # ================= データ =================
    lines = [(f"データ {meta.get('data_ok','?')}/{meta.get('data_total','?')}"
              f"({meta.get('data_coverage',0)*100:.0f}%) {meta.get('source','')}", 17, SUB, False)]
    cn = coverage_note(meta)
    if cn:
        lines.append(("内訳: " + cn, 16, SUB, False))
    lines.append(("カタリスト中身・需給は取得不可 — 発注前にiSPEED/TDnetで要確認", 18, RED, True))
    if sentiment.get("missing"):
        lines.append(("欠落データ: " + ", ".join(sentiment["missing"]), 17, GOLD, True))
    if sentiment.get("hivol_env"):
        lines.append(("高ボラ環境: " + ("前夜SOX反発あり → 値がさ大型は高ボラタグ付きで対象"
                      if sentiment.get("sox_rebound")
                      else sentiment.get("semis_reason", "値がさ大型は除外")), 17, GOLD, True))
    h = 20 + sum(int(s * 1.5) for _, s, _, _ in lines)
    card(y, h)
    yy = y + 12
    for s_, sz, c_, b_ in lines:
        for ln in _wrap(d, s_, f(sz, b_), CARD_W - 40):
            d.text((MARGIN + 20, yy), ln, font=f(sz, b_), fill=c_); yy += int(sz * 1.5)
    y += h + 14

    # ================= セクター / 点灯サマリー(2カラム) =================
    colw = (CARD_W - 14) // 2
    left = [("セクター(等ウェイト代理・直近5日)", 20, INK, True)]
    if sr["top"]:
        left.append(("↑上位: " + " / ".join(f"{s}({r:+.1f}%)" for s, r in sr["top"]), 18, GREEN, True))
    if sr["bottom"]:
        left.append(("↓回避: " + " / ".join(f"{s}({r:+.1f}%)" for s, r in sr["bottom"]), 18, RED, True))
    right = [(f"本日の候補 — 点灯 材料反応{tc.get('材料反応',0)}"
              f"・押し目{tc.get('押し目',0)}・高値ブレイク{tc.get('高値ブレイク',0)}", 20, INK, True)]
    if st.get("slot_note"):
        right.append(("▼" + st["slot_note"], 17, BLUE, True))
    if st.get("concentration"):
        right.append(("⚠" + st["concentration"], 17, GOLD, True))

    def block_h(items, wdt):
        t = 20
        for s_, sz, _, b_ in items:
            t += len(_wrap(d, s_, f(sz, b_), wdt - 36)) * int(sz * 1.5)
        return t
    bh = max(block_h(left, colw), block_h(right, colw))
    for i, items in enumerate((left, right)):
        x0 = MARGIN + i * (colw + 14)
        d.rounded_rectangle([x0, y, x0 + colw, y + bh], 16, fill=CARD, outline=BORDER, width=2)
        yy = y + 12
        for s_, sz, c_, b_ in items:
            for ln in _wrap(d, s_, f(sz, b_), colw - 36):
                d.text((x0 + 18, yy), ln, font=f(sz, b_), fill=c_); yy += int(sz * 1.5)
    y += bh + 14

    if res["picked"]:
        card(y, 50, fill=(250, 233, 229), outline=RED, width=2, r=12)
        d.text((MARGIN + 18, y + 12), "[要確認] iSPEED/TDnetでカタリストを確認するまで発注不可",
               font=f(19, True), fill=RED)
        y += 50 + 14
    else:
        card(y, 58)
        d.text((MARGIN + 20, y + 16), "該当なし — ゼロ件はゼロ件。無理に格下げ採用しない。",
               font=f(20, True), fill=SUB)
        y += 58 + 14

    # ================= 候補カード =================
    BOXW = 182
    boxes_w = BOXW * 3 + 20
    rx = MARGIN + 26 + boxes_w + 22            # 右カラムの開始x
    rw = W - MARGIN - 22 - rx                  # 右カラムの幅

    for i, c in enumerate(res["picked"], 1):
        trig_lines = _wrap(d, c.trigger_text, f(17), CARD_W - 60)
        rlines = _wrap(d, f"失効 {c.expire_date}({cfg.zone_expire_days}営業日) / "
                          f"時間ストップ {c.time_stop}営業日 / 1単元{c.unit_cost/1e4:,.0f}万円",
                       f(17, True), rw)
        rlines += _wrap(d, f"期待度 {c.score:.0f}/10 = {c.score_reason}", f(16), rw)
        risk_lines = []
        for r in c.risks[:2]:
            risk_lines.extend(_wrap(d, "⚠ " + r, f(16), CARD_W - 60))
        for fl in c.flags:
            risk_lines.extend(_wrap(d, fl, f(16, True), CARD_W - 60))

        mid_h = max(84, len(rlines) * 23 + 6)
        ch = 16 + 46 + 24 + len(trig_lines) * 23 + mid_h + len(risk_lines) * 21 + 18
        card(y, ch)

        cx, cy = MARGIN + 26, y + 16
        # 番号丸
        d.ellipse([cx, cy + 2, cx + 38, cy + 40], fill=CARD_ALT, outline=BORDER, width=2)
        d.text((cx + 12 if i < 10 else cx + 6, cy + 8), str(i), font=f(20, True), fill=SUB)
        # トリガータグ
        tagcol = BLUE_TAG if c.tag == "1ヶ月" else GOLD_TAG
        tx = cx + 52
        d.rounded_rectangle([tx, cy + 2, tx + 132, cy + 40], 10, fill=tagcol)
        d.text((tx + 14, cy + 9), c.trigger, font=f(19, True), fill=(255, 252, 246))
        # めやすチップ
        tx2 = tx + 142
        d.rounded_rectangle([tx2, cy + 2, tx2 + 116, cy + 40], 10, fill=CARD_ALT, outline=BORDER, width=2)
        d.text((tx2 + 12, cy + 10), "めやす" + c.tag, font=f(17, True), fill=SUB)
        # 銘柄名
        d.text((tx2 + 132, cy), f"{c.code} {c.name}", font=f(26, True), fill=INK)
        cy += 46

        d.text((cx, cy), f"{c.sector}" + ("(順風)" if c.tailwind else "(逆風)" if c.headwind else ""),
               font=f(17), fill=GREEN if c.tailwind else (RED if c.headwind else SUB))
        cy += 24
        for ln in trig_lines:
            d.text((cx, cy), ln, font=f(17), fill=SUB); cy += 23

        # --- 3ボックス(左) ---
        bx = cx
        d.rounded_rectangle([bx, cy, bx + BOXW, cy + 78], 12, fill=(255, 253, 249),
                            outline=GREEN, width=2)
        d.text((bx + 12, cy + 8), "INゾーン(指値)", font=f(15), fill=SUB)
        d.text((bx + 12, cy + 27), f"{c.zone_hi:,.0f} 円", font=f(20, True), fill=INK)
        d.text((bx + 12, cy + 50), f"〜 {c.zone_lo:,.0f} 円", font=f(20, True), fill=INK)
        b2 = bx + BOXW + 10
        d.rounded_rectangle([b2, cy, b2 + BOXW, cy + 78], 12, fill=(255, 253, 249),
                            outline=RED, width=2)
        d.text((b2 + 12, cy + 8), "STOP(構造)", font=f(15), fill=SUB)
        d.text((b2 + 12, cy + 30), f"{c.stop:,.0f} 円", font=f(23, True), fill=RED)
        d.text((b2 + 12, cy + 58), "どこで入っても同じ", font=f(14), fill=SUB)
        b3 = bx + 2 * (BOXW + 10)
        d.rounded_rectangle([b3, cy, b3 + BOXW, cy + 78], 12, fill=(255, 253, 249),
                            outline=BORDER, width=2)
        d.text((b3 + 12, cy + 8), "1Rの幅(リスク幅)", font=f(15), fill=SUB)
        d.text((b3 + 12, cy + 28), f"浅く {c.risk_shallow:,.0f}円 ({c.risk_pct_shallow:.1f}%)",
               font=f(16, True), fill=INK)
        d.text((b3 + 12, cy + 52), f"深く {c.risk_deep:,.0f}円 ({c.risk_pct_deep:.1f}%)",
               font=f(16, True), fill=GREEN)

        # --- 右カラム(失効・時間ストップ・期待度) ---
        ry = cy + 2
        for k, ln in enumerate(rlines):
            d.text((rx, ry), ln, font=f(17, True) if k < len(rlines) - 1 and "期待度" not in ln
                   else f(16), fill=INK if "期待度" not in ln else SUB)
            ry += 23
        cy += mid_h

        d.text((cx, cy), "出口: +1Rで半分利確(2単元以上) → 残玉は構造まで引上げ+トレーリング",
               font=f(16), fill=BLUE)
        cy += 22
        for ln in risk_lines:
            d.text((cx, cy), ln, font=f(16), fill=GOLD); cy += 21
        _paw(d, W - MARGIN - 34, y + ch - 30, 0.62)
        y += ch + 14

    # ================= ゾーン待ちの動き / 保有 =================
    for title, items, render in (
        ("ゾーン待ち候補の動き", pending_events, "pending"),
        ("保有中の銘柄", notable, "held")):
        if not items:
            continue
        y = text(MARGIN + 4, y + 4, title, 21, INK, True)
        for it in items:
            if render == "pending":
                mark, mc = {"reached": ("到達", GREEN), "expired": ("失効", SUB),
                            "broken": ("下端割れ", RED)}.get(it["event"], ("", SUB))
                lns = _wrap(d, f"[{mark}] {it['code']} {it['name']}: {it['note']}",
                            f(17, True), CARD_W - 44)
                bh2 = 12 + len(lns) * 22 + 8
                card(y, bh2, fill=CARD_ALT, r=12)
                yy = y + 8
                for ln in lns:
                    d.text((MARGIN + 20, yy), ln, font=f(17, True), fill=mc); yy += 22
            else:
                tag, tcol = {"stop": ("ストップ割れ", RED), "partial": ("+1R到達", GREEN),
                             "time": ("時間ストップ", GOLD)}.get(it.get("hit"), ("要確認", GOLD))
                lns = _wrap(d, it["note"], f(16), CARD_W - 240)
                bh2 = 14 + 24 + len(lns) * 21 + 10
                card(y, bh2, fill=CARD_ALT, r=12)
                d.text((MARGIN + 20, y + 10), f"[{tag}] {it['code']} {it['name']}",
                       font=f(18, True), fill=tcol)
                if it.get("days_held") is not None and it.get("time_stop"):
                    d.text((W - MARGIN - 170, y + 12),
                           f"{it['days_held']}/{it['time_stop']}営業日", font=f(17, True), fill=SUB)
                yy = y + 36
                for ln in lns:
                    d.text((MARGIN + 20, yy), ln, font=f(16), fill=SUB); yy += 21
            y += bh2 + 8

    # ================= 次点 / イベント(2カラム) =================
    wl, wr = [], []
    for c in res.get("watch", []):
        wl.append(f"{c.code} {c.name} [{c.trigger}] {c.trigger_text}")
        if c.flags:
            wl.append("　" + c.flags[0])
    if events:
        for e in events:
            wr.append(f"・{e['date']} {e['label']}")
    else:
        wr.append("登録なし — 経済指標カレンダーは自動取得不可(手動確認)")

    def col_h(title, rows, wdt):
        t = 20 + 30
        for r in rows:
            t += len(_wrap(d, r, f(16), wdt - 36)) * 21
        return t
    hh = max(col_h("次点(監視)", wl, colw), col_h("イベント", wr, colw))
    for i, (title, rows) in enumerate((("次点(監視)", wl), ("今週の重要イベント(events.csv・手動管理)", wr))):
        x0 = MARGIN + i * (colw + 14)
        d.rounded_rectangle([x0, y, x0 + colw, y + hh], 16, fill=CARD, outline=BORDER, width=2)
        d.text((x0 + 18, y + 12), title, font=f(19, True), fill=INK)
        yy = y + 46
        for r in rows:
            for ln in _wrap(d, r, f(16), colw - 36):
                d.text((x0 + 18, yy), ln, font=f(16), fill=SUB); yy += 21
    y += hh + 16

    # ================= 最後に: 今日の地合い =================
    cnote = closing_note(sentiment)
    ccol = GREEN if cnote["score"] >= 4 else (GOLD if cnote["score"] == 3 else RED)
    cbg = ((234, 244, 236) if cnote["score"] >= 4
           else ((251, 244, 226) if cnote["score"] == 3 else (250, 233, 229)))
    body = []
    for ln in cnote["lines"]:
        body.extend(_wrap(d, ln, f(17), CARD_W - 90))
    bh3 = 16 + 36 + len(body) * 24 + 14
    card(y, bh3, fill=cbg, outline=ccol, width=3)
    _paw(d, MARGIN + 34, y + 34, 0.85, ccol)
    d.text((MARGIN + 66, y + 14),
           f"最後に: 今日の地合い {cnote['score']}/5"
           f"{'(暫定)' if cnote['provisional'] else ''} — {cnote['stance']}",
           font=f(22, True), fill=ccol)
    yy = y + 52
    for ln in body:
        d.text((MARGIN + 66, yy), ln, font=f(17), fill=INK); yy += 24
    y += bh3 + 14

    for s_ in ["共通リスク: カタリストの中身は未確認(価格痕跡のみ)。悪材料の可能性を常に残す。",
               "免責: AI候補提示で投資助言ではない。最終判断と結果責任はユーザーにある。"]:
        for ln in _wrap(d, s_, f(16), CARD_W):
            y = text(MARGIN + 4, y, ln, 16, SUB)

    img = img.crop((0, 0, W, min(est_h, y + MARGIN)))
    img.save(outpath)
    return outpath
