"""topdown — PNGレポート v3(見た目の全面刷新・2026-07-24).

温かみのあるクリーム/木目調の配色、角丸カード、番号丸、肉球モチーフに変更した。
情報量は v2 から一切減らしていない(安全側の警告・出口・リスクも全て残す)。

マスコット画像について:
  写真調のイラストはPILの図形描画では作れないため、画像ファイルがあれば貼り込む方式にした。
  リポジトリ直下に assets/mascot.png (背景透過推奨) を置くとヘッダー右に合成される。
  無い場合は肉球を描いて代替するので、ファイルが無くてもレイアウトは崩れない。
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

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


# --- マスコット(図形描画のクマ) ---
FUR      = (198, 160, 106)
FUR_DARK = (172, 133, 82)
FUR_IN   = (233, 212, 178)
NOSE     = (74, 55, 40)
BAND     = (250, 245, 235)


def _bear(d, cx, cy, r, band: bool = True):
    """クマの顔を図形で描く。r=顔の半径。写真調のイラストは描けないため、
    角丸カード基調のこのレポートに馴染むフラットな形で作っている。
    左右のパーツはPILのラスタライズで別々に丸められると数画素ずれるため、
    オフセットを整数化して厳密な鏡像になるようにしている。"""
    cx, cy = int(round(cx)), int(round(cy))
    I = lambda v: int(round(v * r))          # 半径比 → 整数オフセット

    # 耳(外→内)
    eo, er, ir = I(0.78), I(0.36), I(0.19)
    ey = cy - I(0.74)
    for sx in (-1, 1):
        ex = cx + sx * eo
        d.ellipse([ex - er, ey - er, ex + er, ey + er], fill=FUR_DARK)
        d.ellipse([ex - ir, ey - ir, ex + ir, ey + ir], fill=FUR_IN)

    # 顔
    R = I(1.0)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=FUR)

    # ハチマキ(顔の円からはみ出さないよう、各高さでの円の半幅に合わせた台形)
    if band:
        t0, t1 = -0.52, -0.26
        hw0 = int(round(r * math.sqrt(max(0.0, 1 - t0 * t0))))
        hw1 = int(round(r * math.sqrt(max(0.0, 1 - t1 * t1))))
        y0, y1 = cy + I(t0), cy + I(t1)
        d.polygon([(cx - hw0, y0), (cx + hw0, y0), (cx + hw1, y1), (cx - hw1, y1)], fill=BAND)
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=FUR_DARK, width=max(1, I(0.05)))

    # 目(閉じた弧・上に凸)
    xo, ew = I(0.40), I(0.20)
    ey0, ey1 = cy - I(0.14), cy + I(0.20)
    for sx in (-1, 1):
        exc = cx + sx * xo
        d.arc([exc - ew, ey0, exc + ew, ey1], start=200, end=340,
              fill=NOSE, width=max(2, I(0.085)))

    # 口元(明るい楕円 + 鼻 + ∪型の弧2つ)
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
        # PILの角度は3時から時計回り。15→165は下半分を通るので∪型になる
        d.arc([x0, my1, x1, my2], start=15, end=165, fill=NOSE, width=lw)


def _asset(name: str, cfg):
    """assets/<name> があれば読み込む(無ければ None)。

    写真素材(実写のマスコット・ヒーロー画像)はコードでは描けないため、
    リポジトリ直下の assets/ に置く方式にした。無い場合は図形描画で代替する。
    """
    # assets/ に置くのが本来だが、GitHubアプリはフォルダ作成・画像アップロードができない。
    # リポジトリ直下に置いただけでも拾えるよう、探索先を広げてある。
    for p in (Path("assets") / name, Path(name),
              Path(getattr(cfg, "outdir", ".")) / name):
        try:
            if p.exists():
                return Image.open(p).convert("RGBA")
        except Exception:
            pass
    return None


def _mascot(cfg):
    # 透過が必要なのでPNGのみ
    return _asset("mascot.png", cfg)


def _hero(cfg):
    # 背景写真は透過不要。容量が小さいJPEGも受け付ける
    for name in ("hero.jpg", "hero.jpeg", "hero.png"):
        im = _asset(name, cfg)
        if im is not None:
            return im
    return None


def _paste_bg(img, photo, H: int, photo_h: int = 620, wash: float = 0.16,
              tex_wash: float = 0.62):
    """背景を作る。

    参考図と同じく「クマ+お社」を右上に丸ごと収めるため、写真は幅いっぱいに
    引き伸ばさず、指定した高さに縮小して右上へ配置する(以前は右へずらしていたため
    お社・だるまがキャンバス外に切れていた)。
    残りの領域は写真の卓上(左下)を強くぼかしたテクスチャで埋め、左端と下端を
    フェードして写真をなじませる。
    """
    pw, ph = photo.size
    p = photo.convert("RGB")

    # 下地: 卓上あたりを強くぼかして全面に敷き、クリームと混ぜて淡くする
    tex = p.crop((0, int(ph * 0.72), pw, ph)).resize((W, max(1, H)), Image.LANCZOS)
    tex = tex.filter(ImageFilter.GaussianBlur(26))
    tex = Image.blend(tex, Image.new("RGB", tex.size, BG), tex_wash)
    img.paste(tex, (0, 0))

    # 写真本体: 高さを合わせて右上へ。全幅が入るので被写体が切れない
    sw = max(1, int(pw * photo_h / ph))
    r = p.resize((sw, photo_h), Image.LANCZOS)
    r = Image.blend(r, Image.new("RGB", r.size, BG), wash)
    x0 = max(0, W - sw)

    # 左端と下端をフェードして下地へ溶かす
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


def render_png(outpath: str, today: str, meta: dict, sentiment: dict, res: dict,
               position_alerts: list, pending_summary: dict, pending_events: list,
               events: list, cfg) -> str:
    f = F(cfg)
    st = res["stats"]
    sr = res["sector_rank"]
    tc = st["trigger_count"]
    notable = [a for a in (position_alerts or []) if a.get("hit") or a.get("hit") is None]

    est_h = (1250 + len(pending_events) * 62 + len(notable) * 95
             + len(res["picked"]) * 250 + len(res.get("watch", [])) * 62
             + len(events) * 34 + 620)
    img = Image.new("RGB", (W, est_h), BG)
    hero = _hero(cfg)
    if hero is not None:
        _paste_bg(img, hero, est_h)
    d = ImageDraw.Draw(img)

    m = _mascot(cfg)
    # 写真が無い場合、上部カードを狭いままにすると右側が空白になるので全幅にする
    HAS_PHOTO = hero is not None

    def mascot_at(x, y, size):
        """マスコット写真を貼る。無ければ図形のクマを描いて代替。"""
        if m is not None:
            mm = m.resize((size, size), Image.LANCZOS)
            img.paste(mm, (int(x), int(y)), mm)
        else:
            _bear(d, x + size / 2, y + size / 2, size / 2 * 0.86)

    def fit(txt, maxw, size, bold=True, minsize=14):
        """maxw に収まる最大のフォントサイズを返す(収まらなければ縮める)。"""
        while size > minsize and d.textbbox((0, 0), txt, font=f(size, bold))[2] > maxw:
            size -= 1
        return size

    def text(x, y, s, size=20, color=INK, bold=False):
        d.text((x, y), s, font=f(size, bold), fill=color)
        return y + int(size * 1.5)

    TOPW = int(CARD_W * 0.52) if HAS_PHOTO else CARD_W

    def card(y, h, fill=CARD, outline=BORDER, width=2, r=16, w=None, x=MARGIN):
        d.rounded_rectangle([x, y, x + (w if w else CARD_W), y + h], r,
                            fill=fill, outline=outline, width=width)

    # ================= ヘッダー(クマが左上に重なる) =================
    y = MARGIN + 14
    d.rounded_rectangle([MARGIN + 62, y, MARGIN + TOPW, y + 76], 20, fill=HEADER)
    # タイトルはバーの内側に必ず収める。入らなければフォントを1ptずつ縮める。
    t1, t2 = "AIトム", "のスイングトレード・スクリーニング"
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
    mascot_at(MARGIN - 4, y - 14, 92)          # ヘッダーに重なるクマ
    y += 76 + 12

    d.rounded_rectangle([MARGIN, y, MARGIN + 176, y + 36], 12,
                        fill=CARD_ALT, outline=BORDER, width=2)
    d.text((MARGIN + 18, y + 6), today, font=f(20, True), fill=SUB)
    y += 36 + 12

    # ================= 地合いバナー =================
    sc = int(sentiment.get("score", 3))
    prov = "(暫定)" if sentiment.get("provisional") else ""
    col = GREEN if sc >= 4 else (GOLD if sc == 3 else RED)
    bg = (234, 244, 236) if sc >= 4 else ((251, 244, 226) if sc == 3 else (250, 233, 229))
    sub = " / ".join(sentiment.get("reasons", [])[:4])
    if sentiment.get("vi_proxy") is not None:
        sub += f" / VI代理{sentiment['vi_proxy']:.0f}"
    sub_l = _wrap(d, sub, f(17), TOPW - 110)[:2]      # 途中で切らず2行まで
    ban_h = 84 + len(sub_l) * 24
    card(y, ban_h, fill=bg, outline=col, width=3, w=TOPW)
    mascot_at(MARGIN + 14, y + 20, 60)
    headline = (f"地合い {sc}/5{prov} — {sentiment.get('stance','')}"
                f" / 本日の候補 {st['picked']}件")
    d.text((MARGIN + 88, y + 14), headline,
           font=f(fit(headline, TOPW - 104, 26), True), fill=col)
    yy = y + 50
    for ln in sub_l:
        d.text((MARGIN + 88, yy), ln, font=f(17), fill=SUB); yy += 24
    _paw(d, MARGIN + 96, yy + 10, 0.5, BLUE)
    d.text((MARGIN + 112, yy),
           f"保有中 {len(notable)}件 / ゾーン待ち {pending_summary.get('pending', 0)}件",
           font=f(17, True), fill=BLUE)
    y += ban_h + 12

    # ================= データ =================
    lines = [(f"データ {meta.get('data_ok','?')}/{meta.get('data_total','?')}"
              f"({meta.get('data_coverage',0)*100:.0f}%) {meta.get('source','')}", 17, SUB, False)]
    cnv = coverage_note(meta)
    if cnv:
        lines.append(("内訳: " + cnv, 16, SUB, False))
    lines.append(("カタリスト中身・需給は取得不可 — 発注前にiSPEED/TDnetで要確認", 18, RED, True))
    if sentiment.get("missing"):
        lines.append(("欠落データ: " + ", ".join(sentiment["missing"]), 17, GOLD, True))
    if sentiment.get("hivol_env"):
        lines.append(("高ボラ環境: " + ("前夜SOX反発あり → 値がさ大型は高ボラタグ付きで対象"
                      if sentiment.get("sox_rebound")
                      else sentiment.get("semis_reason", "値がさ大型は除外")), 17, GOLD, True))
    h = 20 + sum(len(_wrap(d, s_, f(sz, b_), TOPW - 40)) * int(sz * 1.5)
                 for s_, sz, _, b_ in lines)
    card(y, h, w=TOPW)
    yy = y + 12
    for s_, sz, c_, b_ in lines:
        for ln in _wrap(d, s_, f(sz, b_), TOPW - 40):
            d.text((MARGIN + 20, yy), ln, font=f(sz, b_), fill=c_); yy += int(sz * 1.5)
    y += h + 12

    # ================= セクター / 本日の候補(2カラム) =================
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
    warn_h = 46 if res["picked"] else 0
    bh = max(block_h(left, colw), block_h(right, colw) + warn_h)
    for i, items in enumerate((left, right)):
        x0 = MARGIN + i * (colw + 14)
        d.rounded_rectangle([x0, y, x0 + colw, y + bh], 16, fill=CARD, outline=BORDER, width=2)
        yy = y + 12
        for s_, sz, c_, b_ in items:
            for ln in _wrap(d, s_, f(sz, b_), colw - 36):
                d.text((x0 + 18, yy), ln, font=f(sz, b_), fill=c_); yy += int(sz * 1.5)
        # 要確認の赤枠は右カラムの中に入れる(参考図と同じ)
        if i == 1 and res["picked"]:
            d.rounded_rectangle([x0 + 14, y + bh - 52, x0 + colw - 14, y + bh - 12], 10,
                                fill=(250, 233, 229), outline=RED, width=2)
            d.text((x0 + 28, y + bh - 45), "[要確認] iSPEED/TDnetでカタリストを確認するまで発注不可",
                   font=f(17, True), fill=RED)
    y += bh + 14

    if not res["picked"]:
        card(y, 58)
        d.text((MARGIN + 20, y + 16), "該当なし — ゼロ件はゼロ件。無理に格下げ採用しない。",
               font=f(20, True), fill=SUB)
        y += 58 + 14

    # ================= 候補カード =================
    BOXW = 182
    rx = MARGIN + 26 + BOXW * 3 + 20 + 22
    rw = W - MARGIN - 22 - rx

    for i, c in enumerate(res["picked"], 1):
        trig_lines = _wrap(d, c.trigger_text, f(17), CARD_W - 60)
        rlines = [(f"失効 {c.expire_date}({cfg.zone_expire_days}営業日) / "
                   f"時間ストップ {c.time_stop}営業日 / 1単元{c.unit_cost/1e4:,.0f}万円", True)]
        rlines = [(ln, True) for ln in _wrap(d, rlines[0][0], f(17, True), rw)]
        rlines += [(ln, False) for ln in
                   _wrap(d, f"期待度 {c.score:.0f}/10 = {c.score_reason}", f(16), rw)]
        risk_lines = []
        for r in c.risks[:2]:
            risk_lines.extend(_wrap(d, "⚠ " + r, f(16), CARD_W - 60))
        for fl in c.flags:
            risk_lines.extend(_wrap(d, fl, f(16, True), CARD_W - 60))

        mid_h = max(84, len(rlines) * 23 + 6)
        ch = 16 + 46 + 24 + len(trig_lines) * 23 + mid_h + len(risk_lines) * 21 + 18
        card(y, ch)
        _paw(d, W - MARGIN - 36, y + 30, 0.72)     # 参考図と同じく右上に肉球

        cx, cy = MARGIN + 26, y + 16
        d.ellipse([cx, cy + 2, cx + 38, cy + 40], fill=CARD_ALT, outline=BORDER, width=2)
        d.text((cx + (12 if i < 10 else 6), cy + 8), str(i), font=f(20, True), fill=SUB)
        mascot_at(cx + 46, cy + 2, 38)
        tagcol = BLUE_TAG if c.tag == "1ヶ月" else GOLD_TAG
        tx = cx + 94
        d.rounded_rectangle([tx, cy + 2, tx + 132, cy + 40], 10, fill=tagcol)
        d.text((tx + 14, cy + 9), c.trigger, font=f(19, True), fill=(255, 252, 246))
        tx2 = tx + 142
        d.rounded_rectangle([tx2, cy + 2, tx2 + 116, cy + 40], 10,
                            fill=CARD_ALT, outline=BORDER, width=2)
        d.text((tx2 + 12, cy + 10), "めやす" + c.tag, font=f(17, True), fill=SUB)
        d.text((tx2 + 132, cy), f"{c.code} {c.name}", font=f(26, True), fill=INK)
        cy += 46

        d.text((cx, cy), f"{c.sector}" + ("(順風)" if c.tailwind else "(逆風)" if c.headwind else ""),
               font=f(17), fill=GREEN if c.tailwind else (RED if c.headwind else SUB))
        cy += 24
        for ln in trig_lines:
            d.text((cx, cy), ln, font=f(17), fill=SUB); cy += 23

        for k, (bx, oc, lbl, v1, v2, c1, c2) in enumerate((
                (cx, GREEN, "INゾーン(指値)", f"{c.zone_hi:,.0f} 円", f"〜 {c.zone_lo:,.0f} 円", INK, INK),
                (cx + BOXW + 10, RED, "STOP(構造)", f"{c.stop:,.0f} 円", "どこで入っても同じ", RED, SUB),
                (cx + 2 * (BOXW + 10), BORDER, "1Rの幅(リスク幅)",
                 f"浅く {c.risk_shallow:,.0f}円 ({c.risk_pct_shallow:.1f}%)",
                 f"深く {c.risk_deep:,.0f}円 ({c.risk_pct_deep:.1f}%)", INK, GREEN))):
            d.rounded_rectangle([bx, cy, bx + BOXW, cy + 78], 12,
                                fill=(255, 253, 249), outline=oc, width=2)
            d.text((bx + 12, cy + 8), lbl, font=f(15), fill=SUB)
            if k == 1:
                d.text((bx + 12, cy + 30), v1, font=f(23, True), fill=c1)
                d.text((bx + 12, cy + 58), v2, font=f(14), fill=c2)
            elif k == 0:
                d.text((bx + 12, cy + 27), v1, font=f(20, True), fill=c1)
                d.text((bx + 12, cy + 50), v2, font=f(20, True), fill=c2)
            else:
                d.text((bx + 12, cy + 28), v1, font=f(16, True), fill=c1)
                d.text((bx + 12, cy + 52), v2, font=f(16, True), fill=c2)

        ry = cy + 2
        for ln, bold in rlines:
            d.text((rx, ry), ln, font=f(17, True) if bold else f(16),
                   fill=INK if bold else SUB)
            ry += 23
        cy += mid_h

        d.text((cx, cy), "出口: +1Rで半分利確(2単元以上) → 残玉は構造まで引上げ+トレーリング",
               font=f(16), fill=BLUE)
        cy += 22
        for ln in risk_lines:
            d.text((cx, cy), ln, font=f(16), fill=GOLD); cy += 21
        y += ch + 14

    # ================= ゾーン待ち / 保有 =================
    for title, items, kind in (("ゾーン待ち候補の動き", pending_events, "pending"),
                               ("保有中の銘柄", notable, "held")):
        if not items:
            continue
        y = text(MARGIN + 4, y + 4, title, 21, INK, True)
        for it in items:
            if kind == "pending":
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

    # ================= 次点 | イベント+吹き出し(2カラム) =================
    wl = []
    for c in res.get("watch", []):
        wl.append(f"{c.code} {c.name} [{c.trigger}] {c.trigger_text}")
        if c.flags:
            wl.append("　" + c.flags[0])
    wr = [f"・{e['date']} {e['label']}" for e in events] or          ["登録なし — 経済指標カレンダーは自動取得不可(手動確認)"]

    cnote = closing_note(sentiment)
    ccol = GREEN if cnote["score"] >= 4 else (GOLD if cnote["score"] == 3 else RED)
    BEARW = 132
    bubw = colw - BEARW - 12
    bub = [f"地合い {cnote['score']}/5{'(暫定)' if cnote['provisional'] else ''}"
           f" — {cnote['stance']}"]
    bub += cnote["lines"]
    bub_l = []
    for i, ln in enumerate(bub):
        bub_l.extend([(x, i == 0) for x in _wrap(d, ln, f(16, i == 0), bubw - 44)])

    lh = 20 + 30 + sum(len(_wrap(d, r, f(16), colw - 36)) * 21 for r in wl)
    ev_h = 20 + 30 + sum(len(_wrap(d, r, f(16), colw - 36)) * 21 for r in wr)
    bub_h = 20 + len(bub_l) * 22 + 12
    rh = ev_h + 12 + max(bub_h, 118)
    hh = max(lh, rh)

    d.rounded_rectangle([MARGIN, y, MARGIN + colw, y + hh], 16, fill=CARD,
                        outline=BORDER, width=2)
    d.text((MARGIN + 18, y + 12), "次点(監視)", font=f(19, True), fill=INK)
    yy = y + 46
    for r in wl:
        for ln in _wrap(d, r, f(16), colw - 36):
            d.text((MARGIN + 18, yy), ln, font=f(16), fill=SUB); yy += 21

    xr = MARGIN + colw + 14
    d.rounded_rectangle([xr, y, xr + colw, y + ev_h], 16, fill=CARD, outline=BORDER, width=2)
    d.text((xr + 18, y + 12), "今週の重要イベント(events.csv・手動管理)", font=f(19, True), fill=INK)
    yy = y + 46
    for r in wr:
        for ln in _wrap(d, r, f(16), colw - 36):
            d.text((xr + 18, yy), ln, font=f(16), fill=SUB); yy += 21

    by = y + ev_h + 12
    d.rounded_rectangle([xr, by, xr + bubw, by + bub_h], 18, fill=(255, 253, 249),
                        outline=ccol, width=3)
    d.polygon([(xr + bubw, by + bub_h * 0.42), (xr + bubw + 22, by + bub_h * 0.52),
               (xr + bubw, by + bub_h * 0.66)], fill=(255, 253, 249), outline=ccol)
    yy = by + 12
    for ln, bold in bub_l:
        d.text((xr + 20, yy), ln, font=f(16, bold), fill=ccol if bold else INK); yy += 22
    mascot_at(xr + bubw + 26, by + max(0, (bub_h - BEARW + 40) // 2), min(BEARW - 24, 116))
    y += hh + 16

    # ================= フッター =================
    for s_ in ["共通リスク: カタリストの中身は未確認(価格痕跡のみ)。悪材料の可能性を常に残す。",
               "免責: AI候補提示で投資助言ではない。最終判断と結果責任はユーザーにある。"]:
        for ln in _wrap(d, s_, f(16), CARD_W):
            y = text(MARGIN + 4, y, ln, 16, SUB)
    y += 8
    sig = "スイングトレーダー AIトム"
    sw = d.textbbox((0, 0), sig, font=f(19, True))[2]
    _paw(d, W / 2 - sw / 2 - 22, y + 12, 0.62, (150, 124, 90))
    d.text((W / 2 - sw / 2, y), sig, font=f(19, True), fill=(110, 88, 62))
    y += 34

    img = img.crop((0, 0, W, min(est_h, y + MARGIN)))
    img.save(outpath)
    return outpath
