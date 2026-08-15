"""v7.1 第3部 — 次元定義(①〜⑤)。

共通ルール(§3.0):
  - 各次元スコアは 0.0〜1.0
  - 次元内の複数サブ成分は単純平均(多数決を使わない)
  - 各次元にハードフロアを設ける。未達なら他成分に関わらず 0.0
  - 二値化しない。離散判定はカバレッジ判定でのみ行う

なぜ平均であり多数決でないか(§3.0.1):
  多数決は離散閾値であり、各成分の強度情報と閾値近傍の連続性を失う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import (sma, atr_wilder, slope_normalized, to_weekly, find_swings,
                   alternate_swings, clamp01, linear_score, peak_score, sign_score,
                   kojiro_stage, kojiro_stage_days)


# ④時間に関する定数。
#
# D4B_MAX_DAYS: 4-b「200日線を上抜けてからの経過日数」の測定上限。探索ループの
#   上限であり、同時にpeak_scoreのhi(=これ以上は「遅すぎ」として0)でもある。
#
#   ★この値は必ず「実際に測定可能な日数」より小さくなければならない。
#     測定可能日数 = 取得したバー数 − SMA200のウォームアップ(199本)。
#     ここを超える値を置くと、日数がデータ窓の端で頭打ちになり、
#     「銘柄の性質」ではなく「取得設定」に依存した値が混入する。
#     しかもスコアとしては成立してしまうため、誰も気づけない。
#     (実際にhi=400・探索上限300で運用され、約120銘柄が上限に張り付いていた)
#
#   250は「200日線超えから1年強＝十分に遅い」を表す。これ以上細かく
#   「どれだけ遅いか」を区別する意味は薄いため、250以上は一律0でよい。
#   history_days=500(≒330〜500バー)に対して十分内側であり、
#   多少データ長が変動しても定数側が拘束条件であり続ける。
D4B_MAX_DAYS = 250
D4B_LO_DAYS = 5        # これ以下は「上抜け直後で未確認」
D4B_IDEAL_DAYS = 60    # ワインスタインStage2として理想的な経過
SMA200_WARMUP = 199    # sma(c,200)の先頭NaN本数

# ④は4項目すべてが揃って初めて意味を持つ(§9.3)。一部欠落のまま平均すると
# 銘柄間で分母が変わり比較できなくなるため、揃わない場合はNoneを返す。
D4_REQUIRED_PARTS = ("4-a", "4-b", "4-c", "4-d")


# ============================================================
# ① トレンド方向 — 自己価格系列における上昇トレンドの成立度
#    証拠: 階層B(時系列モメンタム。日本株個別での直接検証なし)
#    ハードフロア: 長期移動平均線が下向き → 0.0
# ============================================================

def dim1_trend(daily: pd.DataFrame, cfg) -> dict:
    c = daily["Close"].dropna()
    if len(c) < cfg.d1_ma_long + cfg.d1_slope_window:
        return {"score": None, "parts": {}, "reason": "履歴不足"}

    s50 = sma(c, cfg.d1_ma_short)
    s150 = sma(c, cfg.d1_ma_mid)
    s200 = sma(c, cfg.d1_ma_long)
    if any(pd.isna(x.iloc[-1]) for x in (s50, s150, s200)):
        return {"score": None, "parts": {}, "reason": "MA算出不可"}

    close_now = float(c.iloc[-1])
    v50, v150, v200 = float(s50.iloc[-1]), float(s150.iloc[-1]), float(s200.iloc[-1])

    # ハードフロア: 長期線が下向きなら他成分に関わらず0.0
    slope200 = slope_normalized(s200, cfg.d1_slope_window)
    if slope200 is None or slope200 < 0:
        return {"score": 0.0, "parts": {}, "reason": "ハードフロア: 200日線が下向き"}

    p = {}
    p["1-a"] = 1.0 if close_now > v200 else 0.0
    # 1-b: 傾きを連続写像。日次0.05%/バー を基準スケールとする
    p["1-b"] = sign_score(slope200, 0.0005)
    p["1-c"] = (int(v50 > v150) + int(v150 > v200)) / 2.0
    p["1-d"] = 1.0 if close_now > v50 else 0.0

    # 1-e/1-f は階層E(Liu-Liu-Ma 2011: 日本で非有意)。切替可能にしてある
    if cfg.d1_use_52w_components and len(c) >= 252:
        lo52 = float(c.iloc[-252:].min())
        hi52 = float(c.iloc[-252:].max())
        if lo52 > 0:
            gain = close_now / lo52 - 1.0
            p["1-e"] = linear_score(gain, 0.0, cfg.d1_low52_gain)
        if hi52 > 0:
            gap = (hi52 - close_now) / hi52
            # 乖離が小さいほど高スコア(閾値超で0.0へ逓減)
            p["1-f"] = clamp01(1.0 - gap / cfg.d1_high52_gap)

    # 1-g: 週足長期線
    try:
        w = to_weekly(daily)["Close"]
        wa, wb = sma(w, cfg.d1_weekly_ma_a), sma(w, cfg.d1_weekly_ma_b)
        cw = float(w.iloc[-1])
        above = 0
        if not pd.isna(wa.iloc[-1]) and cw > float(wa.iloc[-1]): above += 1
        if not pd.isna(wb.iloc[-1]) and cw > float(wb.iloc[-1]): above += 1
        p["1-g"] = above / 2.0
    except Exception:
        pass

    # 1-h: 小次郎ステージ(階層D)
    st = kojiro_stage(c, cfg.d1_stage_short, cfg.d1_stage_mid, cfg.d1_stage_long)
    if st:
        p["1-h"] = cfg.d1_stage_scores.get(st, 0.0)

    score = float(np.mean(list(p.values()))) if p else None
    return {"score": score, "parts": p, "reason": "",
            "meta": {"stage": st, "slope200": slope200}}


# ============================================================
# ② 相対力 — 他資産に対する優位性
#    証拠: 階層C(残差モメンタムは論争中)
#    直近1ヶ月除外は階層A。「変更しない」(第12部)
#    ハードフロア: 対市場残差モメンタムが下位一定%タイル → 0.0
# ============================================================

def dim2_relative_raw(daily: pd.DataFrame, market: pd.Series, cfg) -> dict:
    """②の生値を返す。パーセンタイル化はユニバース全体で行うため別段階。"""
    c = daily["Close"].dropna()
    need = cfg.d2_lookback_days + cfg.d2_skip_days + 5
    if len(c) < need or len(market) < need:
        return {"resid_mom": None, "rs_slope": None, "reason": "履歴不足"}

    # 2-a: 直近1ヶ月を除外したリターンの、市場に対する残差
    # skip期間を除くことで短期リバーサル(階層A)の混入を防ぐ
    try:
        end = -cfg.d2_skip_days
        start = end - cfg.d2_lookback_days
        r_stock = float(c.iloc[end] / c.iloc[start] - 1)
        m = market.reindex(c.index).ffill().dropna()
        if len(m) < need:
            return {"resid_mom": None, "rs_slope": None, "reason": "市場系列不足"}
        r_market = float(m.iloc[end] / m.iloc[start] - 1)
        resid = r_stock - r_market
    except Exception:
        return {"resid_mom": None, "rs_slope": None, "reason": "算出エラー"}

    # 2-d: RS線(対市場比価)の傾き
    rs_slope = None
    try:
        ratio = (c / m.reindex(c.index).ffill()).dropna()
        rs_slope = slope_normalized(ratio, cfg.d2_rs_slope_window)
    except Exception:
        pass

    return {"resid_mom": resid, "rs_slope": rs_slope, "reason": ""}


def dim2_relative(raw: dict, resid_pct: float | None,
                  sector_rank_pct: float | None, sector_self_pct: float | None,
                  cfg) -> dict:
    """②のスコア化。resid_pct等はユニバース内パーセンタイル(0〜100)。"""
    if raw.get("resid_mom") is None:
        return {"score": None, "parts": {}, "reason": raw.get("reason", "算出不可")}

    # ハードフロア: 残差モメンタムが下位%タイル
    if resid_pct is not None and resid_pct < cfg.d2_floor_percentile:
        return {"score": 0.0, "parts": {},
                "reason": f"ハードフロア: 残差モメンタムが下位{cfg.d2_floor_percentile:.0f}%タイル"}

    p = {}
    if resid_pct is not None:
        p["2-a"] = clamp01(resid_pct / 100.0)
    if sector_rank_pct is not None:
        p["2-b"] = clamp01(sector_rank_pct / 100.0)
    if sector_self_pct is not None:
        p["2-c"] = clamp01(sector_self_pct / 100.0)
    if raw.get("rs_slope") is not None:
        p["2-d"] = sign_score(raw["rs_slope"], 0.0005)

    score = float(np.mean(list(p.values()))) if p else None
    return {"score": score, "parts": p, "reason": ""}


# ============================================================
# ③ 需給・出来高 — 価格形成の背後にある売買主体の動き
#    証拠: 階層D(日本株での増分説明力に査読研究なし)
#    重要: ブレイク時の出来高急増はエントリー・トリガーであって
#          スクリーニング条件ではない。ベース期間中の蓄積のみ評価(§3-③)
# ============================================================

def dim3_supply_demand(daily: pd.DataFrame, cfg, base_start_idx: int | None = None) -> dict:
    v = daily["Volume"].dropna()
    c = daily["Close"].dropna()
    if len(v) < cfg.d3_vol_long + 5 or len(c) < cfg.d3_vol_long + 5:
        return {"score": None, "parts": {}, "reason": "履歴不足"}

    p = {}

    # 3-a: ベース形成期間内の出来高トレンド(逓減が望ましい)
    seg = v.iloc[base_start_idx:] if (base_start_idx is not None and
                                      base_start_idx < len(v) - 5) else v.iloc[-cfg.d3_vol_long:]
    sl = slope_normalized(seg, min(len(seg), cfg.d3_vol_long))
    if sl is not None:
        # 逓減(負の傾き)を高スコアに。符号を反転
        p["3-a"] = sign_score(-sl, 0.01)

    # 3-b: 上昇日の平均出来高 ÷ 下落日の平均出来高
    try:
        w = min(cfg.d3_updown_window, len(c) - 1)
        ret = c.pct_change().iloc[-w:]
        vv = v.iloc[-w:]
        up_v = float(vv[ret > 0].mean()) if (ret > 0).any() else np.nan
        dn_v = float(vv[ret < 0].mean()) if (ret < 0).any() else np.nan
        if np.isfinite(up_v) and np.isfinite(dn_v) and dn_v > 0:
            p["3-b"] = linear_score(up_v / dn_v, 0.7, 1.5)
    except Exception:
        pass

    # 3-c: 直近出来高 ÷ 長期平均(ベース期は低いほど良い=逓減の裏付け)
    try:
        short = float(v.iloc[-cfg.d3_vol_short:].mean())
        long_ = float(v.iloc[-cfg.d3_vol_long:].mean())
        if long_ > 0:
            ratio = short / long_
            # 0.6〜1.0が理想。1.5超で0
            p["3-c"] = peak_score(ratio, 0.3, 0.75, 1.6)
    except Exception:
        pass

    # 3-d: 出来高増を伴う下落日の頻度(少ないほど高スコア)
    try:
        w = min(cfg.d3_updown_window, len(c) - 1)
        ret = c.pct_change().iloc[-w:]
        vv = v.iloc[-w:]
        vma = v.rolling(cfg.d3_vol_long).mean().iloc[-w:]
        heavy_down = ((ret < 0) & (vv > vma)).sum()
        p["3-d"] = clamp01(1.0 - float(heavy_down) / max(1, w) / 0.3)
    except Exception:
        pass

    score = float(np.mean(list(p.values()))) if p else None
    return {"score": score, "parts": p, "reason": ""}


# ============================================================
# ④ 時間・日柄 — セットアップの成熟度
#    証拠: 階層D(日本株のベース期間分布に実証が存在しない)
#    ①②③⑤が空間軸なのに対し、④のみが時間軸(§3-④)
#    各成分は両側評価。短すぎるも長すぎるも低スコア
# ============================================================

def detect_base(daily: pd.DataFrame, cfg) -> dict | None:
    """ベース(保ち合い)の起点・終点・深さを機械的に定める(§3-④ IBD基準)。

    base_start = 先行上昇の後、初めて閾値超で下落した週
    base_end   = 直近(ブレイク前)
    """
    try:
        w = to_weekly(daily)
        if len(w) < 12:
            return None
        wc = w["Close"]
        # 直近の週足最高値をベース上限の候補とする
        look = min(len(w), cfg.d4_base_max_weeks + 10)
        seg = wc.iloc[-look:]
        peak_pos = int(np.argmax(seg.values))
        peak_price = float(seg.iloc[peak_pos])
        # そこから閾値超の下落が始まった週をベース起点とする
        start_pos = None
        for i in range(peak_pos, len(seg)):
            if (peak_price - float(seg.iloc[i])) / peak_price >= cfg.d4_base_threshold:
                start_pos = peak_pos
                break
        if start_pos is None:
            return None
        base = w.iloc[-(look - start_pos):]
        if len(base) < 2:
            return None
        bh, bl = float(base["High"].max()), float(base["Low"].min())
        return {
            "weeks": len(base),
            "base_high": bh, "base_low": bl,
            "depth": (bh - bl) / bh if bh > 0 else None,
            "start_date": base.index[0],
        }
    except Exception:
        return None


def dim4_time(daily: pd.DataFrame, cfg, base: dict | None) -> dict:
    c = daily["Close"].dropna()
    if len(c) < 120:
        return {"score": None, "parts": {}, "reason": "履歴不足"}

    p = {}

    # 4-a: ベース継続週数(両側評価)
    if base and base.get("weeks"):
        p["4-a"] = peak_score(base["weeks"], cfg.d4_base_min_weeks - 2,
                              cfg.d4_base_ideal_weeks, cfg.d4_base_max_weeks)

    # 4-b: 直前の下降局面終了からの経過(ワインスタインStage2の経過日数)。
    # 200日線を上抜けてからの経過日数で近似する。
    #
    # この指標は本質的に上に開いた量(いくらでも長くなりうる)なので、
    # peak_scoreの有限なhiに通すには必ず飽和点が要る。問題はその飽和点が
    # 「定数」で決まるか「たまたまのデータ長」で決まるかで、後者だと
    # 銘柄の性質ではなく取得設定に依存した値が混じる(旧実装がこれだった)。
    #
    # そこで:
    #   1. 飽和点をD4B_MAX_DAYSに固定し、探索上限とhiを同一の定数から引く
    #   2. 測定可能日数(バー数 − SMA200ウォームアップ)が飽和点に満たない
    #      銘柄は、上限に張り付いた値を返さずNone(測定不能)とする
    #   3. 現在200日線より下にある銘柄はNoneとする。「Stage 2に入って
    #      いない」のは「Stage 2で日柄が悪い」のと違う。ここを区別せず
    #      0を返すと、「まだ始まっていない」銘柄と「250日超え継続で
    #      終わりかけ」の銘柄が同じ0.0に潰れ、4-bが情報を失う
    #      (実測でθ未満の62%がこの0.0に集中していた)。
    # 2・3が無いと、測れないものに低スコアを割り当てることになる。
    # データが足りない・状態が違うことと、遅すぎることは別である。
    try:
        s200 = sma(c, cfg.d1_ma_long)
        measurable = len(c) - SMA200_WARMUP
        in_stage2 = c.iloc[-1] > s200.iloc[-1]
        if measurable >= D4B_MAX_DAYS and in_stage2:
            above = (c > s200).astype(int)
            days_since_cross = 0
            for i in range(1, D4B_MAX_DAYS + 1):
                if above.iloc[-i] == 1:
                    days_since_cross += 1
                else:
                    break
            p["4-b"] = peak_score(days_since_cross, D4B_LO_DAYS,
                                  D4B_IDEAL_DAYS, D4B_MAX_DAYS)
    except Exception:
        pass

    # 4-c: 小次郎ステージの滞在日数(両側評価)
    days = kojiro_stage_days(c, cfg.d1_stage_short, cfg.d1_stage_mid, cfg.d1_stage_long)
    if days:
        p["4-c"] = peak_score(days, 1, cfg.d4_stage_ideal_days, cfg.d4_stage_max_days)

    # 4-d: ベースカウント(1st/2nd=1.0, 3rd=0.5, 4th以降=0.2)
    # 200日線上での「ベース回数」を、週足スイング高値の更新回数で近似する
    try:
        w = to_weekly(daily)
        sw = alternate_swings(find_swings(w, cfg.fractal_n_weekly, lookback=60))
        highs = [s["price"] for s in sw if s["kind"] == "H"]
        # 高値を切り上げた回数をベース回数とみなす
        count = 1
        for i in range(1, len(highs)):
            if highs[i] > highs[i - 1]:
                count += 1
        p["4-d"] = 1.0 if count <= 2 else (0.5 if count == 3 else 0.2)
    except Exception:
        pass

    # ④は4項目(4-a〜4-d)の平均として定義される。一部が欠けたまま平均を取ると
    # 銘柄によって「4項目平均」と「3項目平均」が混在し、④の値が銘柄間で
    # 比較できなくなる(実測で4-a欠損18.6%)。
    # ベースが検出できない銘柄は「セットアップが無い」のであり、
    # 「セットアップの成熟度が低い」のとは意味が違う。低い値を割り当てるのでは
    # なくNone(測定不能)を返し、カバレッジ判定では未充足として扱わせる。
    if len(p) < len(D4_REQUIRED_PARTS):
        lacking = [k for k in D4_REQUIRED_PARTS if k not in p]
        return {"score": None, "parts": p,
                "reason": f"測定不能(欠落: {','.join(lacking)})",
                "meta": {"base_weeks": base.get("weeks") if base else None}}

    score = float(np.mean(list(p.values())))
    return {"score": score, "parts": p, "reason": "",
            "meta": {"base_weeks": base.get("weeks") if base else None}}


# ============================================================
# ⑤ ボラティリティ収縮 — 価格変動の二次モーメント
#    証拠: 階層D。5次元中で最も投機的。VCPの査読検証は世界のどこにも存在しない
#    ハードフロア: なし(収縮していないこと自体は失格要件ではない)
#    ★二値化禁止(§3-⑤)。連続スコアとして測定する
# ============================================================

def dim5_volatility(daily: pd.DataFrame, cfg, base: dict | None) -> dict:
    c = daily["Close"].dropna()
    if len(c) < cfg.d5_bb_compare + 10:
        return {"score": None, "parts": {}, "reason": "履歴不足"}

    atr = atr_wilder(daily["High"], daily["Low"], daily["Close"], cfg.d5_atr_period)
    if pd.isna(atr.iloc[-1]) or float(atr.iloc[-1]) <= 0:
        return {"score": None, "parts": {}, "reason": "ATR算出不可"}
    atr_now = float(atr.iloc[-1])

    p = {}

    # 5-a: (ATR/株価)の時系列トレンド。低下が望ましい
    try:
        ratio = (atr / c).dropna()
        sl = slope_normalized(ratio, min(cfg.d5_atr_trend_window, len(ratio)))
        if sl is not None:
            p["5-a"] = sign_score(-sl, 0.005)
    except Exception:
        pass

    # 5-b: ベース内の各押し目の深さが縮小しているか(VCPの本質・§3-⑤)
    p5b = _contraction_score(daily, cfg, atr_now)
    if p5b is not None:
        p["5-b"] = p5b

    # 5-c: ボリンジャーバンド幅の縮小度
    try:
        m = sma(c, cfg.d5_bb_period)
        sd = c.rolling(cfg.d5_bb_period).std()
        bw = ((sd * 4) / m).dropna()     # バンド幅(±2σ)を価格で正規化
        if len(bw) >= cfg.d5_bb_compare:
            now = float(bw.iloc[-1])
            base_bw = float(bw.iloc[-cfg.d5_bb_compare:].median())
            if base_bw > 0:
                # 現在幅が過去中央値の何倍か。狭いほど高スコア
                p["5-c"] = clamp01(1.0 - (now / base_bw - 0.4) / 1.0)
    except Exception:
        pass

    # 5-d: 3本MAの相互間隔の縮小
    # ★検証で判明した性質(2026-08): 5-a/5-c/5-d は「直近の状態」を測るのに対し、
    # 5-b のみが「収縮の順序」を測る。深い押しの直後は戻りでMA間隔が急収縮するため、
    # 収縮列が拡大しているパターンでも 5-d が高く出ることがある。
    # 仕様書§3-⑤が 5-b を「VCPの本質」と特記しているのはこのため。
    # 単純平均のまま(仕様書§3.0の規定)にしておき、Level 2の増分検証で
    # 各成分の寄与を測ってから取捨する。ここでは重み付けによる介入をしない。
    try:
        s1 = sma(c, cfg.d1_stage_short)
        s2 = sma(c, cfg.d1_stage_mid)
        s3 = sma(c, cfg.d1_stage_long)
        spread = ((s1 - s3).abs() + (s2 - s3).abs()) / c
        sl = slope_normalized(spread.dropna(), min(60, len(spread.dropna())))
        if sl is not None:
            p["5-d"] = sign_score(-sl, 0.01)
    except Exception:
        pass

    score = float(np.mean(list(p.values()))) if p else None
    return {"score": score, "parts": p, "reason": ""}


def _contraction_score(daily: pd.DataFrame, cfg, atr_now: float) -> float | None:
    """5-b: 収縮列の測定(§3-⑤ 決定論的手順)。

    1. fractal(n)でスイング検出
    2. スイングハイ→次のスイングローの下落率リストを生成
    3. 各ciをATRで正規化(価格帯バイアス除去)
    4. 単調縮小度・最終収縮のタイトさ・収縮回数充足度を連続値で算出

    ★合否判定ではなく測定。二値化してはならない。
    """
    try:
        sw = alternate_swings(find_swings(daily, cfg.fractal_n, lookback=150))
        if len(sw) < 3:
            return None
        # ベース起点(直近の最高スイング高値)以降に限定する
        highs = [x for x in sw if x["kind"] == "H"]
        if not highs:
            return None
        top = max(highs, key=lambda x: x["price"])
        sw_base = [x for x in sw if x["idx"] >= top["idx"]]
        if len(sw_base) < 3:
            return None

        cons = []
        for i in range(len(sw_base) - 1):
            a, b = sw_base[i], sw_base[i + 1]
            if a["kind"] == "H" and b["kind"] == "L" and a["price"] > 0:
                drop = a["price"] - b["price"]
                if drop > 0 and atr_now > 0:
                    cons.append(drop / atr_now)      # ATR正規化
        if len(cons) < 2:
            return None

        # (i) 単調縮小度: 隣接比がどの程度1未満に揃っているか
        ratios = [cons[i + 1] / cons[i] for i in range(len(cons) - 1) if cons[i] > 0]
        if not ratios:
            return None
        mono = float(np.mean([clamp01((1.0 - r) / (1.0 - cfg.d5_shrink_ratio))
                              for r in ratios]))

        # (ii) 最終収縮のタイトさ: c[-1]の絶対水準(ATR単位)
        tight = clamp01(1.0 - cons[-1] / max(1e-9, cfg.d5_final_tight_atr))

        # (iii) 収縮回数の充足度
        n = len(cons)
        cnt = peak_score(n, cfg.d5_contract_min - 1,
                         (cfg.d5_contract_min + cfg.d5_contract_max) / 2.0,
                         cfg.d5_contract_max + 2)

        return clamp01(cfg.d5_w_monotonic * mono +
                       cfg.d5_w_tightness * tight +
                       cfg.d5_w_count * cnt)
    except Exception:
        return None
