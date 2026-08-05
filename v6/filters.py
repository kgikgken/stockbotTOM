"""v6 Layer 1〜3 — 週足ステージ / トレンドテンプレート / 日足ステージ。

各レイヤーはAND条件で、スコアによる救済は行わない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============ 共通ユーティリティ ============

def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """日足→週足。金曜終値ベース(ワインスタイン原典に忠実)。

    最終週が未完成(週の途中)の場合もそのまま最終行として扱う。
    30週線の計算には十分な週数が必要なため、呼び出し側で長さを確認すること。
    """
    o = daily["Open"].resample("W-FRI").first()
    h = daily["High"].resample("W-FRI").max()
    l = daily["Low"].resample("W-FRI").min()
    c = daily["Close"].resample("W-FRI").last()
    v = daily["Volume"].resample("W-FRI").sum()
    w = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
    return w.dropna(subset=["Close"])


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()


# ============ Layer 1: 週足ステージ(ワインスタイン) ============

STAGE_W_BASE, STAGE_W_UP, STAGE_W_TOP, STAGE_W_DOWN = 1, 2, 3, 4


def weekly_stage(daily: pd.DataFrame, cfg) -> dict | None:
    """30週線に対する位置と傾きで4ステージに分類する。

    ステージ2(上昇期)のみが通過。ステージ1(底固め)は「そのうち上がるかも」の
    期待値取引になるため、初動を取り逃してもよいと割り切る設計(仕様書§3)。
    """
    w = to_weekly(daily)
    n = cfg.weekly_ma
    if len(w) < n + cfg.weekly_slope_weeks:
        return None
    ma = sma(w["Close"], n)
    if pd.isna(ma.iloc[-1]) or pd.isna(ma.iloc[-1 - cfg.weekly_slope_weeks]):
        return None

    ma_now = float(ma.iloc[-1])
    ma_past = float(ma.iloc[-1 - cfg.weekly_slope_weeks])
    if ma_past <= 0:
        return None
    slope = (ma_now - ma_past) / ma_past
    close_w = float(w["Close"].iloc[-1])
    above = close_w > ma_now
    th = cfg.weekly_slope_th

    if above and slope > th:
        stage = STAGE_W_UP
    elif above and slope <= th:
        stage = STAGE_W_TOP
    elif (not above) and slope < -th:
        stage = STAGE_W_DOWN
    else:
        stage = STAGE_W_BASE

    return {"stage_w": stage, "ma30w": ma_now, "slope_w": slope, "close_w": close_w,
            "weeks": len(w)}


def rs_vs_index(daily: pd.DataFrame, index_daily: pd.DataFrame, weeks: int) -> float | None:
    """26週リターンの対指数超過。ワインスタインが最重視する「指数に勝っているか」。

    市場全体の上昇に乗っているだけの銘柄をここで落とす。
    """
    bars = weeks * 5
    try:
        c = daily["Close"].dropna()
        ic = index_daily["Close"].dropna()
        if len(c) < bars + 1 or len(ic) < bars + 1:
            return None
        r_stock = float(c.iloc[-1] / c.iloc[-1 - bars] - 1)
        r_index = float(ic.iloc[-1] / ic.iloc[-1 - bars] - 1)
        return r_stock - r_index
    except Exception:
        return None


# ============ Layer 2: トレンドテンプレート(ミネルヴィニ) ============

def trend_template(daily: pd.DataFrame, cfg) -> dict | None:
    """ミネルヴィニのトレンドテンプレート8条件。RSレーティング以外の7条件を判定する。

    RSレーティング(条件8)はユニバース全体の分布が必要なため、別途 rs_rating() で計算し
    呼び出し側で合成する。
    """
    c = daily["Close"].dropna()
    if len(c) < 252:
        return None
    s50, s150, s200 = sma(c, 50), sma(c, 150), sma(c, 200)
    if any(pd.isna(x.iloc[-1]) for x in (s50, s150, s200)):
        return None
    if len(s200) <= cfg.tt_ma200_lookback or pd.isna(s200.iloc[-1 - cfg.tt_ma200_lookback]):
        return None

    close_now = float(c.iloc[-1])
    v50, v150, v200 = float(s50.iloc[-1]), float(s150.iloc[-1]), float(s200.iloc[-1])
    v200_past = float(s200.iloc[-1 - cfg.tt_ma200_lookback])
    lo52 = float(c.iloc[-252:].min())
    hi52 = float(c.iloc[-252:].max())

    conds = {
        "1_above_150_200": close_now > v150 and close_now > v200,
        "2_150_above_200": v150 > v200,
        "3_200_rising": v200 > v200_past,
        "4_50_above_rest": v50 > v150 and v50 > v200,
        "5_close_above_50": close_now > v50,
        "6_above_52w_low": close_now >= lo52 * cfg.tt_low_mult,
        "7_near_52w_high": close_now >= hi52 * cfg.tt_high_mult,
    }
    passed = all(conds.values())
    return {"tt_pass_7": passed, "tt_conds": conds,
            "sma50": v50, "sma150": v150, "sma200": v200,
            "low52": lo52, "high52": hi52,
            "pct_from_high": (close_now / hi52 - 1) if hi52 > 0 else None}


def rs_performance(daily: pd.DataFrame) -> float | None:
    """IBD方式の簡略版パフォーマンス値。直近四半期に2倍の重みを置く。

    perf = 0.4*ret63 + 0.2*ret126 + 0.2*ret189 + 0.2*ret252
    ユニバース内でパーセンタイル化することでRSレーティングになる。
    """
    try:
        c = daily["Close"].dropna()
        if len(c) < 253:
            return None
        p = float(c.iloc[-1])
        r63 = p / float(c.iloc[-64]) - 1
        r126 = p / float(c.iloc[-127]) - 1
        r189 = p / float(c.iloc[-190]) - 1
        r252 = p / float(c.iloc[-253]) - 1
        return 0.4 * r63 + 0.2 * r126 + 0.2 * r189 + 0.2 * r252
    except Exception:
        return None


def rs_ratings(perf_map: dict) -> dict:
    """パフォーマンス値の辞書 {ticker: perf} をパーセンタイル(0〜100)に変換する。"""
    items = [(k, v) for k, v in perf_map.items() if v is not None and np.isfinite(v)]
    if not items:
        return {}
    s = pd.Series({k: v for k, v in items})
    ranks = s.rank(pct=True) * 100.0
    return {k: float(v) for k, v in ranks.items()}


# ============ Layer 3: 日足ステージ(小次郎講師) ============

STAGE_D_NAMES = {
    1: "上昇期", 2: "上昇トレンド終わり", 3: "下降入り口",
    4: "下降期", 5: "下降終わり", 6: "上昇の始まり",
}


def _stage_from(s: float, m: float, l: float) -> int:
    if s > m > l: return 1
    if m > s > l: return 2
    if m > l > s: return 3
    if l > m > s: return 4
    if l > s > m: return 5
    if s > l > m: return 6
    return 0   # 同値などの縮退


def daily_stage(daily: pd.DataFrame, cfg) -> dict | None:
    """短期5日・中期20日・長期40日SMAの大小関係で6ステージに分類する。

    ステージ1のみ発注可、ステージ6は監視のみ(発注禁止)。滞在日数も記録する。
    """
    c = daily["Close"].dropna()
    need = cfg.stage_long + 60
    if len(c) < need:
        return None
    ss, sm, sl = sma(c, cfg.stage_short), sma(c, cfg.stage_mid), sma(c, cfg.stage_long)
    if any(pd.isna(x.iloc[-1]) for x in (ss, sm, sl)):
        return None

    stage = _stage_from(float(ss.iloc[-1]), float(sm.iloc[-1]), float(sl.iloc[-1]))

    # 現ステージの連続滞在日数(最大60日まで遡る)
    days = 1
    for i in range(2, min(61, len(c) + 1)):
        if any(pd.isna(x.iloc[-i]) for x in (ss, sm, sl)):
            break
        st = _stage_from(float(ss.iloc[-i]), float(sm.iloc[-i]), float(sl.iloc[-i]))
        if st != stage:
            break
        days += 1

    return {"stage_d": stage, "stage_d_name": STAGE_D_NAMES.get(stage, "不定"),
            "stage1_days": days if stage == 1 else 0,
            "stage_days": days,
            "sma_s": float(ss.iloc[-1]), "sma_m": float(sm.iloc[-1]), "sma_l": float(sl.iloc[-1])}
