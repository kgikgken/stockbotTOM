from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils.util import sma, rsi14, atr14, adv20, atr_pct_last, safe_float, clamp

@dataclass
class SetupInfo:
    setup: str
    tier: int
    entry_low: float
    entry_high: float
    sl: float
    tp2: float
    tp1: float
    rr: float
    expected_days: float
    rday: float
    trend_strength: float
    pullback_quality: float
    gu: bool
    breakout_line: Optional[float] = None

def _trend_strength(c: pd.Series, ma20: pd.Series, ma50: pd.Series) -> float:
    c_last = safe_float(c.iloc[-1], np.nan)
    m20 = safe_float(ma20.iloc[-1], np.nan)
    m50 = safe_float(ma50.iloc[-1], np.nan)
    if not (np.isfinite(c_last) and np.isfinite(m20) and np.isfinite(m50)):
        return 0.0

    slope = safe_float(ma20.pct_change(fill_method=None).iloc[-1], 0.0)

    if c_last > m20 > m50:
        pos = 1.0
    elif c_last > m20:
        pos = 0.7
    elif m20 > m50:
        pos = 0.5
    else:
        pos = 0.3

    ang = clamp((slope / 0.004), -1.0, 1.5)
    base = 0.85 + 0.20 * pos + 0.10 * ang
    return float(clamp(base, 0.80, 1.20))

def _pullback_quality(df: pd.DataFrame, setup: str, ma25: float, ma50: float, atr: float) -> float:
    """押し目の健全性（0.4〜1.4）を返す。

    v2.3+（思想更新）：
    - 押し目は「高値更新→3〜7本の調整→MA25〜MA50付近で下げ止まり」の再現性を最優先
    - ただし過剰にクリーンにし過ぎない（許容幅を残す）
    """
    if df is None or len(df) < 60 or atr <= 0:
        return 1.0

    c = float(df["Close"].iloc[-1])
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    v = df["Volume"].astype(float)

    # 直近20本の最高値（直近バーは除く）
    recent20 = h.iloc[-21:-1]
    if recent20.empty:
        return 1.0
    hi20 = float(recent20.max())
    # 最高値の出現からの本数
    idx_hi = int(recent20.values.argmax())  # 0..19
    bars_since_high = 20 - 1 - idx_hi  # 0..19

    # 押しの深さ（高値→現在）をATRで正規化
    depth_atr = max(0.0, (hi20 - c) / atr)

    # MA帯への接近（A1はMA25〜MA50、A2/Bは弱め）
    if setup in ("A1", "A1-Strong"):
        band_center = (ma25 + ma50) / 2.0
        dist_band = abs(c - band_center) / atr
        ma_score = 1.0 - min(1.0, dist_band / 2.0)  # 0..1
    else:
        ma_score = 0.6

    # 調整本数：理想3〜7本（外れても即死させず点数で扱う）
    if 3 <= bars_since_high <= 7:
        bar_score = 1.0
    elif 2 <= bars_since_high <= 10:
        bar_score = 0.85
    else:
        bar_score = 0.65

    # 出来高収縮（直近3本平均がその前3本平均より小さいほど良い）
    vol_score = 0.75
    if len(v) >= 8:
        v1 = float(v.iloc[-3:].mean())
        v0 = float(v.iloc[-6:-3].mean())
        if v0 > 0:
            ratio = v1 / v0
            # 0.6以下で満点、1.2以上で最低
            vol_score = 1.0 - min(1.0, max(0.0, (ratio - 0.6) / 0.6)) * 0.4  # 0.6..1.0

    # 高値更新後の押し目として「深すぎない」こと（ただし強すぎる縛りは避ける）
    # 0.2〜1.3ATRの範囲を好む
    if depth_atr <= 0.2:
        depth_score = 0.8
    elif depth_atr <= 1.3:
        depth_score = 1.0
    elif depth_atr <= 2.0:
        depth_score = 0.8
    else:
        depth_score = 0.6

    # 合成（0.4〜1.4程度）
    q = 0.35 * ma_score + 0.25 * bar_score + 0.25 * vol_score + 0.15 * depth_score
    return float(max(0.4, min(1.4, 0.4 + q)))


def detect_setup(df: pd.DataFrame) -> Optional[SetupInfo]:
    """セットアップ判定（A1 / A1-Strong / A2 / B）

    方向性（最新版）：
    ① 押し目トレンドフォロー（主軸：再現性×資金効率）
    ② 初動ブレイク（短期最大効率：ただしロット抑制の前提）
    ③ 需給歪み（刃：限定稼働。ここでは“候補生成”のみ）
    """
    if df is None or len(df) < 80:
        return None

    # 正規化（欠損除去）
    df = df.copy()
    df = df.dropna(subset=["Open","High","Low","Close","Volume"])
    if len(df) < 80:
        return None

    c = float(df["Close"].iloc[-1])
    o = float(df["Open"].iloc[-1])
    h = float(df["High"].iloc[-1])
    l = float(df["Low"].iloc[-1])
    v = float(df["Volume"].iloc[-1])

    ma25 = float(sma(df["Close"], 25).iloc[-1])
    ma50 = float(sma(df["Close"], 50).iloc[-1])
    ma75 = float(sma(df["Close"], 75).iloc[-1])
    rsi = float(rsi14(df["Close"]).iloc[-1])
    atr = float(atr14(df["High"], df["Low"], df["Close"]).iloc[-1])
    if atr <= 0 or math.isnan(atr):
        return None

    # 出来高（簡易）
    vol20 = float(df["Volume"].astype(float).rolling(20).mean().iloc[-1])
    vol_ratio = (v / vol20) if vol20 > 0 else 1.0

    # トレンド強度（0.6〜1.4）
    trend_strength = _trend_strength(c, ma25, ma50, ma75)

    # --- A1 / A1-Strong: 押し目トレンドフォロー ---
    trend_up = (ma25 > ma50 > ma75) and (sma(df["Close"], 25).iloc[-1] - sma(df["Close"], 25).iloc[-6] > 0)

    # 直近高値更新の有無（20本）
    hi20_prev = float(df["High"].astype(float).iloc[-21:-1].max())
    # 高値更新後の押し目（現在が高値からある程度下）
    pullback_from_high = (hi20_prev - c) / atr

    a1_candidate = trend_up and (hi20_prev > 0) and (pullback_from_high >= 0.2) and (c >= ma50 * 0.98)

    if a1_candidate:
        pb_q = _pullback_quality(df, "A1", ma25, ma50, atr)

        # A1-Strong（より厳格）
        # - RSI高め、押しの深さが浅め、MA25付近で止まりやすい
        strong = (
            (rsi >= 55.0) and
            (pullback_from_high <= 1.3) and
            (c >= ma25 * 0.97) and
            (pb_q >= 1.0)
        )
        setup = "A1-Strong" if strong else "A1"

        # エントリー帯（A1はMA25〜MA50中心に寄せる）
        entry_low = min(ma25, ma50)
        entry_high = max(ma25, ma50)
        # ただし帯が広すぎる場合は、ATRで最大幅を制限（中央表示のための安定化）
        max_width = 0.9 * atr
        if (entry_high - entry_low) > max_width:
            mid = (entry_high + entry_low) / 2.0
            entry_low = mid - max_width/2.0
            entry_high = mid + max_width/2.0

        return SetupInfo(
            setup=setup,
            entry_low=float(entry_low),
            entry_high=float(entry_high),
            breakout_line=None,
            trend_strength=float(trend_strength),
            pullback_quality=float(pb_q),
            gu_flag=False,
        )

    # --- A2: 初動ブレイク（出来高急増のブレイク → 翌日押し待ち） ---
    # レンジ性（直近60本の値幅が相対的に小さい）
    rng60 = float(df["High"].astype(float).rolling(60).max().iloc[-1] - df["Low"].astype(float).rolling(60).min().iloc[-1])
    rng_ratio = rng60 / c if c > 0 else 1.0
    range_like = (rng_ratio < 0.35) and (ma25 <= ma50 * 1.03)

    hi20_ex = float(df["High"].astype(float).iloc[-21:-1].max())
    breakout = c >= hi20_ex * 1.005

    if range_like and breakout and (vol_ratio >= 1.5):
        pb_q = _pullback_quality(df, "A2", ma25, ma50, atr)
        setup = "A2"
        breakout_line = hi20_ex

        # 押し待ちの指値帯（ブレイクライン〜-0.5ATR）
        entry_high = breakout_line
        entry_low = breakout_line - 0.5 * atr
        if entry_low < 0:
            entry_low = breakout_line * 0.98

        return SetupInfo(
            setup=setup,
            entry_low=float(entry_low),
            entry_high=float(entry_high),
            breakout_line=float(breakout_line),
            trend_strength=float(trend_strength),
            pullback_quality=float(pb_q),
            gu_flag=False,
        )

    # --- B: 需給歪み（イベント・指数ズレなど） ---
    # “過剰売り→需給修復”の典型：大陰線の翌に包み足/大陽線（最終バー時点で成立しているもの）
    if len(df) >= 3:
        o1 = float(df["Open"].iloc[-2])
        c1 = float(df["Close"].iloc[-2])
        h1 = float(df["High"].iloc[-2])
        l1 = float(df["Low"].iloc[-2])
        v1 = float(df["Volume"].iloc[-2])

        red_big = (c1 < o1) and ((o1 - c1) / atr >= 1.2)
        green_engulf = (c > o) and (c >= h1) and (vol_ratio >= 1.5)

        if red_big and green_engulf:
            pb_q = 0.9  # 歪みは押し目品質より「需給サイン」を優先（ここでは固定）
            setup = "B"

            # エントリーは反転の押し（終値近辺〜-0.4ATR）
            entry_high = c
            entry_low = c - 0.4 * atr
            if entry_low < 0:
                entry_low = c * 0.98

            return SetupInfo(
                setup=setup,
                entry_low=float(entry_low),
                entry_high=float(entry_high),
                breakout_line=None,
                trend_strength=float(trend_strength),
                pullback_quality=float(pb_q),
                gu_flag=False,
            )

    return None


def entry_band(df: pd.DataFrame, setup: str) -> Tuple[float, float, float, Optional[float]]:
    c = df["Close"].astype(float)
    ma20 = sma(c, 20)
    ma50 = sma(c, 50)
    a = atr14(df)
    atr = safe_float(a.iloc[-1], np.nan)
    if not np.isfinite(atr) or atr <= 0:
        atr = safe_float(c.iloc[-1], 0.0) * 0.015

    m20 = safe_float(ma20.iloc[-1], safe_float(c.iloc[-1], 0.0))
    m50 = safe_float(ma50.iloc[-1], safe_float(c.iloc[-1], 0.0))
    breakout_line = None

    if setup in ("A1-Strong", "A1"):
        k = 0.4 if setup == "A1-Strong" else 0.5
        lo = m20 - k * atr
        hi = m20 + k * atr
    elif setup == "A2":
        center = (m20 + m50) / 2.0
        lo = center - 0.6 * atr
        hi = center + 0.6 * atr
    elif setup == "B":
        hh20 = float(c.tail(21).max())
        breakout_line = hh20
        lo = hh20 - 0.3 * atr
        hi = hh20 + 0.3 * atr
    else:
        lo = hi = safe_float(c.iloc[-1], 0.0)

    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    lo = max(lo, 1.0)
    hi = max(hi, lo + 0.1)
    return lo, hi, float(atr), (float(breakout_line) if breakout_line is not None else None)

def gu_flag(df: pd.DataFrame, atr: float) -> bool:
    if df is None or df.empty or len(df) < 2:
        return False
    o = safe_float(df["Open"].iloc[-1], np.nan)
    pc = safe_float(df["Close"].iloc[-2], np.nan)
    if not (np.isfinite(o) and np.isfinite(pc) and np.isfinite(atr) and atr > 0):
        return False
    return bool(o > pc + 1.0 * atr)

def liquidity_filters(df: pd.DataFrame, price_min=200.0, price_max=15000.0, adv_min=200e6, atrpct_min=1.5):
    price = safe_float(df["Close"].iloc[-1], np.nan)
    adv = adv20(df)
    atrp = atr_pct_last(df)
    ok = True
    if not (np.isfinite(price) and price_min <= price <= price_max):
        ok = False
    if not (np.isfinite(adv) and adv >= adv_min):
        ok = False
    if not (np.isfinite(atrp) and atrp >= atrpct_min):
        ok = False
    return ok, float(price), float(adv), float(atrp)

def structure_sl_tp(df: pd.DataFrame, entry_mid: float, atr: float, macro_on: bool):
    lookback = 12
    low = float(df["Low"].astype(float).tail(lookback).min())
    sl1 = entry_mid - 1.2 * atr
    sl = min(sl1, low - 0.1 * atr)

    sl = min(sl, entry_mid * (1.0 - 0.02))
    sl = max(sl, entry_mid * (1.0 - 0.10))

    risk = max(entry_mid - sl, 0.01)

    rr_target = 2.6
    hi_window = 60 if len(df) >= 60 else len(df)
    high_60 = float(df["Close"].astype(float).tail(hi_window).max())
    tp2_raw = entry_mid + rr_target * risk
    tp2 = min(tp2_raw, high_60 * 0.995, entry_mid * (1.0 + 0.35))

    tp2_min = entry_mid + 2.0 * risk
    if tp2 < tp2_min:
        tp2 = tp2_min

    tp2_max = entry_mid + 3.5 * risk
    tp2 = min(tp2, tp2_max)

    if macro_on:
        tp2 = entry_mid + (tp2 - entry_mid) * 0.85

    tp1 = entry_mid + 1.5 * risk
    rr = (tp2 - entry_mid) / risk
    exp_days = (tp2 - entry_mid) / max(atr, 1e-6)

    return float(sl), float(tp1), float(tp2), float(rr), float(exp_days)

def build_setup_info(df: pd.DataFrame, macro_on: bool) -> SetupInfo:
    setup, tier = detect_setup(df)
    lo, hi, atr, breakout_line = entry_band(df, setup)
    entry_mid = (lo + hi) / 2.0

    gu = gu_flag(df, atr)
    sl, tp1, tp2, rr, exp_days = structure_sl_tp(df, entry_mid, atr, macro_on=macro_on)
    rday = rr / max(exp_days, 1e-6)

    c = df["Close"].astype(float)
    ma20 = sma(c, 20)
    ma50 = sma(c, 50)
    ts = _trend_strength(c, ma20, ma50)
    pq = _pullback_quality(c, ma20, ma50, atr, setup)

    return SetupInfo(
        setup=setup,
        tier=int(tier),
        entry_low=float(lo),
        entry_high=float(hi),
        sl=float(sl),
        tp2=float(tp2),
        tp1=float(tp1),
        rr=float(rr),
        expected_days=float(exp_days),
        rday=float(rday),
        trend_strength=float(ts),
        pullback_quality=float(pq),
        gu=bool(gu),
        breakout_line=breakout_line,
    )
