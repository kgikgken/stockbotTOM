"""v7.1 共通計算基盤 — テクニカル指標とスイング検出器。

仕様書§6.3: 検出器はfractal固定。ZigZagは確定ラグが不定(repaint)のため不採用。
「3,000超の銘柄を日次で回すスクリーニングにおいて、ラグが銘柄ごとに異なり
事前に分からない性質は許容できない」
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ============ 移動平均・ATR ============

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()


def slope_normalized(s: pd.Series, window: int) -> float | None:
    """直近windowバーの回帰スロープを、系列の水準で正規化して返す。

    生のスロープは価格水準に依存する(1000円の株と100円の株で意味が変わる)ため、
    平均値で割って「1バーあたり何%動くか」に揃える。
    """
    try:
        y = s.dropna().iloc[-window:]
        if len(y) < window or float(y.mean()) == 0:
            return None
        x = np.arange(len(y), dtype=float)
        b = float(np.polyfit(x, y.values.astype(float), 1)[0])
        return b / float(y.mean())
    except Exception:
        return None


# ============ 週足変換 ============

def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """日足→週足(金曜終値ベース)。"""
    w = pd.DataFrame({
        "Open": daily["Open"].resample("W-FRI").first(),
        "High": daily["High"].resample("W-FRI").max(),
        "Low": daily["Low"].resample("W-FRI").min(),
        "Close": daily["Close"].resample("W-FRI").last(),
        "Volume": daily["Volume"].resample("W-FRI").sum(),
    })
    return w.dropna(subset=["Close"])


# ============ fractal スイング検出(§6.3) ============

def find_swings(df: pd.DataFrame, n: int, lookback: int | None = None) -> list:
    """前後n本より高い(安い)バーをスイング点とする。

    確定ラグは常にn本で固定。これが仕様書がfractalを選んだ理由。
    戻り値: [{"idx": 位置, "price": 価格, "kind": "H"/"L", "date": 日付}, ...] 時系列順
    """
    d = df.iloc[-lookback:] if (lookback and len(df) > lookback) else df
    h, l = d["High"].values, d["Low"].values
    out = []
    for i in range(n, len(d) - n):
        wh, wl = h[i - n:i + n + 1], l[i - n:i + n + 1]
        if h[i] == wh.max() and wh.argmax() == n:
            out.append({"idx": i, "price": float(h[i]), "kind": "H", "date": d.index[i]})
        elif l[i] == wl.min() and wl.argmin() == n:
            out.append({"idx": i, "price": float(l[i]), "kind": "L", "date": d.index[i]})
    return out


def alternate_swings(swings: list) -> list:
    """H/Lが交互になるよう整理する(連続する同種は極値のみ残す)。"""
    if not swings:
        return []
    out = [swings[0]]
    for s in swings[1:]:
        if s["kind"] == out[-1]["kind"]:
            if (s["kind"] == "H" and s["price"] > out[-1]["price"]) or \
               (s["kind"] == "L" and s["price"] < out[-1]["price"]):
                out[-1] = s
        else:
            out.append(s)
    return out


# ============ 正規化ヘルパー ============

def clamp01(x: float) -> float:
    if x is None or not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def linear_score(value: float | None, lo: float, hi: float) -> float:
    """lo以下で0.0、hi以上で1.0へ線形写像。"""
    if value is None or not math.isfinite(value) or hi == lo:
        return 0.0
    return clamp01((value - lo) / (hi - lo))


def peak_score(value: float | None, lo: float, ideal: float, hi: float) -> float:
    """両側評価(上に凸)。仕様書§3-④「短すぎるも長すぎるも低スコア」。

    lo以下=0.0、ideal=1.0、hi以上=0.0 の山形。
    """
    if value is None or not math.isfinite(value):
        return 0.0
    v = float(value)
    if v <= lo or v >= hi:
        return 0.0
    if v <= ideal:
        return clamp01((v - lo) / (ideal - lo)) if ideal > lo else 1.0
    return clamp01((hi - v) / (hi - ideal)) if hi > ideal else 1.0


def sign_score(value: float | None, scale: float) -> float:
    """符号付きの値を0.0〜1.0へ連続写像。0で0.5、+scaleで1.0、-scaleで0.0。

    仕様書§3-①1-b「正で1.0、0付近0.5、負0.0に連続写像」の実装。
    """
    if value is None or not math.isfinite(value) or scale <= 0:
        return 0.5
    return clamp01(0.5 + float(value) / (2.0 * scale))


# ============ 小次郎ステージ(1-h・4-c で共用) ============

def kojiro_stage(close: pd.Series, s: int, m: int, l: int) -> int:
    """3本MAの並び順による6ステージ分類。0は縮退(判定不能)。"""
    try:
        ss, sm, sl = sma(close, s), sma(close, m), sma(close, l)
        if any(pd.isna(x.iloc[-1]) for x in (ss, sm, sl)):
            return 0
        a, b, c = float(ss.iloc[-1]), float(sm.iloc[-1]), float(sl.iloc[-1])
    except Exception:
        return 0
    if a > b > c: return 1
    if b > a > c: return 2
    if b > c > a: return 3
    if c > b > a: return 4
    if c > a > b: return 5
    if a > c > b: return 6
    return 0


def kojiro_stage_days(close: pd.Series, s: int, m: int, l: int, cap: int = 90) -> int:
    """現ステージの連続滞在日数。"""
    cur = kojiro_stage(close, s, m, l)
    if cur == 0:
        return 0
    days = 1
    for i in range(2, min(cap + 1, len(close) + 1)):
        if kojiro_stage(close.iloc[:-(i - 1)], s, m, l) != cur:
            break
        days += 1
    return days
