from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from utils.util import clamp


def detect_setup(df: pd.DataFrame) -> Tuple[str, Dict]:
    """
    Setupを A1 / A2 / B に分離
      A1: MA20付近の浅押し（回転・速度寄り）
      A2: MA50寄りの深押し（時間はかかるが伸びやすい）
      B : ブレイク初動（追いかけ禁止、押し待ち前提）

    戻り: (setup_type, meta)
      setup_type: "A1" / "A2" / "B" / "-"
    """
    if df is None or len(df) < 80:
        return "-", {"reason": "データ不足"}

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    ma20 = df["ma20"].astype(float)
    ma50 = df["ma50"].astype(float)
    rsi = df["rsi14"].astype(float)

    c = float(close.iloc[-1])
    m20 = float(ma20.iloc[-1]) if np.isfinite(ma20.iloc[-1]) else float("nan")
    m50 = float(ma50.iloc[-1]) if np.isfinite(ma50.iloc[-1]) else float("nan")
    r = float(rsi.iloc[-1]) if np.isfinite(rsi.iloc[-1]) else float("nan")

    atr = float(df["atr14"].iloc[-1]) if "atr14" in df.columns and np.isfinite(df["atr14"].iloc[-1]) else float("nan")
    if not (np.isfinite(atr) and atr > 0):
        return "-", {"reason": "ATR不足"}

    # Trend前提（順張りのみ）
    trend_ok = bool(np.isfinite(c) and np.isfinite(m20) and np.isfinite(m50) and c > m20 > m50)

    # MA20 slope（5日差分）
    slope20 = float((ma20.iloc[-1] / ma20.iloc[-6] - 1.0)) if len(ma20) >= 6 and np.isfinite(ma20.iloc[-6]) else 0.0
    slope_ok = bool(slope20 > 0)

    # “完璧すぎない”＝過熱排除
    rsi_ok = bool(np.isfinite(r) and 40.0 <= r <= 62.0)

    # 押しの深さ（MA20からの距離 / ATR）
    pull20 = abs(c - m20) / atr
    pull50 = abs(c - m50) / atr

    # ブレイク判定（HH20 + 出来高増）
    hh20 = float(close.rolling(20).max().iloc[-2]) if len(close) >= 22 else float("nan")
    vol_now = float(vol.iloc[-1]) if np.isfinite(vol.iloc[-1]) else float("nan")
    vol_ma20 = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float("nan")
    breakout = bool(np.isfinite(hh20) and c > hh20 and np.isfinite(vol_now) and np.isfinite(vol_ma20) and vol_ma20 > 0 and vol_now >= 1.5 * vol_ma20)

    if breakout and trend_ok and slope_ok:
        return "B", {"hh20": hh20, "pull20": pull20, "pull50": pull50}

    # A系
    if not (trend_ok and slope_ok and rsi_ok):
        return "-", {"reason": "形不一致"}

    # A1: MA20付近の浅押し（abs(C-MA20) <= 0.8ATR かつ MA20上〜軽い下）
    # A2: MA50寄り（MA20から離れてるがMA50近い）
    if pull20 <= 0.80:
        return "A1", {"pull20": pull20, "pull50": pull50}
    if pull50 <= 0.90 and c >= m50:
        return "A2", {"pull20": pull20, "pull50": pull50}

    return "-", {"reason": "押し条件不足"}