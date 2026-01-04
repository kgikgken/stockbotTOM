from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from utils.util import clamp


def compute_entry_zone(df: pd.DataFrame, setup_type: str, setup_meta: Dict) -> Dict:
    """
    Entry中心と帯、GU、乖離を計算して行動を決める
    行動：即エントリー可 / 指値待ち / 今日は監視
    """
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    ma20 = df["ma20"].astype(float)
    ma50 = df["ma50"].astype(float)

    c = float(close.iloc[-1])
    o = float(open_.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else c

    atr = float(df["atr14"].iloc[-1]) if "atr14" in df.columns and np.isfinite(df["atr14"].iloc[-1]) else float("nan")
    if not (np.isfinite(atr) and atr > 0):
        atr = max(c * 0.01, 1.0)

    # GU判定（必須）
    gu = bool(np.isfinite(o) and np.isfinite(prev_close) and (o > prev_close + 1.0 * atr))

    # Center
    if setup_type == "A1":
        center = float(ma20.iloc[-1])
        band = 0.50 * atr
    elif setup_type == "A2":
        center = float(ma50.iloc[-1])
        band = 0.50 * atr
    elif setup_type == "B":
        center = float(setup_meta.get("hh20", c))
        band = 0.30 * atr
    else:
        center = c
        band = 0.50 * atr

    if not np.isfinite(center) or center <= 0:
        center = c

    low = center - band
    high = center + band

    # 乖離（追いかけ禁止）
    dist_atr = abs(c - center) / atr if atr > 0 else 999.0

    # 行動（裁量ゼロ）
    # - GUなら監視
    # - 乖離>0.8なら監視
    # - 帯の中なら「即エントリー可」
    # - 帯の外だが0.8以内なら「指値待ち」
    action = "指値待ち"
    if gu or dist_atr > 0.80:
        action = "今日は監視"
    else:
        if low <= c <= high:
            action = "即エントリー可"
        else:
            action = "指値待ち"

    return {
        "center": float(round(center, 1)),
        "low": float(round(low, 1)),
        "high": float(round(high, 1)),
        "atr": float(round(atr, 1)),
        "gu": bool(gu),
        "dist_atr": float(dist_atr),
        "action": action,
        "price_now": float(round(c, 1)),
    }