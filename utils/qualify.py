from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return float("nan")


def _ma(series: pd.Series, w: int) -> float:
    if series is None or len(series) < w:
        return _last(series)
    return float(series.rolling(w).mean().iloc[-1])


def _pct(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b) and b != 0):
        return float("nan")
    return float((a / b - 1.0) * 100.0)


def runner_strength(hist: pd.DataFrame) -> Tuple[float, str]:
    """
    走行能力（0-100）とラベルを返す。
    A2_prebreak: 高値圏で押している（走る準備）
    A3_break    : 直近で高値更新〜ブレイク進行
    B           : 走行はあるが弱い
    C           : 走らない
    """
    if hist is None or len(hist) < 120:
        return 0.0, "C"

    df = hist.copy()
    close = df["Close"].astype(float)

    c = _last(close)
    ma20 = _ma(close, 20)
    ma50 = _ma(close, 50)
    ma120 = _ma(close, 120)

    if not (np.isfinite(c) and np.isfinite(ma20) and np.isfinite(ma50) and np.isfinite(ma120)):
        return 0.0, "C"

    hi60 = float(close.tail(60).max()) if len(close) >= 60 else float(close.max())
    off_high = _pct(c, hi60)  # 高値からの距離(%)

    ret5 = _pct(c, float(close.iloc[-6])) if len(close) >= 6 else float("nan")
    ret20 = _pct(c, float(close.iloc[-21])) if len(close) >= 21 else float("nan")

    r = close.pct_change(fill_method=None)
    vola20 = float(r.rolling(20).std().iloc[-1]) if len(r) >= 25 else float("nan")

    sc = 0.0

    # トレンド構造（最大40）
    if c > ma20 > ma50 > ma120:
        sc += 40
    elif c > ma20 > ma50:
        sc += 30
    elif c > ma20 and ma20 > ma50:
        sc += 22
    elif ma20 > ma50:
        sc += 14
    else:
        sc += 5

    # 高値近接（最大25）
    if np.isfinite(off_high):
        if off_high >= 0:
            sc += 25
        elif off_high >= -2.0:
            sc += 22
        elif off_high >= -5.0:
            sc += 16
        elif off_high >= -10.0:
            sc += 10
        else:
            sc += 4

    # 勢い（最大25）
    if np.isfinite(ret20):
        if ret20 >= 15:
            sc += 14
        elif ret20 >= 8:
            sc += 10
        elif ret20 >= 3:
            sc += 6
        else:
            sc += 2

    if np.isfinite(ret5):
        if ret5 >= 6:
            sc += 11
        elif ret5 >= 3:
            sc += 8
        elif ret5 >= 0:
            sc += 5
        else:
            sc += 0

    # ボラ過大は減点（最大-10）
    if np.isfinite(vola20):
        if vola20 > 0.05:
            sc -= 10
        elif vola20 > 0.035:
            sc -= 6
        elif vola20 > 0.028:
            sc -= 3

    sc = float(np.clip(sc, 0, 100))

    label = "C"
    if sc >= 78 and np.isfinite(off_high) and off_high >= -5.0:
        label = "A2_prebreak"
        if off_high >= -1.0 and np.isfinite(ret5) and ret5 > 0:
            label = "A3_break"
    elif sc >= 65:
        label = "B"
    else:
        label = "C"

    return sc, label


def pullback_quality(gap_pct: float, in_rank: str) -> float:
    """
    押し目の“実行しやすさ”を0-100で返す。
    gap_pct: 現在価格が基準INより何%上か（+は上、-は下）
    """
    sc = 50.0

    if in_rank == "強IN":
        sc += 20
    elif in_rank == "通常IN":
        sc += 10
    elif in_rank == "弱めIN":
        sc += 0
    else:
        sc -= 10

    # 追い気味は減点（+2%を超えると大きく減点）
    if np.isfinite(gap_pct):
        if gap_pct <= 0.8:
            sc += 20
        elif gap_pct <= 1.5:
            sc += 8
        elif gap_pct <= 2.5:
            sc -= 10
        elif gap_pct <= 4.0:
            sc -= 22
        else:
            sc -= 35

    return float(np.clip(sc, 0, 100))


def decide_al_for_swing(runner_sc: float, runner_label: str, rr: float, ev_r: float, in_rank: str) -> int:
    """
    SwingのAL（1/2/3）を返す。
    """
    if not (np.isfinite(rr) and np.isfinite(ev_r) and np.isfinite(runner_sc)):
        return 1

    if rr < 1.8 or ev_r < 0.3:
        return 1

    if runner_label.startswith("A") and runner_sc >= 75 and ev_r >= 0.6 and rr >= 2.5 and in_rank in ("強IN", "通常IN"):
        return 3

    if runner_sc >= 65 and ev_r >= 0.45 and rr >= 2.0 and in_rank in ("強IN", "通常IN", "弱めIN"):
        return 2

    return 1


def al3_score(runner_sc: float, rr: float, ev_r: float, pb_sc: float) -> float:
    """
    AL3の最終選抜スコア（大勝ちモードの一点集中用）。
    """
    s = 0.0
    if np.isfinite(runner_sc):
        s += runner_sc * 0.45
    if np.isfinite(ev_r):
        s += (ev_r * 100.0) * 0.35
    if np.isfinite(rr):
        s += (rr * 10.0) * 0.15
    if np.isfinite(pb_sc):
        s += pb_sc * 0.05
    return float(s)


def enforce_single_al3(cands: List[Dict], event_near: bool = False) -> List[Dict]:
    """
    1日1銘柄 AL3 ルール。
    - event_near の日は AL3 1銘柄以外を表示しない
    - 通常日は AL3 1位のみAL3、残りはAL2へ格下げ
    """
    if not cands:
        return cands

    al3 = [c for c in cands if int(c.get("al", 1)) == 3]
    if not al3:
        return cands

    al3_sorted = sorted(al3, key=lambda x: float(x.get("al3_score", 0.0)), reverse=True)
    best = al3_sorted[0]
    best_ticker = str(best.get("ticker", ""))

    if event_near:
        return [best]

    out: List[Dict] = []
    for c in cands:
        if str(c.get("ticker", "")) == best_ticker:
            c["al"] = 3
            out.append(c)
        else:
            if int(c.get("al", 1)) == 3:
                c["al"] = 2
            out.append(c)

    return out
