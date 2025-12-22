from __future__ import annotations

import numpy as np
import pandas as pd


def _last(s: pd.Series) -> float:
    try:
        return float(s.iloc[-1])
    except Exception:
        return np.nan


def _add(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()

    c = df["Close"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    df["ma20"] = c.rolling(20).mean()
    df["ma50"] = c.rolling(50).mean()
    df["ma60h"] = c.rolling(60).max()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # 20日売買代金（概算）
    df["turnover"] = (c * v).rolling(20).mean()

    # 60日高値からの距離（%）
    df["off_high"] = (c - df["ma60h"]) / (df["ma60h"] + 1e-9) * 100.0

    # 20MAの傾き（5日平均）
    df["ma20_slope5"] = df["ma20"].pct_change(fill_method=None).rolling(5).mean()

    return df


# ------------------------------------------------------------
# TrendGate（順張り専用）
# ------------------------------------------------------------
def trend_gate(hist: pd.DataFrame) -> bool:
    """逆張りを排除する前提条件（トレンドが"継続"しているか）"""
    if hist is None or len(hist) < 80:
        return False

    df = _add(hist)

    c = _last(df["Close"])
    ma20 = _last(df["ma20"])
    ma50 = _last(df["ma50"])

    if not (np.isfinite(c) and np.isfinite(ma20) and np.isfinite(ma50)):
        return False

    # 1) 20MA > 50MA（今日）
    if not (ma20 > ma50):
        return False

    # 2) 終値が50MAの上（下降トレンドの戻りを排除）
    if not (c > ma50):
        return False

    # 3) 20MA > 50MA が直近15営業日で継続
    ma20s = df["ma20"].astype(float)
    ma50s = df["ma50"].astype(float)
    if len(ma20s) < 65:
        return False
    cond = (ma20s > ma50s).tail(15)
    if cond.isna().any() or (cond.sum() < 15):
        return False

    # 4) 20MAの傾きがプラス（横ばい〜下向きを排除）
    slope5 = _last(df["ma20_slope5"])
    if not np.isfinite(slope5) or slope5 <= 0:
        return False

    return True


# ------------------------------------------------------------
# 押し目判定（STEP2）
# ------------------------------------------------------------
def calc_inout_for_stock(hist: pd.DataFrame):
    """戻り値: in_rank, tp_dummy, sl_dummy（互換用）"""
    if hist is None or len(hist) < 80:
        return "様子見", 0, 0

    df = _add(hist)

    c = _last(df["Close"])
    ma20 = _last(df["ma20"])
    ma50 = _last(df["ma50"])
    rsi = _last(df["rsi"])
    off_high = _last(df["off_high"])
    turnover = _last(df["turnover"])

    # まずはTrendGateの前提（順張り）
    if not trend_gate(hist):
        return "様子見", 0, 0

    if not all(np.isfinite(x) for x in [c, ma20, ma50, rsi, off_high, turnover]):
        return "様子見", 0, 0

    # 共通: 高値から遠すぎ/近すぎを排除（"勝てる形"の押し目だけ）
    # （-10%〜-3%：押し目はあるが崩れてはいない）
    if not (-10.0 <= off_high <= -3.0):
        return "様子見", 0, 0

    # A) 正統派トレンド押し目（最優先）
    # - 20MA付近で反発待ち（±0.8%）
    # - RSIは熱すぎず弱すぎず
    if (
        ma20 > ma50 and
        abs(c / (ma20 + 1e-9) - 1.0) <= 0.008 and
        42.0 <= rsi <= 55.0 and
        turnover >= 1e8
    ):
        return "強IN", 0, 0

    # B) ブレイク後の初押し（やや許容）
    if (
        ma20 > ma50 and
        abs(c / (ma20 + 1e-9) - 1.0) <= 0.015 and
        40.0 <= rsi <= 60.0 and
        turnover >= 1e8
    ):
        return "通常IN", 0, 0

    return "様子見", 0, 0


# ------------------------------------------------------------
# スコア（内部フィルタ用：表示しない前提）
# ------------------------------------------------------------
def score_stock(hist: pd.DataFrame) -> float | None:
    """0-100（内部の足切り用。表示用途ではない）"""
    if hist is None or len(hist) < 80:
        return None

    df = _add(hist)
    score = 0.0

    ma20 = _last(df["ma20"])
    ma50 = _last(df["ma50"])
    turnover = _last(df["turnover"])
    off_high = _last(df["off_high"])
    rsi = _last(df["rsi"])

    if np.isfinite(ma20) and np.isfinite(ma50) and ma20 > ma50:
        score += 30.0
    if np.isfinite(turnover) and turnover >= 1e8:
        score += 30.0
    if np.isfinite(off_high) and (-10.0 <= off_high <= -3.0):
        score += 20.0
    if np.isfinite(rsi) and (40.0 <= rsi <= 60.0):
        score += 20.0

    return float(np.clip(score, 0.0, 100.0))
