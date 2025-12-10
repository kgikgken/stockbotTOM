import numpy as np
import pandas as pd


def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _vola20(close: pd.Series) -> float:
    try:
        r = np.log(close / close.shift(1))
        return float(np.nanstd(r.tail(20)))
    except Exception:
        return np.nan


def compute_rr(hist: pd.DataFrame, mkt_score: int) -> dict:
    """
    ・エントリー: 「明日寄り ≒ 直近終値」を想定
    ・損切り: 直近スイング安値 or ATR
    ・利確: 直近スイング高値
    から RR を算出する
    """
    if hist is None or len(hist) < 40:
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0)

    close = hist["Close"].astype(float)
    high = hist["High"].astype(float)
    low = hist["Low"].astype(float)

    entry = _last(close)
    if not np.isfinite(entry) or entry <= 0:
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0)

    # 20日ボラ（リスクの目安）
    vola = _vola20(close)
    if not np.isfinite(vola) or vola <= 0:
        vola = 0.02

    # スイング安値（直近10〜20日）
    try:
        swing_low_10 = float(low.tail(10).min())
        swing_low_20 = float(low.tail(20).min())
        swing_low = min(swing_low_10, swing_low_20)
    except Exception:
        swing_low = entry * (1.0 - 2.0 * vola)

    sl_pct_raw = swing_low / entry - 1.0  # マイナス想定

    # 安値がエントリーより上に来てしまう or 浅すぎる場合は ATR から補正
    if sl_pct_raw >= -0.005:
        sl_pct_raw = -max(1.5 * vola, 0.015)

    # スイング高値（直近20〜40日）
    try:
        swing_high_20 = float(high.tail(20).max())
        swing_high_40 = float(high.tail(40).max())
        swing_high = max(swing_high_20, swing_high_40)
    except Exception:
        swing_high = entry * (1.0 + 4.0 * vola)

    tp_pct_raw = swing_high / entry - 1.0

    # 上方向がほぼ無い場合はボラから最低限のターゲットを設定
    if tp_pct_raw < 0.02:
        tp_pct_raw = max(3.0 * vola, 0.03)

    # ボラに応じて過度なターゲットを抑制（現実的な伸びに）
    tp_pct = float(np.clip(tp_pct_raw, 0.04, 0.30))
    sl_pct = float(np.clip(sl_pct_raw, -0.15, -0.01))

    rr = tp_pct / abs(sl_pct) if sl_pct < 0 else 0.0

    # 地合いで少しだけ RR を補正（悪い地合いほど RR が下がりにくいように）
    if mkt_score < 45:
        rr *= 0.95
    elif mkt_score > 65:
        rr *= 1.05

    return dict(
        rr=float(rr),
        entry=float(entry),
        tp_pct=float(tp_pct),
        sl_pct=float(sl_pct),
    )