import numpy as np
import pandas as pd


# ==============================================
# 内部ヘルパ
# ==============================================
def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _atr_like(df: pd.DataFrame, window: int = 20) -> float:
    """
    簡易ATR（High/Low/Close から計算）
    """
    try:
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window).mean().iloc[-1]
        return float(atr)
    except Exception:
        return np.nan


# ==============================================
# エントリー強度ラベル
# ==============================================
def _judge_in_label(df: pd.DataFrame) -> str:
    """
    かなりシンプルな IN 強度判定
    - RSI, 20MA乖離、直近高値からの押しで分類
    """
    try:
        close = df["Close"].astype(float)
        ma20 = close.rolling(20).mean()

        # RSI14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        off_ma = (close - ma20) / ma20 * 100
        rolling_high = close.rolling(60).max()
        off_high = (close - rolling_high) / rolling_high * 100

        rsi_last = float(rsi.iloc[-1])
        off_ma_last = float(off_ma.iloc[-1])
        off_high_last = float(off_high.iloc[-1])

        # 強IN: 上昇トレンド中の初押しイメージ
        if 30 <= rsi_last <= 55 and -5 <= off_ma_last <= 2 and -25 <= off_high_last <= -5:
            return "強IN"

        # 通常IN
        if 25 <= rsi_last <= 60 and -8 <= off_ma_last <= 4 and -30 <= off_high_last <= 5:
            return "通常IN"

        # それ以外
        return "弱めIN"
    except Exception:
        return "通常IN"


# ==============================================
# RR計算（export）
# ==============================================
def compute_rr(
    hist: pd.DataFrame,
    mkt_score: int,
    entry_price: float | None = None,
) -> dict:
    """
    hist: yfinance の DataFrame
    mkt_score: 地合いスコア（0-100）
    entry_price: 既存ポジション用のエントリー価格（任意）

    戻り値:
        {
          "rr": RR倍数,
          "entry": エントリー価格,
          "tp_pct": 利確パーセンテージ(+),
          "sl_pct": 損切りパーセンテージ(-),
          "in_label": "強IN/通常IN/弱めIN"
        }
    """
    if hist is None or len(hist) < 20:
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0, in_label="")

    close = hist["Close"].astype(float)

    # エントリー価格：指定があればそれ、なければ直近終値
    if entry_price and entry_price > 0:
        entry = float(entry_price)
    else:
        entry = float(close.iloc[-1])

    # ボラ計算
    atr = _atr_like(hist)
    if not np.isfinite(atr) or atr <= 0:
        # ATR計算不能の場合はエントリーの2%を想定
        atr = entry * 0.02

    # 基本ストップ幅: 1.2 * ATR
    stop_abs = 1.2 * atr
    stop_pct = stop_abs / entry  # 正の値

    # 地合いに応じてストップ微調整（弱いときはややタイトに）
    if mkt_score <= 40:
        stop_pct *= 0.9
    elif mkt_score >= 60:
        stop_pct *= 1.1

    # 上下限（あまりにもタイト/広すぎるのを防ぐ）
    stop_pct = float(np.clip(stop_pct, 0.03, 0.12))  # 3%〜12%

    # ターゲット幅：ベース 2.4R（RR 2.4〜3.0 のレンジを目指す）
    base_rr = 2.4

    # IN強度でTPを調整
    in_label = _judge_in_label(hist)
    if in_label == "強IN":
        base_rr = 2.8
    elif in_label == "弱めIN":
        base_rr = 2.0

    tp_pct = stop_pct * base_rr

    rr = tp_pct / stop_pct if stop_pct > 0 else 0.0

    return dict(
        rr=float(rr),
        entry=float(entry),
        tp_pct=float(tp_pct),
        sl_pct=float(-stop_pct),
        in_label=in_label,
    )