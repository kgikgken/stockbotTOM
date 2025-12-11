import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 基本の安全取得
# ---------------------------------------------------------
def _last(series):
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


# ---------------------------------------------------------
# スコアリング
# ---------------------------------------------------------
def score_stock(hist: pd.DataFrame) -> float:
    """
    0〜100点のスコア
    main.py の MIN_SCORE と連動するように調整済み
    """

    if hist is None or len(hist) < 60:
        return 0.0

    df = hist.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # ---- 移動平均 ----
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    score = 0.0

    # トレンド（MA20 > MA50）
    if _last(ma20) > _last(ma50):
        score += 25
    elif _last(ma20) > _last(ma50) * 0.995:
        score += 15

    # MA5 > MA20（短期勢い）
    if _last(ma5) > _last(ma20):
        score += 15

    # ボラティリティ（安定性評価）
    vola = close.pct_change().rolling(20).std().iloc[-1]
    if np.isfinite(vola):
        if vola < 0.015:
            score += 15
        elif vola < 0.025:
            score += 8

    # 直近強さ（5日騰落）
    try:
        chg_5d = close.iloc[-1] / close.iloc[-6] - 1
        if chg_5d > 0.03:
            score += 15
        elif chg_5d > 0.00:
            score += 8
    except Exception:
        pass

    # 出来高増（勢い）
    try:
        vol = df["Volume"].astype(float)
        if vol.iloc[-1] > vol.rolling(20).mean().iloc[-1] * 1.3:
            score += 10
    except Exception:
        pass

    return float(np.clip(score, 0, 100))


# ---------------------------------------------------------
# IN/OUT 判定（INランク + TP/SL）
# ---------------------------------------------------------
def calc_inout_for_stock(hist: pd.DataFrame):
    """
    戻り値
      - in_rank（強IN / 通常IN / 弱めIN / 様子見）
      - tp_pct（%）
      - sl_pct（% ※マイナスで返す）
    """

    if hist is None or len(hist) < 60:
        return "様子見", 0.0, 0.0

    df = hist.copy()
    close = df["Close"].astype(float)

    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]

    price = close.iloc[-1]

    # ---- IN ランク ----
    # 強IN条件：短期・中期が揃い押し目気味
    if price > ma5 > ma20 > ma50 and abs(price - ma20) / price < 0.03:
        in_rank = "強IN"

    # 通常IN：トレンド良好で押し目入り
    elif price > ma20 and ma20 > ma50:
        in_rank = "通常IN"

    # 弱めIN：一応上昇トレンド
    elif price > ma50 and ma20 > ma50:
        in_rank = "弱めIN"

    else:
        return "様子見", 0.0, 0.0

    # ---- TP/SL 設定 ----
    # ATRから強さと逆行幅を推定
    atr = _atr(df)

    if atr <= 0 or not np.isfinite(atr):
        atr = price * 0.02  # 保険

    # ATR を % に変換
    atr_pct = atr / price * 100

    # ランク別に TP/SL を変える（可変RR）
    if in_rank == "強IN":
        tp_pct = atr_pct * 3.0    # 大きめ
        sl_pct = -atr_pct * 1.0
    elif in_rank == "通常IN":
        tp_pct = atr_pct * 2.4
        sl_pct = -atr_pct * 1.0
    else:  # 弱めIN
        tp_pct = atr_pct * 2.0
        sl_pct = -atr_pct * 1.2

    return in_rank, float(tp_pct), float(sl_pct)


# ---------------------------------------------------------
# ATR（標準）
# ---------------------------------------------------------
def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs()
    ], axis=1).max(axis=1)

    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if np.isfinite(val) else 0.0