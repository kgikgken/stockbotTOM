import numpy as np
import pandas as pd


# ============================================================
# 内部ユーティリティ
# ============================================================
def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance.Ticker(ticker).history() の DataFrame に
    スコア計算用の指標を追加する。
    """
    df = df.copy()

    if df.empty:
        return df

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float)

    # 価格＆移動平均
    df["close"] = close
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 売買代金 & 20日平均
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    # ボラティリティ20（日次リターンstd×√20）
    ret = close.pct_change()
    df["vola20"] = ret.rolling(20).std() * np.sqrt(20)

    # 60日高値からの距離 & 日数
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100.0

        tail = close.tail(60)
        idx_max = int(np.argmax(tail.values))
        df["days_since_high60"] = (len(tail) - 1) - idx_max
    else:
        df["off_high_pct"] = np.nan
        df["days_since_high60"] = np.nan

    # 20MAの傾き
    df["trend_slope20"] = df["ma20"].pct_change()

    # 下ヒゲ比率
    rng = high - low
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["lower_shadow_ratio"] = np.where(rng > 0, lower_shadow / rng, 0.0)

    return df


def _extract_metrics(df: pd.DataFrame) -> dict:
    """
    スコア計算に必要な指標を最終行から抜き出す。
    NaN はそのまま（あとで 0 扱いにする）。
    """
    last = df.iloc[-1]
    keys = [
        "close",
        "ma20",
        "ma50",
        "rsi14",
        "turnover_avg20",
        "vola20",
        "off_high_pct",
        "trend_slope20",
        "lower_shadow_ratio",
        "days_since_high60",
    ]
    m = {}
    for k in keys:
        m[k] = _safe_float(last.get(k, np.nan))
    return m


# ============================================================
# 各コンポーネントスコア
# ============================================================
def _trend_score(m: dict) -> float:
    """
    トレンド強度スコア（0〜40）
      - 20MAの傾き
      - MA並び(価格 > MA20 > MA50)
      - 高値からの位置（押しが深すぎないか）
    """
    close = m.get("close", np.nan)
    ma20 = m.get("ma20", np.nan)
    ma50 = m.get("ma50", np.nan)
    slope = m.get("trend_slope20", np.nan)
    off_high = m.get("off_high_pct", np.nan)

    score = 0.0

    # 1) MA並び（最大18点）
    if np.isfinite(close) and np.isfinite(ma20) and np.isfinite(ma50):
        if close > ma20 > ma50:
            score += 18.0  # きれいな上昇トレンド
        elif close > ma20 or ma20 > ma50:
            score += 10.0  # どちらかは上向き
        else:
            score += 4.0   # まだ崩壊していない

    # 2) 20MAの傾き（最大12点）
    if np.isfinite(slope):
        # 0.0〜0.01（0〜1%/日）くらいまでをフルスコア
        if slope <= 0:
            score += 0.0
        elif slope >= 0.01:
            score += 12.0
        else:
            score += 12.0 * (slope / 0.01)

    # 3) 高値からの位置（最大10点）
    if np.isfinite(off_high):
        # 0〜-5% → 押し浅め、フルスコア
        if 0 >= off_high >= -5:
            score += 10.0
        # -5〜-15% → 押しとして悪くない（線形に減点）
        elif -15 <= off_high < -5:
            score += max(4.0, 10.0 - (abs(off_high + 5) * 0.6))
        # それ以下（-15%以上の崩れ）は加点なし

    return float(np.clip(score, 0.0, 40.0))


def _pullback_score(m: dict) -> float:
    """
    押し目の質スコア（0〜40）
      - RSI(14)
      - 高値からの下落率
      - 日柄（高値からの日数）
      - 下ヒゲの強さ
    """
    rsi = m.get("rsi14", np.nan)
    off_high = m.get("off_high_pct", np.nan)
    days = m.get("days_since_high60", np.nan)
    shadow = m.get("lower_shadow_ratio", np.nan)

    score = 0.0

    # 1) RSI（最大15点）
    if np.isfinite(rsi):
        if 30 <= rsi <= 45:
            score += 15.0  # 理想的な押しゾーン
        elif 25 <= rsi < 30 or 45 < rsi <= 55:
            score += 9.0
        elif 20 <= rsi < 25 or 55 < rsi <= 65:
            score += 4.0
        else:
            score += 1.0

    # 2) 高値からの下落率（最大12点）
    if np.isfinite(off_high):
        if -8 <= off_high <= -4:
            score += 12.0  # きれいな浅め押し
        elif -15 <= off_high < -8:
            score += 8.0
        elif -25 <= off_high < -15:
            score += 3.0
        else:
            score += 0.0

    # 3) 日柄（最大8点）
    if np.isfinite(days):
        if 3 <= days <= 12:
            score += 8.0
        elif 1 <= days < 3 or 12 < days <= 20:
            score += 4.0

    # 4) 下ヒゲ（最大5点）
    if np.isfinite(shadow):
        if shadow >= 0.6:
            score += 5.0
        elif shadow >= 0.4:
            score += 3.0
        elif shadow >= 0.25:
            score += 1.0

    return float(np.clip(score, 0.0, 40.0))


def _liquidity_score(m: dict) -> float:
    """
    流動性 & ボラティリティスコア（0〜20）
    """
    t = m.get("turnover_avg20", np.nan)
    v = m.get("vola20", np.nan)

    score = 0.0

    # 1) 売買代金（最大14点）
    if np.isfinite(t):
        # 20億以上 → フル
        if t >= 20e8:
            score += 14.0
        elif t >= 2e8:
            # 2億〜20億 → 線形
            score += 4.0 + (t - 2e8) / (18e8) * 10.0
        elif t >= 1e8:
            score += 2.0  # 最低ラインクリア
        # それ未満は0点（そもそもフィルタで落とす想定）

    # 2) ボラ（最大6点）
    if np.isfinite(v):
        # 2〜5% くらいのボラが一番扱いやすい
        if 0.02 <= v <= 0.05:
            score += 6.0
        elif 0.01 <= v < 0.02 or 0.05 < v <= 0.08:
            score += 3.0
        else:
            score += 1.0

    return float(np.clip(score, 0.0, 20.0))


# ============================================================
# 公開インターフェース
# ============================================================
def score_stock(hist: pd.DataFrame) -> int | None:
    """
    個別銘柄の総合スコア（0〜100）を返す。

    Aランク: 80以上
    Bランク: 70以上

    NaN が混ざっていても、最終的に 0〜100 の int に必ず収まるよう設計。
    条件を満たせない場合は None を返す。
    """
    if hist is None or len(hist) < 60:
        return None

    # インジケータ付与
    df = _add_indicators(hist)
    if df.empty:
        return None

    # メトリクス抽出
    m = _extract_metrics(df)

    # 重要指標が全部 NaN ならスキップ
    if all(not np.isfinite(v) for v in m.values()):
        return None

    # 各スコア
    trend = _trend_score(m)        # 0〜40
    pullback = _pullback_score(m)  # 0〜40
    liq = _liquidity_score(m)      # 0〜20

    total = float(trend + pullback + liq)

    if not np.isfinite(total):
        return None

    total = float(np.clip(total, 0.0, 100.0))
    return int(round(total))
