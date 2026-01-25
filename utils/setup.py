from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils.util import sma, rsi14, atr14, adv20, atr_pct_last, safe_float, clamp


# ------------------------------------------------------------
# Setup (形) 判定
# ------------------------------------------------------------
# A1 / A1-Strong : 押し目トレンドフォロー（主軸）
# A2            : 初動ブレイク（補助）
# B             : 初動ブレイク（レンジ上抜け寄り）
# S             : 需給歪み（例外・小枠）
# ------------------------------------------------------------


@dataclass
class SetupInfo:
    ticker: str
    sector: str

    setup: str  # A1 / A1-Strong / A2 / B / S
    tier: int  # 0/1/2（内部。NO-TRADE例外などで使用）

    action: str  # LINE表示用（指値で待つ、など）
    gu: bool  # ギャップアップ（寄り後再判定の目安）

    # 価格（すべて円）
    entry_price: float  # 中央指値（統一）
    entry_low: float    # 内部計算・参考（表示しない）
    entry_high: float   # 内部計算・参考（表示しない）

    sl: float
    tp1: float
    tp2: float

    # 指標
    rr: float        # TP2基準のRR
    rr_tp1: float    # TP1基準のRR（= 期待R）
    expected_r: float  # 期待R（TP1到達時のRで固定）
    expected_days: float  # 想定日数（中央値イメージ）
    rday: float      # 回転効率（期待R/日）

    # 品質（0-1想定、クランプ）
    trend_strength: float
    pullback_quality: float

    # 期待値系
    p_reach_tp1: float     # TP1到達確率（推定）
    adj_ev: float          # 期待値（補正）= 想定期待R（補正後）
    cagr_contrib: float    # CAGR寄与度 = (期待R×到達確率)/想定日数（時間効率ペナ込み後）


def liquidity_filters(df: pd.DataFrame) -> Tuple[bool, float, float, float]:
    """流動性フィルタ（ベース互換）
    Returns:
      ok, last_price, adv20, atr_pct
    """
    try:
        last_price = float(df["Close"].iloc[-1])
    except Exception:
        return False, 0.0, 0.0, 0.0

    try:
        a = float(adv20(df))
    except Exception:
        a = 0.0

    try:
        ap = float(atr_pct_last(df))
    except Exception:
        ap = 0.0

    # 最低限のフィルタ（ベース運用の安全装置）
    # - 価格: 100円以上
    # - ADV: 1000万円以上
    # - ATR%: 0.8%以上
    ok = (last_price >= 100.0) and (a >= 10e6) and (ap >= 0.8)
    return ok, last_price, a, ap



def _calc_trend_strength(close: pd.Series) -> float:
    """上昇トレンドの強さ（0-1目安）。"""
    ma25 = sma(close, 25).iloc[-1]
    ma75 = sma(close, 75).iloc[-1]
    if not np.isfinite(ma25) or not np.isfinite(ma75) or ma75 <= 0:
        return 0.0
    # 25>75 かつ乖離が大きいほど強い。上限を穏やかに。
    raw = (ma25 / ma75) - 1.0
    return float(clamp(raw / 0.10, 0.0, 1.0))


def _calc_pullback_quality(df: pd.DataFrame) -> float:
    """押し目品質（0-1目安）。"""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    v = df.get("Volume", pd.Series(index=df.index, dtype=float))

    # 直近高値更新後の調整が「浅く」「健全」なら高評価
    # - 戻り高値からの下落率
    hh = high.rolling(20).max().iloc[-1]
    cur = close.iloc[-1]
    if not np.isfinite(hh) or hh <= 0 or not np.isfinite(cur):
        return 0.0
    dd = (hh - cur) / hh  # 0〜
    dd_score = float(clamp(1.0 - dd / 0.08, 0.0, 1.0))  # 8%超の押しは減点

    # RSIが極端に崩れてない
    rsi = float(rsi14(close).iloc[-1])
    rsi_score = float(clamp((rsi - 45.0) / 15.0, 0.0, 1.0))  # 45→0, 60→1

    # 出来高が過熱し過ぎてない（押しで細る）
    if len(v) >= 10:
        v5 = float(v.iloc[-5:].mean()) if v.iloc[-5:].mean() == v.iloc[-5:].mean() else 0.0
        v20 = float(v.iloc[-20:].mean()) if len(v) >= 20 else float(v.mean())
        vol_score = float(clamp(1.0 - (v5 / max(v20, 1.0) - 1.0) / 0.5, 0.0, 1.0))
    else:
        vol_score = 0.5

    # 下ヒゲ・反転気配（ざっくり）
    body = abs(close.iloc[-1] - df["Open"].iloc[-1])
    wick = (min(close.iloc[-1], df["Open"].iloc[-1]) - low.iloc[-1])
    wick_score = float(clamp(wick / max(body, 1e-6), 0.0, 1.0))

    q = 0.40 * dd_score + 0.25 * rsi_score + 0.20 * vol_score + 0.15 * wick_score
    return float(clamp(q, 0.0, 1.0))


def _detect_setup(df: pd.DataFrame) -> str:
    """形の分類。"""
    close = df["Close"]
    high = df["High"]
    vol = df.get("Volume", pd.Series(index=df.index, dtype=float))

    # 上昇トレンド前提
    ma25 = sma(close, 25).iloc[-1]
    ma75 = sma(close, 75).iloc[-1]
    up = (close.iloc[-1] > ma25) and (ma25 > ma75)

    # A1系: トレンド継続で押し目
    # A1-Strong: 高値更新後の初押しに近い（高値からの調整が浅い）
    hh20 = high.rolling(20).max().iloc[-1]
    dd = (hh20 - close.iloc[-1]) / max(hh20, 1e-6) if np.isfinite(hh20) else 1.0
    if up and dd <= 0.06:
        # 強押し目: 直近20日高値付近からの浅い調整 + 25MA付近維持
        if close.iloc[-1] >= ma25 * 0.98:
            return "A1-Strong"
        return "A1"

    # A2: 直近の安値圏からの初動ブレイク（5日高値）
    hh5 = high.rolling(5).max().iloc[-1]
    if close.iloc[-1] >= hh5 and (vol.iloc[-1] > vol.iloc[-2] * 1.2 if len(vol) >= 2 else False):
        return "A2"

    # B: レンジ上抜け（20日高値）
    if close.iloc[-1] >= hh20 and (vol.iloc[-1] > vol.iloc[-2] * 1.3 if len(vol) >= 2 else False):
        return "B"

    return ""


def _detect_supply_distortion(df: pd.DataFrame, topix_df: Optional[pd.DataFrame]) -> bool:
    """需給歪み（例外）の簡易判定。
    条件例:
    - 指数が弱い（TOPIXが5日でマイナス）
    - 個別は逆行高（5日でプラス）
    - 出来高が枯れ→反転（直近で増加）
    """
    if topix_df is None or len(topix_df) < 10:
        return False
    try:
        s = df["Close"]
        v = df.get("Volume", pd.Series(index=df.index, dtype=float))
        ix = topix_df["Close"]

        r_stock = (float(s.iloc[-1]) / float(s.iloc[-6]) - 1.0)
        r_ix = (float(ix.iloc[-1]) / float(ix.iloc[-6]) - 1.0)

        if r_ix > -0.01:
            return False
        if r_stock < 0.02:
            return False

        v5 = float(v.iloc[-5:].mean()) if len(v) >= 5 else 0.0
        v20 = float(v.iloc[-20:].mean()) if len(v) >= 20 else float(v.mean())
        # 枯れ→増: 直近2日が20日平均以上
        if len(v) >= 3 and (v.iloc[-1] >= v20 and v.iloc[-2] >= v20) and (v5 <= v20 * 1.2):
            return True
    except Exception:
        return False
    return False


def build_setup_info(
    df: pd.DataFrame,
    sector: str,
    ticker: str,
    macro_on: bool = False,
    topix_df: Optional[pd.DataFrame] = None,
) -> Optional[SetupInfo]:
    """OHLCVから SetupInfo を生成。返せない場合は None。"""
    if df is None or len(df) < 80:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    setup = _detect_setup(df)

    # 需給歪み（例外）：強制的に S にする
    if _detect_supply_distortion(df, topix_df):
        setup = "S"

    if not setup:
        return None

    # 価格レンジ（内部用）
    atr = float(atr14(df).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None

    entry_low = float(close.iloc[-1] - 0.40 * atr)
    entry_high = float(close.iloc[-1] + 0.40 * atr)
    entry_price = float((entry_low + entry_high) / 2.0)  # 中央指値に統一

    # 損切りはATR基準（セットアップ別に少し差）
    sl_buf = 1.00 if setup in ("A1-Strong", "A1") else 1.20
    sl = float(entry_price - sl_buf * atr)

    # 利確はTP1/TP2
    # - 期待Rは TP1 を基準に固定（分割利確はスコア外）
    tp1 = float(entry_price + 1.00 * (entry_price - sl))  # 期待R=1Rを基本
    tp2 = float(entry_price + 1.60 * (entry_price - sl))  # 参考（伸ばす側）

    rr = float((tp2 - entry_price) / max(entry_price - sl, 1e-6))
    rr_tp1 = float((tp1 - entry_price) / max(entry_price - sl, 1e-6))
    expected_r = rr_tp1  # TP1基準で固定

    # 想定日数（TP1到達を基準にATRで逆算 → 1〜7日）
    expected_days = float(clamp((tp1 - entry_price) / atr, 1.0, 7.0))

    # 品質
    trend_strength = _calc_trend_strength(close)
    pullback_quality = _calc_pullback_quality(df)

    # 回転効率（期待R/日）
    rday = float(expected_r / max(expected_days, 1e-6))

    # Tier（内部制御用）
    if setup == "A1-Strong":
        tier = 0
    elif setup in ("A1", "A2"):
        tier = 1
    else:
        tier = 2

    action = "指値で待つ（現値IN禁止）"

    # 初動/歪みは保守的に期待R上限を少し抑える（CAGR補助エンジン）
    if setup in ("A2", "B"):
        expected_r = min(expected_r, 0.90)
        rday = float(expected_r / max(expected_days, 1e-6))
    if setup == "S":
        expected_r = min(expected_r, 0.50)
        expected_days = min(expected_days, 2.0)
        rday = float(expected_r / max(expected_days, 1e-6))

    return SetupInfo(
        ticker=ticker,
        sector=sector,
        setup=setup,
        tier=int(tier),
        action=action,
        gu=bool(gu),
        entry_price=round(entry_price, 1),
        entry_low=round(entry_low, 1),
        entry_high=round(entry_high, 1),
        sl=round(sl, 1),
        tp1=round(tp1, 1),
        tp2=round(tp2, 1),
        rr=round(rr, 2),
        rr_tp1=round(rr_tp1, 2),
        expected_r=round(expected_r, 2),
        expected_days=round(expected_days, 1),
        rday=round(rday, 2),
        trend_strength=float(round(trend_strength, 2)),
        pullback_quality=float(round(pullback_quality, 2)),
        p_reach_tp1=0.0,   # rr_ev.calc_ev で埋める
        adj_ev=0.0,        # rr_ev.calc_ev で埋める
        cagr_contrib=0.0,  # rr_ev.calc_ev で埋める
    )
