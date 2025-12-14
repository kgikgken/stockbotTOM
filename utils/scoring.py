from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ============================================================
# v11.1 PERFECT scoring (spec-complete)
# 出力キー:
#   mode: 'swing' / 'day'
#   total_score: float
#   rr_raw / ev_r_raw: float  (生)
#   rr_adj / ev_r_adj: float  (v11.1補正後)
#   tp_price / sl_price / in_price / in_diff_pct
#   gu_danger: bool
#   reach_rate: float (0-1)
#   reject_reason: str
# ============================================================


def _last(series: pd.Series, default=np.nan) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return float(default)

def _sma(series: pd.Series, n: int) -> float:
    if len(series) < n:
        return float("nan")
    return float(series.tail(n).mean())

def _atr(df: pd.DataFrame, n: int = 14) -> float:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    return float(atr.iloc[-1]) if len(atr) else float("nan")

def _rsi(series: pd.Series, n: int = 14) -> float:
    if len(series) < n + 1:
        return float("nan")
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = ru / (rd.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def _rolling_high(df: pd.DataFrame, n: int) -> float:
    if len(df) < n + 1:
        return float("nan")
    return float(df["high"].astype(float).iloc[-n:].max())

def _rolling_low(df: pd.DataFrame, n: int) -> float:
    if len(df) < n + 1:
        return float("nan")
    return float(df["low"].astype(float).iloc[-n:].min())

def _calc_reach_rate_breakout(df: pd.DataFrame, lookback_events: int, forward_days: int, target_mult_of_atr: float) -> float:
    """
    簡易「構造的到達率」：
      - 20日高値ブレイク（終値で上抜け）をイベントと定義
      - エントリー = 翌日始値（近似で当日終値でも可だがここは翌日始値）
      - ターゲット = entry + target_mult_of_atr * ATR(entry時点)
      - forward_days 日以内に高値がターゲット到達した割合

    ※厳密ではなく「届きにくい銘柄の根絶」用の現実フィルタ
    """
    if len(df) < 60:
        return float("nan")

    d = df.copy().reset_index(drop=True)
    close = d["close"].astype(float)
    high = d["high"].astype(float)
    open_ = d["open"].astype(float)

    # 20日高値（当日含まず）
    roll_high = high.rolling(20).max().shift(1)
    breakout = (close > roll_high) & roll_high.notna()

    idx = np.where(breakout.values)[0].tolist()
    if not idx:
        return float("nan")

    # 直近イベントから採用
    idx = idx[-lookback_events:]

    hits = 0
    total = 0
    for i in idx:
        # 翌日が無いと成立しない
        if i + 1 >= len(d):
            continue
        entry = float(open_.iloc[i + 1])
        # ATRをイベント時点の14ATRで近似
        atr = _atr(d.iloc[: i + 1], 14)
        if not (atr and atr > 0 and math.isfinite(atr)):
            continue
        target = entry + target_mult_of_atr * atr
        # 先読みウィンドウ
        j2 = min(len(d) - 1, i + 1 + forward_days)
        max_high = float(high.iloc[i + 1 : j2 + 1].max())
        total += 1
        if max_high >= target:
            hits += 1

    if total == 0:
        return float("nan")
    return float(hits / total)

def _in_price_swing(close: pd.Series, atr: float) -> float:
    # 理想押し目：20SMA - 0.3ATR（やや深め）
    sma20 = _sma(close, 20)
    if math.isnan(sma20) or not (atr and atr > 0):
        return float("nan")
    return float(sma20 - 0.3 * atr)

def _is_perfect_first_leg(df: pd.DataFrame, atr: float) -> bool:
    """
    Swing除外：初動完璧すぎ（＝Day向き）
    - 直近5日で +2.0ATR 以上上昇、かつ押しが浅い
    """
    if len(df) < 10 or not (atr and atr > 0):
        return False
    close = df["close"].astype(float)
    chg = float(close.iloc[-1] - close.iloc[-6])
    if chg >= 2.0 * atr:
        # 押しの浅さ：直近3日で低下が0.5ATR未満
        pull = float(close.iloc[-1] - close.iloc[-3:].min())
        if pull < 0.5 * atr:
            return True
    return False

def score_stock(ticker: str, df: pd.DataFrame, meta: Optional[Dict] = None, run_mode: str = "preopen", cfg: Optional[Dict] = None) -> Dict:
    cfg = cfg or {}
    meta = meta or {}

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)

    last = _last(close)
    atr = _atr(df, 14)
    rsi = _rsi(close, 14)
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    trend_up = bool((not math.isnan(sma20)) and (not math.isnan(sma50)) and sma20 > sma50)

    # Swing：理想押し目
    in_price = _in_price_swing(close, atr)
    in_diff_pct = float((last - in_price) / last * 100.0) if (math.isfinite(in_price) and last > 0) else float("nan")

    # Swing：SL/TP（構造 + ATR）
    sup = _rolling_low(df.iloc[:-1], 20)  # 当日含まず
    brk = _rolling_high(df.iloc[:-1], 20)

    sl = float(sup - 0.2 * atr) if (math.isfinite(sup) and atr and atr > 0) else float("nan")
    tp = float(brk + 0.2 * atr) if (math.isfinite(brk) and atr and atr > 0) else float("nan")

    swing_risk = float(last - sl) if (math.isfinite(sl)) else float("nan")
    swing_reward = float(tp - last) if (math.isfinite(tp)) else float("nan")
    rr_raw = float(swing_reward / swing_risk) if (math.isfinite(swing_reward) and math.isfinite(swing_risk) and swing_risk > 0) else 0.0

    # 追加補助（理論RRだけ高いを潰す：ATR基準）
    stop_atr = float(swing_risk / atr) if (atr and atr > 0 and math.isfinite(swing_risk)) else 0.0
    tgt_atr = float(swing_reward / atr) if (atr and atr > 0 and math.isfinite(swing_reward)) else 0.0
    atr_ok = (stop_atr >= float(cfg.get("MIN_STOP_ATR", 0.7))) and (tgt_atr >= float(cfg.get("MIN_TARGET_ATR", 1.0)))

    # EV（R）：簡易勝率proxy（IN近い/トレンド/過熱で減点）
    in_zone = bool(math.isfinite(in_price) and atr and atr > 0 and abs(last - in_price) / atr <= 0.8)
    p = 0.33 + (0.17 if in_zone else 0.0) + (0.12 if trend_up else 0.0) + (0.06 if (not math.isnan(rsi) and rsi < 65) else 0.0)
    p = float(np.clip(p, 0.10, 0.70))
    ev_raw = float(p * rr_raw - (1 - p) * 1.0)

    # v11.1(②) 構造TPチェック：TP > 20日高値×1.02
    tp_over_mult = float(cfg.get("TP_OVER_20D_HIGH_MULT", 1.02))
    tp_pen = float(cfg.get("TP_OVER_PENALTY_MULT", 0.7))
    tp_excl = float(cfg.get("TP_OVER_EXCLUDE_MULT", 1.06))

    rr_adj = rr_raw
    ev_adj = ev_raw
    reject_reason = ""

    if math.isfinite(brk) and math.isfinite(tp):
        if tp > brk * tp_excl:
            reject_reason = "TPが構造的に遠すぎ（20日高値超過）"
        elif tp > brk * tp_over_mult:
            rr_adj *= tp_pen
            ev_adj *= tp_pen

    # v11.1(②) 構造的到達率
    reach_rate = _calc_reach_rate_breakout(
        df=df,
        lookback_events=int(cfg.get("REACH_RATE_LOOKBACK_EVENTS", 30)),
        forward_days=int(cfg.get("REACH_RATE_FORWARD_DAYS", 10)),
        target_mult_of_atr=1.5,  # Swingの伸び代をざっくり
    )
    if math.isfinite(reach_rate):
        if reach_rate < float(cfg.get("REACH_RATE_TH", 0.60)):
            ev_adj *= float(cfg.get("REACH_RATE_EV_MULT", 0.7))

    # Swing除外：初動完璧すぎ（Day向き）
    if _is_perfect_first_leg(df, atr):
        # Swing側の点を落としてDay側に寄せる
        ev_adj *= 0.7
        rr_adj *= 0.9

    # Swingスコア
    swing_score = 0.0
    swing_score += 35.0 * (1.0 if in_zone else 0.0)
    swing_score += 20.0 * (1.0 if trend_up else 0.0)
    swing_score += 25.0 * float(np.clip(rr_adj / 3.0, 0, 1))
    if not math.isnan(rsi):
        swing_score += 20.0 * float(np.clip((70 - rsi) / 40.0, 0, 1))
    if not atr_ok:
        swing_score -= 35.0
    if reject_reason:
        swing_score -= 999.0

    # ===== Day案 =====
    # トリガー：20日高値（当日含まず）
    trigger = brk
    day_entry = float(trigger) if math.isfinite(trigger) else last
    day_sl = float(day_entry - 1.0 * atr) if (atr and atr > 0) else float("nan")
    day_tp = float(day_entry + 1.5 * atr) if (atr and atr > 0) else float("nan")

    day_risk = float(day_entry - day_sl) if math.isfinite(day_sl) else float("nan")
    day_reward = float(day_tp - day_entry) if math.isfinite(day_tp) else float("nan")
    day_rr_raw = float(day_reward / day_risk) if (math.isfinite(day_reward) and math.isfinite(day_risk) and day_risk > 0) else 0.0

    # GU危険域（v11系）
    gu_danger = False
    if atr and atr > 0 and math.isfinite(trigger):
        if run_mode == "postopen":
            # 実ギャップ：最新バーが当日なら open と前日closeで判定
            if len(df) >= 2:
                gap = float(open_.iloc[-1] - close.iloc[-2])
                gap_atr = abs(gap) / atr
                gu_danger = bool(gap_atr >= float(cfg.get("MAX_GU_DANGER_ATR_POSTOPEN", 1.2)))
        else:
            # 寄り前：trigger までの距離が小さすぎ＝寄りで飛びやすい危険域（想定）
            dist = float(trigger - last)
            gap_atr = abs(dist) / atr
            gu_danger = bool(gap_atr <= float(cfg.get("MAX_GU_DANGER_ATR_PREOPEN", 0.8)) and last >= trigger - 0.2 * atr)

    # Day EV（保守的）
    p2 = 0.28 + (0.17 if trend_up else 0.0) + (0.05 if (not math.isnan(rsi) and rsi > 45) else 0.0)
    p2 = float(np.clip(p2, 0.10, 0.60))
    day_ev_raw = float(p2 * day_rr_raw - (1 - p2) * 1.0)
    day_rr_adj = day_rr_raw
    day_ev_adj = day_ev_raw
    if gu_danger:
        day_ev_adj *= 0.6
        day_rr_adj *= 0.8

    # Dayスコア（押し戻し→再上昇は寄り後判定側で強化する想定。v11.1はベースのみ）
    day_score = 0.0
    day_score += 40.0 * float(np.clip(day_rr_adj / 2.0, 0, 1))
    day_score += 35.0 * (1.0 if trend_up else 0.0)
    if not math.isnan(rsi):
        day_score += 15.0 * float(np.clip((rsi - 40) / 30.0, 0, 1))
    if gu_danger:
        day_score -= 25.0

    # モード選択：Day優位ならDay
    mode = "swing"
    total_score = swing_score
    rr_out_raw = rr_raw
    ev_out_raw = ev_raw
    rr_out_adj = rr_adj
    ev_out_adj = ev_adj
    tp_out = tp
    sl_out = sl

    if math.isfinite(trigger) and day_score > swing_score:
        mode = "day"
        total_score = day_score
        rr_out_raw = day_rr_raw
        ev_out_raw = day_ev_raw
        rr_out_adj = day_rr_adj
        ev_out_adj = day_ev_adj
        tp_out = day_tp
        sl_out = day_sl

    return {
        "mode": mode,
        "total_score": float(total_score),

        "rr_raw": float(rr_out_raw),
        "ev_r_raw": float(ev_out_raw),
        "rr_adj": float(rr_out_adj),
        "ev_r_adj": float(ev_out_adj),

        "tp_price": float(tp_out) if math.isfinite(tp_out) else float("nan"),
        "sl_price": float(sl_out) if math.isfinite(sl_out) else float("nan"),

        "in_price": float(in_price) if math.isfinite(in_price) else float("nan"),
        "in_diff_pct": float(in_diff_pct) if math.isfinite(in_diff_pct) else float("nan"),

        "gu_danger": bool(gu_danger) if mode == "day" else False,
        "reach_rate": float(reach_rate) if math.isfinite(reach_rate) else float("nan"),
        "reject_reason": str(reject_reason),
    }
