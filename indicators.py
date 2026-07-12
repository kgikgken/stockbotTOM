"""2週間スイング(swing2w)向けテクニカル指標 — 日次OHLCVのみから計算。

歪み系・モメンタム系とは独立。ATR/SMAはmispricing、TOB検出heuristicはmomentumの
実装を共有インフラとして再利用し(いずれも純粋な価格計算・特定戦略に依存しないため)、
本ファイルにはこのエンジン固有の指標(RSI・業種内相対z・ギャップ検出・52週ブレイク)のみ実装する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mispricing.indicators import sma, atr_wilder
from momentum.indicators import tob_suspect, bounce_confirmed, bounce_confirmed_strong, swing_high_depth  # noqa: F401


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_swing_features(df: pd.DataFrame, cfg) -> dict | None:
    """swing2w用の特徴量一式。業種内相対z化はプール段階(screen.py)でこの結果をもとに行う。"""
    if df is None or len(df) < max(cfg.breakout_days, 260) + 5:
        return None
    df = df.dropna(subset=["Close"]).copy()
    if len(df) < max(cfg.breakout_days, 260):
        return None

    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    close_now = float(c.iloc[-1])

    atr_n = atr_wilder(h, l, c, cfg.atr_period)
    sma50 = sma(c, 50)
    rsi_n = rsi(c, cfg.rsi_period)
    lr = np.log(c / c.shift(1)).dropna()

    # --- エンジンR用: 直近lookback日リターン(業種内相対z化は後段) ---
    lb = cfg.rel_lookback_days
    ret_lookback = float(np.log(c.iloc[-1] / c.iloc[-1 - lb])) if len(c) > lb else np.nan

    # --- エンジンM用①: 直近数日以内の単日ギャップ+出来高 ---
    vmean20 = float(v.iloc[-21:-1].mean())
    vol_ratio_today = float(v.iloc[-1]) / vmean20 if vmean20 > 0 else np.nan
    window = min(cfg.m_max_days_since_trigger + 1, len(lr))
    recent_lr = lr.iloc[-window:]
    gap_found, gap_days_since, gap_ret, gap_vol_ratio = False, None, None, None
    if len(recent_lr):
        idx_max = recent_lr.idxmax()
        candidate_ret = float(recent_lr.loc[idx_max])
        pos_in_v = v.index.get_loc(idx_max)
        vmean_at = float(v.iloc[max(0, pos_in_v - 21):pos_in_v - 1].mean()) if pos_in_v >= 22 else np.nan
        vol_ratio_at = float(v.iloc[pos_in_v]) / vmean_at if (vmean_at and vmean_at > 0) else np.nan
        if candidate_ret >= cfg.gap_threshold and vol_ratio_at and vol_ratio_at >= cfg.gap_vol_mult:
            gap_found = True
            gap_days_since = int((lr.index >= idx_max).sum()) - 1
            gap_ret = candidate_ret
            gap_vol_ratio = vol_ratio_at

    # --- エンジンM用②: 52週高値ブレイク(出来高確認込み) ---
    breakout_found = False
    if len(h) > cfg.breakout_days:
        donch_prev = float(h.iloc[-cfg.breakout_days - 1:-1].max())
        if close_now > donch_prev and vol_ratio_today and vol_ratio_today >= cfg.breakout_vol_mult:
            breakout_found = True

    adv20_jpy = float((c * v).iloc[-21:-1].mean())
    bounce = bounce_confirmed(df, cfg.bounce_lookback_days, cfg.bounce_min_close_position)
    bounce_strong = bounce_confirmed_strong(df)
    swing_high, days_since_high, depth_atr = swing_high_depth(df, atr_n, cfg.swing_high_lookback_days)
    ma50_ratio = close_now / float(sma50.iloc[-1]) - 1 if not pd.isna(sma50.iloc[-1]) else np.nan

    # 出来高特性(弱条件・情報表示のみ・ゲートにはしない): 直近下落局面の出来高が
    # 過去20日平均より少なければ「健全な押し目」の傍証として付記する(調査レポートで
    # 学術的裏付けが弱いとされたため、除外条件には使わない)。
    decline_window = min(cfg.rel_lookback_days, len(v) - 1)
    vmean20_ctx = float(v.iloc[-21:-1].mean())
    vmean_decline = float(v.iloc[-decline_window:].mean()) if decline_window > 0 else np.nan
    low_volume_decline = bool(vmean_decline < vmean20_ctx) if (vmean_decline == vmean_decline and vmean20_ctx > 0) else None

    return {
        "close": close_now, "atr": float(atr_n.iloc[-1]), "rsi": float(rsi_n.iloc[-1]),
        "ret_lookback": ret_lookback, "vol_ratio_today": vol_ratio_today,
        "gap_found": gap_found, "gap_days_since": gap_days_since, "gap_ret": gap_ret,
        "gap_vol_ratio": gap_vol_ratio,
        "breakout_found": breakout_found, "bounce_confirmed": bounce, "bounce_confirmed_strong": bounce_strong,
        "swing_high": swing_high, "days_since_swing_high": days_since_high, "pullback_depth_atr": depth_atr,
        "ma50_ratio": ma50_ratio, "low_volume_decline": low_volume_decline,
        "adv20_jpy": adv20_jpy,
        "last_date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
    }


def compute_sector_relative_z(items: list, cfg) -> dict:
    """各銘柄の直近lookback日リターンを業種内で相対化してz-score化する
    (エンジンR「業種内で相対的に売られ過ぎ」の主軸指標。歪み系v5.0の業種内リバーサル設計と同じ発想)。
    構成銘柄が少なすぎる業種(既定5銘柄未満)は判定不能として除外する
    (母数が少ないほど、日々のデータ被覆率の揺れでz値そのものが動きやすくなるため、
    安定性を優先して閾値を3→5に引き上げている)。"""
    by_sector: dict = {}
    for it in items:
        sec = it["row"].get("sector") or "不明"
        r = it["feat"]["ret_lookback"]
        if r is not None and not (isinstance(r, float) and np.isnan(r)):
            by_sector.setdefault(sec, []).append((it["row"]["ticker"], r))

    result = {}
    for sec, pairs in by_sector.items():
        if len(pairs) < cfg.rel_min_sector_members:
            continue
        rets = np.array([r for _, r in pairs])
        sd = float(rets.std(ddof=0))
        mu = float(rets.mean())
        for ticker, r in pairs:
            result[ticker] = 0.0 if sd < 1e-12 else (r - mu) / sd
    return result
