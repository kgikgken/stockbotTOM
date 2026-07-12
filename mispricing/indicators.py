"""Technical indicators for Gate1 (単一ソース算出指標 → 既定で未確認扱い)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    au = up.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    ad = dn.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs = au / ad.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(ad.notna(), np.nan)


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()


def compute_features(df: pd.DataFrame) -> dict | None:
    """df: OHLCV daily (Open/High/Low/Close/Volume). Returns latest-bar feature dict."""
    if df is None or len(df) < 260:
        return None
    df = df.dropna(subset=["Close"]).copy()
    if len(df) < 260:
        return None

    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    sma25 = sma(c, 25)
    sma200 = sma(c, 200)
    dev25 = c / sma25 - 1.0

    dev_win = dev25.iloc[-252:].dropna()
    if len(dev_win) < 200:
        return None
    mu, sd = float(dev_win.mean()), float(dev_win.std(ddof=0))
    dev_now = float(dev25.iloc[-1])
    z_dev = (dev_now - mu) / sd if sd > 1e-12 else np.nan
    pctl = float((dev_win <= dev_now).mean() * 100.0)

    rsi14 = float(rsi_wilder(c, 14).iloc[-1])
    atr14 = float(atr_wilder(h, l, c, 14).iloc[-1])
    close_now = float(c.iloc[-1])
    atr_pct = atr14 / close_now * 100.0 if close_now > 0 else np.nan

    logret = np.log(c / c.shift(1))
    ret_sd60 = float(logret.iloc[-61:-1].std(ddof=0))
    vmean20 = float(v.iloc[-21:-1].mean())
    vsd20 = float(v.iloc[-21:-1].std(ddof=0))
    vol_z_now = (float(v.iloc[-1]) - vmean20) / vsd20 if vsd20 > 1e-9 else 0.0

    # カタリスト痕跡: 直近lookback日に |日次リターン|>=kσ or 出来高z>=k の日があるか
    def event_signature(lookback: int, ret_k: float, vol_k: float) -> bool:
        r = logret.iloc[-lookback:]
        if ret_sd60 > 1e-9 and (r.abs() >= ret_k * ret_sd60).any():
            return True
        vv = v.iloc[-lookback:]
        if vsd20 > 1e-9 and ((vv - vmean20) / vsd20 >= vol_k).any():
            return True
        return False

    # 押し目/リバウンド日数: 直近10営業日の高値(安値)から何日経過したか
    c10 = c.iloc[-11:]
    dip_days = int(len(c10) - 1 - int(np.argmax(c10.values)))
    rebound_days = int(len(c10) - 1 - int(np.argmin(c10.values)))

    adv20_jpy = float((c * v).iloc[-21:-1].mean())
    low5 = float(l.iloc[-5:].min())
    high5 = float(h.iloc[-5:].max())

    s200_now = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else np.nan
    s200_prev = float(sma200.iloc[-61]) if len(sma200) > 61 and not np.isnan(sma200.iloc[-61]) else np.nan
    dn_trend = (not np.isnan(s200_now) and not np.isnan(s200_prev)
                and s200_now < s200_prev and close_now < s200_now)
    up_trend = (not np.isnan(s200_now) and not np.isnan(s200_prev)
                and s200_now > s200_prev and close_now > s200_now)

    # 空売り価格規制の近接シグナル(トリガー方式: 基準値段比-10%): 直近2営業日で接近/到達したか
    prev_close = float(c.iloc[-2])
    reg_hit = bool((float(l.iloc[-1]) <= prev_close * 0.90)
                   or (len(c) > 2 and float(l.iloc[-2]) <= float(c.iloc[-3]) * 0.90))

    return {
        "close": close_now, "sma25": float(sma25.iloc[-1]), "sma200": s200_now,
        "dev25": dev_now, "z_dev": float(z_dev), "pctl": pctl,
        "rsi14": rsi14, "atr14": atr14, "atr_pct": float(atr_pct),
        "vol_z": float(vol_z_now), "adv20_jpy": adv20_jpy,
        "low5": low5, "high5": high5,
        "dip_days": dip_days, "rebound_days": rebound_days,
        "down_trend": bool(dn_trend), "up_trend": bool(up_trend),
        "reg_10pct_hit": reg_hit,
        "event_signature": event_signature,
        "last_date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
    }
