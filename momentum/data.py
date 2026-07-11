"""モメンタム用データ層 — 実データ(yfinance)は歪み系と共有のロジックを再利用。
DRYRUN用の合成データはモメンタム3状態(A/B/C)をテストできるよう専用に生成する。
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from mispricing.data import fetch_ohlcv as _fetch_ohlcv_real  # 汎用yfinance取得ロジックを再利用


def fetch_ohlcv(tickers: List[str], history_days: int, dryrun: bool = False) -> Tuple[Dict[str, pd.DataFrame], dict]:
    if dryrun:
        return _synthetic(tickers)
    out, meta = _fetch_ohlcv_real(tickers, history_days, dryrun=False)
    # ★指示⑬: 取得失敗銘柄を記録用に付与(原因の細分類はできないため一括「欠落」扱い)
    missing = [t for t in tickers if t not in out]
    meta["fetch_failures"] = missing
    return out, meta


def fetch_regime_series(cfg) -> Tuple[pd.Series | None, str]:
    """TOPIX(1306.T ETF、フォールバックで日経225)の終値系列を取得。"""
    if cfg.dryrun:
        rng = np.random.default_rng(7)
        end = pd.Timestamp.today().normalize()
        if end.dayofweek >= 5:
            end -= pd.tseries.offsets.BDay(1)
        n = max(cfg.regime_mom_days + 30, 300)
        idx = pd.bdate_range(end=end, periods=n)
        import os
        risk_off = str(os.getenv("MOM_DRYRUN_REGIME", "on")).strip().lower() == "off"
        t = np.arange(n)
        slope = -0.0016 if risk_off else 0.0016  # 決定的トレンド(ノイズは点描のみ)
        trend = 2800.0 * np.exp(slope * t)
        close = pd.Series(trend * (1 + rng.normal(0, 0.003, n)), index=idx)
        return close, cfg.regime_ticker_primary + "(DRYRUN合成)"

    import yfinance as yf
    for tk in (cfg.regime_ticker_primary, cfg.regime_ticker_fallback):
        try:
            h = yf.download(tk, period=f"{cfg.regime_mom_days + 60}d", interval="1d",
                            progress=False, auto_adjust=False)
            c = h["Close"].dropna()
            if hasattr(c, "columns"):
                c = c.iloc[:, 0]
            if len(c) >= cfg.regime_mom_days:
                return c, tk
        except Exception:
            continue
    return None, "取得失敗"


# ---------------------------------------------------------------- synthetic

def _mk_series_momentum(seed: int, n: int, mode: str) -> pd.DataFrame:
    """mode: state_a(既に流入・押し目) / state_b(初動VCPブレイク) / state_c(流出) / normal"""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    if end.dayofweek >= 5:
        end -= pd.tseries.offsets.BDay(1)
    idx = pd.bdate_range(end=end, periods=n)

    if mode == "state_a":
        # 決定的な滑らかトレンド(ノイズは点描のみ)→ 自然な乱歩による偽の山を防ぐ
        t = np.arange(n)
        trend = 1000.0 * np.exp(0.0011 * t)
        close = trend * (1 + rng.normal(0, 0.004, n))
        v = rng.integers(200_000, 900_000, n).astype(float)
        base = float(close[-9])
        # 直近9日: 7日かけて下押し→直近2日ではっきり反発(bounce_confirmedの要件を満たす形)
        decline = np.linspace(1.0, 0.975, 7)
        bounce = np.array([0.975 * 1.006, 0.975 * 1.012])
        close[-9:] = base * np.concatenate([decline, bounce])
    elif mode == "state_b":
        drift, noise = 0.0004, 0.013
        ret = rng.normal(drift, noise, n)
        close = 1000.0 * np.exp(np.cumsum(ret))
        v = rng.integers(200_000, 900_000, n).astype(float)
    elif mode == "state_c":
        drift, noise = 0.0011, 0.015
        ret = rng.normal(drift, noise, n)
        close = 1000.0 * np.exp(np.cumsum(ret))
        v = rng.integers(200_000, 900_000, n).astype(float)
    elif mode == "tob_pattern":
        drift, noise = 0.0002, 0.016
        ret = rng.normal(drift, noise, n)
        close = 1000.0 * np.exp(np.cumsum(ret))
        v = rng.integers(200_000, 900_000, n).astype(float)
    else:
        drift, noise = 0.0002, 0.017
        ret = rng.normal(drift, noise, n)
        close = 1000.0 * np.exp(np.cumsum(ret))
        v = rng.integers(200_000, 900_000, n).astype(float)

    if mode == "state_b":
        # 直近21日でレンジを圧縮(VCP) → 最終日に出来高を伴い明確にブレイク
        base = float(close[-22])
        close[-21:-1] = base * (1.0 + np.linspace(-0.010, 0.012, 20))
        close[-1] = base * 1.048
        vbase = float(v[-21:-1].mean())
        v[-1] = vbase * 2.2  # 出来高スパイク(平均比1.5倍を確実に超える)
    elif mode == "state_c":
        # 過去は上昇していたが、直近55日で明確に崩れ50日線を割る
        base = float(close[-56])
        decline = base * np.exp(np.cumsum(rng.normal(-0.0075, 0.013, 55)))
        close[-55:] = decline
    elif mode == "tob_pattern":
        # 40営業日前に単日+28%ジャンプ(TOB発表想定)→ 以降は買付価格近辺に張り付き低ボラ
        jump_day = -40
        pre_base = float(close[jump_day - 1])
        close[jump_day] = pre_base * 1.28
        pinned = float(close[jump_day])
        close[jump_day + 1:] = pinned * (1 + np.linspace(0, 0, n - (n + jump_day + 1)))
        v[jump_day] = float(v[jump_day - 5:jump_day].mean()) * 5.0

    # 高値・安値のノイズ幅(通常0.006)。state_bは圧縮期間中に段階的に収縮させ、
    # 直近10日 vs 直近20日で「収縮した」と判定できる形にする(実際のVCPは段階的に締まる)。
    hl_scale = np.full(n, 0.006)
    if mode == "state_b":
        hl_scale[-21:-2] = np.linspace(0.0055, 0.0010, 19)  # 序盤は緩め→終盤にかけて強く収縮
        hl_scale[-2] = 0.0010
    if mode == "tob_pattern":
        hl_scale[-40 + 1:] = 0.0008  # 買付価格近辺への張り付き(ほぼ動かない)

    o = close * (1 + rng.normal(0, 0.004, n))
    h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 1, n)) * hl_scale)
    l = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 1, n)) * hl_scale)
    if mode == "state_a":
        # 最終日ははっきり「安値圏で寄り付き高値圏で引ける」反発陽線の形に固定する
        o[-1] = close[-1] * 0.994
        l[-1] = close[-1] * 0.990
        h[-1] = close[-1] * 1.002
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": close, "Volume": v}, index=idx)


def _synthetic(tickers: List[str]) -> Tuple[Dict[str, pd.DataFrame], dict]:
    # 0,1: 同一セクター内にstate_aを2つ(セクター分散キャップの検証用) / 2,3: 同様にstate_bを2つ
    # 4: state_c / 5+: normal
    modes = ["state_a", "state_a", "state_b", "state_b", "state_c", "tob_pattern", "tob_pattern"] + ["normal"] * 5
    out: Dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers[:60]):
        out[t] = _mk_series_momentum(seed=142 + i, n=420,
                                     mode=modes[i % len(modes)] if i < 12 else "normal")
    meta = {
        "data_total": len(tickers[:60]), "data_ok": len(out),
        "data_coverage": 1.0 if tickers else 0.0,
        "source": "SYNTHETIC (DRYRUN・実データではない)",
        "fetched_at": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M JST"),
        "dryrun": True,
    }
    return out, meta
