from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.events import earnings_new_entry_block
from utils.setup import detect_setup, atr14
from utils.rr_ev import build_trade_plan, rr_min_by_market
from utils.diversify import apply_diversification, DiversifyConfig
from utils.state import can_take_new_trades


@dataclass
class UniverseFilterConfig:
    price_min: float = 200.0
    price_max: float = 15000.0
    adv20_min_jpy: float = 200_000_000.0  # 200M
    atrp14_min: float = 1.5               # 1.5%
    earnings_block_bd: int = 3
    abnormal_move_pct: float = 25.0       # proxy for "abnormal"
    history_days: int = 260               # enough for 60d high, MA, ATR


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def fetch_history_daily(ticker: str, period: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return None
        for c in ("Open", "High", "Low", "Close"):
            if c not in df.columns:
                return None
        return df
    except Exception:
        return None


def _get_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if df is None or df.empty:
        return None, None
    tcol = "ticker" if "ticker" in df.columns else ("code" if "code" in df.columns else None)
    scol = "sector" if "sector" in df.columns else ("industry_big" if "industry_big" in df.columns else None)
    # LINE output cap (max 5)
    candidates = (candidates or [])[:MAX_LINE_CANDS]
    return tcol, scol


def pass_universe_filters(
    meta: Dict[str, object],
    hist: pd.DataFrame,
    today_date: date,
    cfg: UniverseFilterConfig,
) -> Tuple[bool, Dict[str, float], str]:
    close = hist["Close"].astype(float)
    vol = hist["Volume"].astype(float) if "Volume" in hist.columns else pd.Series(np.nan, index=hist.index)

    price = _safe_float(close.iloc[-1], np.nan)
    if not (np.isfinite(price) and cfg.price_min <= price <= cfg.price_max):
        return False, {}, "price_range"

    turnover = close * vol
    adv20 = _safe_float(turnover.rolling(20).mean().iloc[-1], np.nan)
    if not (np.isfinite(adv20) and adv20 >= cfg.adv20_min_jpy):
        return False, {}, "adv20<200M"

    atr = atr14(hist)
    atrp = _safe_float((atr / price) * 100.0, np.nan) if np.isfinite(atr) and price > 0 else np.nan
    if not (np.isfinite(atrp) and atrp >= cfg.atrp14_min):
        return False, {}, "atrp14<1.5%"

    # abnormal move proxy: any daily abs return > 25% (last ~12 sessions)
    ret = close.pct_change(fill_method=None).tail(12).abs() * 100.0
    if np.isfinite(ret).any() and float(np.nanmax(ret)) >= cfg.abnormal_move_pct:
        return False, {}, "abnormal_move"

    # earnings block
    ed = str(meta.get("earnings_date", "") or "").strip()
    if ed and earnings_new_entry_block(ed, today_date=today_date, window_bd=cfg.earnings_block_bd):
        return False, {}, "earnings_block"

    return True, {"price": float(price), "adv20": float(adv20), "atrp14": float(atrp)}, ""


MAX_LINE_CANDS = 5


def run_screening(
    universe_path: str,
    today_date: date,
    mkt: Dict[str, object],
    macro_on: bool,
    events: List[Dict[str, str]],  # reserved for future (e.g., event proximity per sector)
    state: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """
    ✅ 仕様準拠（Swing 1〜7日）
    - ユニバースfilter（価格/ADV/ATR%/異常/決算）
    - Setup A1/A2/B
    - Entry帯必須（現値IN禁止）
    - RR/AdjEV/Rday 必須
    - 分散：同一セクター最大2、相関>0.75禁止
    - Macro警戒：EV補正（adjev側）、レバ抑制（report側）、候補数制限（最大2）
    - NO-TRADE：地合いNG / 平均AdjEV<0.5 / Macro未達 / GU比率高
    """
    mkt_score = int(mkt.get("score", 50))
    state = state or {}
    if not can_take_new_trades(state, max_per_week=3):
        return {
            "no_trade": True,
            "no_trade_reasons": ["weekly_new_limit>=3"],
            "candidates": [],
            "stats": {"weekly_new": int(state.get("weekly_new", 0))},
        }

    rr_min = rr_min_by_market(mkt_score)
    max_final = 2 if macro_on else 3

    try:
        uni = pd.read_csv(universe_path)
    except Exception:
        return {"no_trade": True, "no_trade_reasons": ["universe_missing"], "candidates": [], "stats": {}}

    tcol, scol = _get_cols(uni)
    if tcol is None:
        return {"no_trade": True, "no_trade_reasons": ["universe_no_ticker_col"], "candidates": [], "stats": {}}

    cfg = UniverseFilterConfig()
    raw: List[Dict[str, object]] = []
    histories: Dict[str, pd.DataFrame] = {}

    for _, row in uni.iterrows():
        ticker = str(row.get(tcol, "")).strip()
        if not ticker:
            continue

        meta = {
            "name": str(row.get("name", ticker)),
            "sector": str(row.get(scol, "不明")) if scol else "不明",
            "earnings_date": str(row.get("earnings_date", "")).strip(),
        }

        hist = fetch_history_daily(ticker, period=f"{cfg.history_days}d")
        if hist is None or len(hist) < 140:
            continue

        ok, metrics, why = pass_universe_filters(meta, hist, today_date, cfg)
        if not ok:
            continue

        setup_res = detect_setup(hist)
        if setup_res.setup == "NONE":
            continue

        plan = build_trade_plan(
            hist=hist,
            setup=setup_res.setup,
            entry_low=setup_res.entry_low,
            entry_high=setup_res.entry_high,
            entry_mid=setup_res.entry_mid,
            stop_seed=setup_res.stop_seed,
            mkt_score=mkt_score,
            macro_on=macro_on,
        )
        if plan is None:
            continue

        # Filters
        if plan.rr < rr_min:
            continue
        if plan.adjev < 0.5:
            continue
        if plan.r_per_day < 0.5:
            continue

        histories[ticker] = hist
        raw.append(
            {
                "ticker": ticker,
                "name": meta["name"],
                "sector": meta["sector"],
                "setup": setup_res.setup,
                "entry_low": plan.entry_low,
                "entry_high": plan.entry_high,
                "entry_mid": plan.entry_mid,
                "sl": plan.sl,
                "tp1": plan.tp1,
                "tp2": plan.tp2,
                "rr": plan.rr,
                "adjev": plan.adjev,
                "r_per_day": plan.r_per_day,
                "expected_days": plan.expected_days,
                "gu": bool(setup_res.gu),
                "note": setup_res.note,
                "price": metrics.get("price"),
                "adv20": metrics.get("adv20"),
                "atrp14": metrics.get("atrp14"),
            }
        )

    raw.sort(key=lambda x: (float(x["adjev"]), float(x["r_per_day"]), float(x["rr"])), reverse=True)

    diversified = apply_diversification(raw, histories, DiversifyConfig(max_per_sector=2, corr_lookback=60, corr_threshold=0.75))
    diversified = diversified[:max_final]

    # NO-TRADE decision
    no_trade = bool(mkt.get("no_trade_core", False))
    reasons: List[str] = list(mkt.get("no_trade_reasons", [])) if isinstance(mkt.get("no_trade_reasons", []), list) else []

    # avg AdjEV
    if diversified:
        avg_adjev = float(np.mean([float(c["adjev"]) for c in diversified]))
        if avg_adjev < 0.5:
            no_trade = True
            reasons.append("avg_AdjEV<0.5")
    else:
        avg_adjev = 0.0

    # macro stricter: require avgAdjEV >= 0.65 and at least 1 candidate
    if macro_on:
        if len(diversified) == 0:
            no_trade = True
            reasons.append("macro_on & no_candidates")
        if avg_adjev < 0.65:
            no_trade = True
            reasons.append("macro_on & avgAdjEV<0.65")

    # GU ratio high (if most are GU, we don't want entries)
    if diversified:
        gu_ratio = float(np.mean([1.0 if c.get("gu") else 0.0 for c in diversified]))
        if gu_ratio >= 0.67:
            no_trade = True
            reasons.append("GU_ratio_high")
    else:
        gu_ratio = 0.0

    stats = {
        "mkt_score": mkt_score,
        "rr_min": rr_min,
        "raw_count": len(raw),
        "final_count": len(diversified),
        "avg_adjev": float(avg_adjev),
        "gu_ratio": float(gu_ratio),
        "macro_on": bool(macro_on),
        "max_final": int(max_final),
    }

        # LINE output cap
    candidates = (candidates or [])[:MAX_LINE_CANDS]

return {
        "no_trade": bool(no_trade),
        "no_trade_reasons": reasons,
        "candidates": diversified,
        "stats": stats,
    }