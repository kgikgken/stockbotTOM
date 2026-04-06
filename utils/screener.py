from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from utils.blackout import blackout_reason, load_blackouts_from_env
from utils.diversify import apply_corr_filter, apply_sector_cap
from utils.rr_ev import calc_ev, pass_thresholds
from utils.screen_logic import max_display, rs_comp_min_by_market, rs_pct_min_by_market
from utils.setup import build_setup_info, liquidity_filters
from utils.state import in_cooldown
from utils.util import (
    adx,
    atr14,
    bb_width_ratio,
    choppiness_index,
    clamp,
    download_history_bulk,
    down_up_volume_ratio,
    efficiency_ratio,
    env_float,
    env_int,
    is_abnormal_stock,
    percentile_rank,
    safe_float,
)


UNIVERSE_PATH = "universe_jpx.csv"


@dataclass
class UniverseRow:
    ticker: str
    name: str
    sector: str


def _ret_n(close: pd.Series, n: int) -> float:
    if close is None or close.empty or len(close) < n + 1:
        return float("nan")
    base = safe_float(close.iloc[-(n + 1)])
    last = safe_float(close.iloc[-1])
    if not (np.isfinite(base) and np.isfinite(last) and base > 0):
        return float("nan")
    return float((last / base - 1.0) * 100.0)


def _load_universe(path: str | Path = UNIVERSE_PATH) -> list[UniverseRow]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    if df.empty:
        return []

    cols = {str(c).strip().lower(): c for c in df.columns}
    ticker_col = cols.get("ticker") or cols.get("code") or list(df.columns)[0]
    name_col = cols.get("name") or cols.get("company")
    sector_col = cols.get("sector") or cols.get("industry")

    out: list[UniverseRow] = []
    for _, row in df.iterrows():
        ticker = str(row.get(ticker_col, "")).strip()
        if not ticker:
            continue
        name = str(row.get(name_col, "")).strip() if name_col else ""
        sector = str(row.get(sector_col, "Other")).strip() if sector_col else "Other"
        out.append(UniverseRow(ticker=ticker, name=name or ticker, sector=sector or "Other"))
    return out


def _trend_template_metrics(df: pd.DataFrame) -> Dict[str, float | bool]:
    if df is None or df.empty or len(df) < 260:
        return {}
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)
    atr_s = atr14(df)
    atr = safe_float(atr_s.iloc[-1])
    last = safe_float(close.iloc[-1])
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()
    m20 = safe_float(ma20.iloc[-1])
    m50 = safe_float(ma50.iloc[-1])
    m150 = safe_float(ma150.iloc[-1])
    m200 = safe_float(ma200.iloc[-1])
    high52 = safe_float(high.rolling(252).max().iloc[-1])
    low52 = safe_float(low.rolling(252).min().iloc[-1])
    dist_high = (high52 - last) / high52 * 100.0 if high52 > 0 else float("nan")
    from_low = (last / low52 - 1.0) * 100.0 if low52 > 0 else float("nan")
    bb_ratio = bb_width_ratio(close, ma_window=20, lookback=env_int("BB_RATIO_LOOKBACK", 60))
    atr_pct_series = atr_s / close.replace(0, np.nan) * 100.0
    atr_pct = safe_float(atr_pct_series.iloc[-1])
    atrp_ratio = safe_float(
        atr_pct_series.iloc[-1] / atr_pct_series.dropna().tail(max(20, env_int("BB_RATIO_LOOKBACK", 60))).median()
    )
    dry_ratio = down_up_volume_ratio(close, volume, lookback=env_int("VOL_DRY_LOOKBACK", 10))
    eff = safe_float(efficiency_ratio(close, 10).iloc[-1])
    chop = safe_float(choppiness_index(df, 14).iloc[-1])
    adxv = safe_float(adx(df, 14).iloc[-1])
    recent_high20 = safe_float(high.iloc[-21:-1].max())
    recent_low20 = safe_float(low.tail(20).min())
    pivot_gap = (recent_high20 - last) / last * 100.0 if last > 0 else float("nan")
    gap_lookback = env_int("GAP_ATR_LOOKBACK", 60)
    gap_mult = env_float("GAP_ATR_MULT", 1.0)
    prev_close = close.shift(1)
    gap_flag = (df["Open"].astype(float) - prev_close).abs() > gap_mult * atr_s
    gap_count = int(gap_flag.tail(gap_lookback).sum())

    score_parts = [
        0.16 if last > m20 else 0.0,
        0.16 if last > m50 else 0.0,
        0.16 if m20 > m50 else 0.0,
        0.12 if m50 > m150 else 0.0,
        0.12 if m150 > m200 else 0.0,
        0.12 if m200 >= safe_float(ma200.shift(20).iloc[-1], m200) else 0.0,
        0.08 if np.isfinite(dist_high) and dist_high <= env_float("TREND_MAX_DIST_52W_HIGH", 25.0) else 0.0,
        0.08 if np.isfinite(from_low) and from_low >= env_float("TREND_MIN_FROM_52W_LOW", 30.0) else 0.0,
    ]
    trend_score = float(sum(score_parts))

    contraction = np.nanmean(
        [
            clamp((1.25 - safe_float(bb_ratio, 9.0)) / 0.85, 0.0, 1.0),
            clamp((1.25 - safe_float(atrp_ratio, 9.0)) / 0.85, 0.0, 1.0),
            clamp((env_float("VOL_DRY_WARN", 1.35) - safe_float(dry_ratio, 9.0)) / 0.65, 0.0, 1.0),
        ]
    )

    return {
        "last": last,
        "atr": atr,
        "atr_pct": atr_pct,
        "atrp_ratio": atrp_ratio,
        "bb_ratio": bb_ratio,
        "dry_ratio": dry_ratio,
        "eff": eff,
        "chop": chop,
        "adx": adxv,
        "trend_score": trend_score,
        "contraction": float(contraction),
        "ret20": _ret_n(close, 20),
        "ret63": _ret_n(close, 63),
        "ret126": _ret_n(close, 126),
        "ret252": _ret_n(close, 252),
        "dist_high": dist_high,
        "from_low": from_low,
        "pivot_gap": pivot_gap,
        "recent_high20": recent_high20,
        "recent_low20": recent_low20,
        "above_ma50": bool(last > m50),
        "gap_count": gap_count,
    }


def _execution_score(last: float, entry: float, atr: float) -> float:
    if not (np.isfinite(last) and np.isfinite(entry) and np.isfinite(atr) and atr > 0):
        return 0.0
    return clamp(1.0 - abs(last - entry) / (1.6 * atr), 0.0, 1.0)


def _setup_threshold(setup: str) -> float:
    setup = str(setup)
    if setup == "A1-Strong":
        return env_float("TREND_MIN_A1S", 0.76)
    if setup == "A1":
        return env_float("TREND_MIN_A1", 0.70)
    if setup == "A2":
        return env_float("TREND_MIN_A2", 0.62)
    if setup == "B":
        return env_float("TREND_MIN_B", 0.72)
    return 0.70


def _candidate(metric: Dict, plan: Dict, *, ticker: str, name: str, sector: str, lane: str) -> Dict:
    last = safe_float(metric.get("last"))
    atr = safe_float(metric.get("atr"))
    rs_pct = safe_float(metric.get("rs_pct"), 0.0)
    liq_score = safe_float(metric.get("liq_score"), 0.0)
    trend_score = safe_float(metric.get("trend_score"), 0.0)
    contraction = safe_float(metric.get("contraction"), 0.0)
    dist_high = safe_float(metric.get("dist_high"), 99.0)
    breakout_ready = clamp((5.0 - max(safe_float(metric.get("pivot_gap"), 99.0), 0.0)) / 5.0, 0.0, 1.0)
    exec_score = _execution_score(last, safe_float(plan.get("entry")), atr)
    pth_score = clamp((env_float("LEADERS_MAX_DIST_52W_HIGH", 12.0) - dist_high) / max(env_float("LEADERS_MAX_DIST_52W_HIGH", 12.0), 1.0), 0.0, 1.0)
    gap_penalty = min(safe_float(metric.get("gap_count"), 0.0), 5.0) / 5.0

    if lane == "leaders":
        raw_score = 100.0 * (
            0.28 * (rs_pct / 100.0)
            + 0.22 * pth_score
            + 0.18 * trend_score
            + 0.14 * contraction
            + 0.10 * liq_score
            + 0.08 * breakout_ready
        ) - 6.0 * gap_penalty
        why = (
            f"RS {rs_pct:.0f} / 52WH {dist_high:.1f}% / C {contraction:.2f} / "
            f"ADV {safe_float(metric.get('adv20'))/1e8:.1f}億"
        )
    else:
        raw_score = 100.0 * (
            0.36 * trend_score
            + 0.24 * (rs_pct / 100.0)
            + 0.14 * liq_score
            + 0.14 * contraction
            + 0.12 * exec_score
        ) - 5.0 * gap_penalty
        why = (
            f"Trend {trend_score:.2f} / RS {rs_pct:.0f} / C {contraction:.2f} / "
            f"ADV {safe_float(metric.get('adv20'))/1e8:.1f}億"
        )

    out = {
        **plan,
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "lane": lane,
        "score": round(raw_score, 2),
        "why": why,
        "rs_pct": rs_pct,
        "trend_score": trend_score,
        "dist_high": dist_high,
        "from_low": safe_float(metric.get("from_low")),
        "liq_grade": metric.get("liq_grade"),
        "adv20": safe_float(metric.get("adv20")),
        "bb_ratio": safe_float(metric.get("bb_ratio")),
        "atrp_ratio": safe_float(metric.get("atrp_ratio")),
    }
    out["ev"] = calc_ev(out)
    return out


def _finalize(candidates: List[Dict], lane: str, ohlc_map: Dict[str, pd.DataFrame]) -> List[Dict]:
    sorted_cands = sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    sector_cap = env_int("LEADERS_SECTOR_CAP", 2) if lane == "leaders" else env_int("TREND_SECTOR_CAP", 2)
    sorted_cands = apply_sector_cap(sorted_cands, cap=sector_cap)
    corr_th = env_float("LEADERS_CORR_MAX", 0.87) if lane == "leaders" else env_float("TREND_CORR_MAX", 0.90)
    sorted_cands = apply_corr_filter(sorted_cands, ohlc_map, threshold=corr_th)
    return sorted_cands[: max_display(lane)]


def run_screen(
    *,
    today_str: str,
    today_date,
    mkt_score: int,
    delta3: int,
    macro_on: bool,
    state: Dict | None,
    universe_path: str | Path = UNIVERSE_PATH,
    ohlc_map_override: Dict[str, pd.DataFrame] | None = None,
) -> Dict[str, object]:
    universe = _load_universe(universe_path)
    tickers = [row.ticker for row in universe]
    ohlc_map = dict(ohlc_map_override or download_history_bulk(tickers, period="3y"))
    blackouts = load_blackouts_from_env()

    metrics: list[Dict] = []
    raw_rs_values: list[float] = []
    for row in universe:
        df = ohlc_map.get(row.ticker)
        if df is None or df.empty:
            continue
        if is_abnormal_stock(df):
            continue
        m = _trend_template_metrics(df)
        if not m:
            continue
        liq = liquidity_filters(df)
        m.update(
            {
                "ticker": row.ticker,
                "name": row.name,
                "sector": row.sector,
                "liq_ok": bool(liq.get("ok", False)),
                "liq_score": safe_float(liq.get("score"), 0.0),
                "liq_grade": str(liq.get("grade")),
                "adv20": safe_float(liq.get("adv20")),
            }
        )
        raw_rs = np.nanmean([
            0.20 * safe_float(m.get("ret20")),
            0.40 * safe_float(m.get("ret63")),
            0.25 * safe_float(m.get("ret126")),
            0.15 * safe_float(m.get("ret252")),
        ])
        m["rs_raw"] = float(raw_rs)
        metrics.append(m)
        raw_rs_values.append(float(raw_rs))

    if not metrics:
        return {
            "trend_candidates": [],
            "leader_candidates": [],
            "meta": {
                "data_warn": True,
                "data_ok": 0,
                "data_total": len(universe),
                "data_coverage": 0.0,
                "data_coverage_min": env_float("SCREEN_DATA_COVERAGE_MIN", 0.60),
                "breadth_regime": "unknown",
                "breadth_score": 0.0,
                "breadth_force_no_trade": True,
                "mkt_score_eff": mkt_score,
            },
            "ohlc_map": ohlc_map,
        }

    median_rs = float(np.nanmedian(raw_rs_values)) if raw_rs_values else 0.0
    for m in metrics:
        m["rs_pct"] = percentile_rank(raw_rs_values, safe_float(m.get("rs_raw")))
        m["rs_comp"] = safe_float(m.get("rs_raw")) - median_rs

    breadth = 100.0 * np.mean([1.0 if bool(m.get("above_ma50")) else 0.0 for m in metrics])
    if breadth >= 62:
        breadth_regime = "strong"
    elif breadth >= 45:
        breadth_regime = "normal"
    else:
        breadth_regime = "weak"
    breadth_force = breadth < env_float("BREADTH_FORCE_NO_TRADE_LT", 35.0) and int(mkt_score) < 50

    trend_candidates: list[Dict] = []
    leader_candidates: list[Dict] = []

    trend_tt_min = env_float("TREND_TEMPLATE_MIN_SCORE", 0.70)
    leader_tt_min = env_float("LEADERS_MIN_TREND_SCORE", 0.78)
    leader_mkt_min = env_int("LEADERS_MARKET_SCORE_MIN", 55)
    for m in metrics:
        ticker = str(m.get("ticker"))
        if blackout_reason(ticker, blackouts):
            continue
        if state and in_cooldown(state, ticker):
            continue
        if not bool(m.get("liq_ok")):
            continue

        df = ohlc_map.get(ticker)
        if df is None or df.empty:
            continue
        plans_trend = build_setup_info(df, lane="trend")
        plans_leaders = build_setup_info(df, lane="leaders")

        # Existing trend lane.
        if safe_float(m.get("trend_score")) >= trend_tt_min and safe_float(m.get("rs_pct")) >= rs_pct_min_by_market(mkt_score) and safe_float(m.get("rs_comp")) >= rs_comp_min_by_market(mkt_score):
            for plan in plans_trend:
                if safe_float(m.get("trend_score")) < _setup_threshold(str(plan.get("setup"))):
                    continue
                cand = _candidate(m, plan, ticker=ticker, name=str(m.get("name")), sector=str(m.get("sector")), lane="trend")
                if pass_thresholds(cand):
                    trend_candidates.append(cand)
                    break

        # Research-based leaders lane.
        if int(mkt_score) >= leader_mkt_min and not macro_on:
            leader_gate = all(
                [
                    safe_float(m.get("trend_score")) >= leader_tt_min,
                    safe_float(m.get("rs_pct")) >= env_float("LEADERS_MIN_RS_PCT", 80.0),
                    safe_float(m.get("dist_high")) <= env_float("LEADERS_MAX_DIST_52W_HIGH", 12.0),
                    safe_float(m.get("from_low")) >= env_float("LEADERS_MIN_FROM_52W_LOW", 35.0),
                    safe_float(m.get("bb_ratio")) <= env_float("LEADERS_MAX_BB_RATIO", 1.05),
                    safe_float(m.get("atrp_ratio")) <= env_float("LEADERS_MAX_ATRP_RATIO", 1.05),
                    safe_float(m.get("liq_score")) >= env_float("LEADERS_MIN_LIQ_SCORE", 0.65),
                ]
            )
            if leader_gate:
                for plan in plans_leaders:
                    cand = _candidate(m, plan, ticker=ticker, name=str(m.get("name")), sector=str(m.get("sector")), lane="leaders")
                    if pass_thresholds(cand):
                        leader_candidates.append(cand)
                        break

    trend_candidates = _finalize(trend_candidates, "trend", ohlc_map)
    leader_candidates = _finalize(leader_candidates, "leaders", ohlc_map)

    coverage = len(metrics) / max(1, len(universe))
    meta = {
        "data_warn": coverage < env_float("SCREEN_DATA_COVERAGE_MIN", 0.60),
        "data_ok": len(metrics),
        "data_total": len(universe),
        "data_coverage": coverage,
        "data_coverage_min": env_float("SCREEN_DATA_COVERAGE_MIN", 0.60),
        "breadth_regime": breadth_regime,
        "breadth_score": breadth,
        "breadth_force_no_trade": breadth_force,
        "mkt_score_eff": mkt_score,
        "trend_count": len(trend_candidates),
        "leader_count": len(leader_candidates),
    }
    return {
        "trend_candidates": trend_candidates,
        "leader_candidates": leader_candidates,
        "meta": meta,
        "ohlc_map": ohlc_map,
    }
