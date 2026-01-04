from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.setup import Config
from utils.util import (
    clamp,
    jst_today_date,
    parse_date_yyyy_mm_dd,
    safe_read_csv,
)
from utils.market import get_market_state, MarketState
from utils.sector import compute_sector_ranking_5d
from utils.features import sma, atr, rsi
from utils.entry import compute_entry_pullback, compute_entry_breakout, EntryPlan
from utils.rr_ev import compute_rr_ev, RRResult
from utils.diversify import correlation_filter
from utils.events import is_event_eve


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Candidate:
    ticker: str
    name: str
    sector: str
    setup_type: str
    in_low: float
    in_high: float
    in_center: float
    stop: float
    tp1: float
    tp2: float
    rr: float
    pwin: float
    ev: float
    adj_ev: float
    exp_days: float
    r_per_day: float
    gu_flag: bool
    monitor_only: bool
    in_dist_atr: float
    note: str = ""


@dataclass
class ScreeningResult:
    date: object  # datetime.date
    market: MarketState
    no_trade: bool
    no_trade_reasons: List[str]
    sector_ranks: List[Dict[str, object]]
    picks: List[Candidate]
    watchlist: List[Candidate]


# ============================================================
# Internal Helpers
# ============================================================

def _download_ohlcv(ticker: str, days: int = 240) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=f"{days}d",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    return df.rename(columns=str.title)


def _adv20_jpy(df: pd.DataFrame) -> float:
    if df.empty or "Close" not in df.columns or "Volume" not in df.columns:
        return float("nan")
    turnover = (df["Volume"].astype(float) * df["Close"].astype(float)).rolling(20).mean()
    return float(turnover.iloc[-1]) if len(turnover) else float("nan")


def _has_earnings_block(row: pd.Series, today, cfg: Config) -> bool:
    for col in ["earnings_date", "earnings", "report_date"]:
        if col in row.index:
            d = parse_date_yyyy_mm_dd(row[col])
            if d is None:
                continue
            if abs((d - today).days) <= cfg.EARNINGS_EXCLUDE_DAYS:
                return True
    return False


def _normalize_sector_signal(sector_rank: int, cfg: Config) -> float:
    # rank1=+1.0, rank5=+0.2, >5 negative
    if sector_rank <= cfg.SECTOR_TOP_K:
        return float(1.0 - (sector_rank - 1) * 0.2)
    return -0.2


def _compute_setup_and_entry(
    df: pd.DataFrame,
    cfg: Config,
) -> Tuple[Optional[str], Optional[EntryPlan], Dict[str, float], float, float]:
    """
    Returns:
      setup_type, entry_plan, feature_signals, atr_last, prev_close
    """
    if df.empty or len(df) < 80:
        return None, None, {}, float("nan"), float("nan")

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float)

    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    atr14 = atr(high, low, close, 14)

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
    last_open = float(open_.iloc[-1])

    sma20_last = float(sma20.iloc[-1])
    sma50_last = float(sma50.iloc[-1])
    atr_last = float(atr14.iloc[-1])

    if not (np.isfinite(sma20_last) and np.isfinite(sma50_last) and np.isfinite(atr_last) and atr_last > 0):
        return None, None, {}, atr_last, prev_close

    # --- Trend strength (normalized by ATR) ---
    sma20_slope = (
        (sma20.iloc[-1] - sma20.iloc[-6]) / atr_last
        if len(sma20) >= 6 else 0.0
    )

    # ========================================================
    # Setup A: Trend Pullback
    # ========================================================
    is_trend = (last_close > sma20_last) and (sma20_last > sma50_last) and (sma20_slope > 0)
    pullback_ok = abs(last_close - sma20_last) <= 0.8 * atr_last
    setup_pullback = is_trend and pullback_ok

    # ========================================================
    # Setup B: Breakout
    # ========================================================
    hh20 = float(close.rolling(20).max().iloc[-2]) if len(close) >= 22 else float("nan")
    vol20 = float(vol.rolling(20).mean().iloc[-1])
    setup_breakout = (
        np.isfinite(hh20)
        and last_close > hh20
        and vol20 > 0
        and float(vol.iloc[-1]) >= cfg.VOL_BREAKOUT_MULT * vol20
    )

    entry_plan: Optional[EntryPlan] = None
    setup_type: Optional[str] = None
    break_line = hh20 if np.isfinite(hh20) else last_close

    if setup_pullback:
        setup_type = "PULLBACK"
        entry_plan = compute_entry_pullback(
            sma20_last, atr_last, last_close, last_open, prev_close, cfg
        )
    elif setup_breakout:
        setup_type = "BREAKOUT"
        entry_plan = compute_entry_breakout(
            break_line, atr_last, last_close, last_open, prev_close, cfg
        )
    else:
        return None, None, {}, atr_last, prev_close

    # ========================================================
    # Feature signals for Pwin proxy
    # ========================================================
    trend_sig = float(np.clip(sma20_slope / 2.0, -1.0, 1.0))

    # Volume quality
    if setup_type == "BREAKOUT" and vol20 > 0:
        vol_sig = float(np.clip((float(vol.iloc[-1]) / vol20 - 1.0), -1.0, 1.5))
    elif setup_type == "PULLBACK" and vol20 > 0:
        vol_sig = float(np.clip((1.0 - float(vol.iloc[-1]) / vol20), -1.0, 1.0))
    else:
        vol_sig = 0.0

    gaprisk = -1.0 if entry_plan.gu_flag else 0.5

    feats = {
        "trend": trend_sig,
        "volq": vol_sig,
        "gaprisk": gaprisk,
        # rs / sector は後で付与
    }

    return setup_type, entry_plan, feats, atr_last, prev_close


# ============================================================
# Main Screening
# ============================================================

def run_screening(
    universe_path: str,
    positions_path: str,
    events,
    cfg: Config,
) -> ScreeningResult:

    today = jst_today_date()

    # --------------------------------------------------------
    # Market State (TOPIX)
    # --------------------------------------------------------
    market = get_market_state(
        cfg.TOPIX_TICKER,
        cfg.MKT_SMA_FAST,
        cfg.MKT_SMA_SLOW,
        cfg.MKT_SMA_RISK,
    )

    # --------------------------------------------------------
    # Load Universe
    # --------------------------------------------------------
    universe = safe_read_csv(universe_path)
    if universe.empty:
        universe = pd.DataFrame(columns=["ticker", "name", "sector", "earnings_date"])

    # normalize columns
    cols = {c.lower(): c for c in universe.columns}
    if "ticker" not in cols and "code" in cols:
        universe = universe.rename(columns={cols["code"]: "ticker"})
    if "name" not in cols:
        universe["name"] = ""
    if "sector" not in cols:
        universe["sector"] = "NA"

    universe["ticker"] = universe["ticker"].astype(str)

    # --------------------------------------------------------
    # Sector Ranking (5d)
    # --------------------------------------------------------
    sector_ranks, _ = compute_sector_ranking_5d(
        universe[["ticker", "sector"]].copy(),
        max_names_per_sector=25,
    )
    sector_rank_map = {s.sector: s.rank for s in sector_ranks}

    # --------------------------------------------------------
    # NO-TRADE pre-check (Market)
    # --------------------------------------------------------
    no_trade_reasons: List[str] = []

    if market.market_score < cfg.NO_TRADE_MKT_SCORE_LT:
        no_trade_reasons.append(f"MarketScore<{cfg.NO_TRADE_MKT_SCORE_LT:.0f}")

    if (
        market.momentum_3d <= cfg.NO_TRADE_MOMENTUM_3D_LTE
        and market.market_score < cfg.NO_TRADE_MKT_SCORE_LT_WHEN_MOM_DOWN
    ):
        no_trade_reasons.append(
            f"ΔMarketScore_3d<={cfg.NO_TRADE_MOMENTUM_3D_LTE:.0f} & MarketScore<{cfg.NO_TRADE_MKT_SCORE_LT_WHEN_MOM_DOWN:.0f}"
        )

    is_eve = is_event_eve(events, pd.Timestamp(today))

    # --------------------------------------------------------
    # Pre-fetch TOPIX for RS
    # --------------------------------------------------------
    idx_df = yf.download(
        cfg.TOPIX_TICKER,
        period="120d",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    idx_close = (
        idx_df["Close"].dropna().astype(float)
        if idx_df is not None and not idx_df.empty
        else pd.Series(dtype=float)
    )
    idx_ret20 = (
        float(idx_close.iloc[-1] / idx_close.iloc[-21] - 1.0)
        if len(idx_close) >= 21 else 0.0
    )

    candidates: List[Candidate] = []
    watchlist: List[Candidate] = []

    gu_count = 0
    evaluated = 0

    # --------------------------------------------------------
    # Iterate Universe
    # --------------------------------------------------------
    for _, row in universe.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker or ticker.lower() in ("nan", "none"):
            continue

        name = str(row.get("name", "") or "")
        sector = str(row.get("sector", "NA") or "NA")

        # Earnings block
        if _has_earnings_block(row, today, cfg):
            continue

        # Sector filter
        s_rank = sector_rank_map.get(sector, 999)
        if s_rank > cfg.SECTOR_TOP_K:
            continue

        df = _download_ohlcv(ticker)
        if df.empty:
            continue

        last_close = float(df["Close"].iloc[-1])

        # Price filter
        if not (cfg.PRICE_MIN <= last_close <= cfg.PRICE_MAX):
            continue

        # Liquidity
        adv20 = _adv20_jpy(df)
        if not (np.isfinite(adv20) and adv20 >= cfg.ADV20_MIN_JPY):
            continue

        # Setup / Entry
        setup_type, entry_plan, feats, atr_last, _ = _compute_setup_and_entry(df, cfg)
        if setup_type is None or entry_plan is None or not np.isfinite(atr_last):
            continue

        # ATR%
        if (atr_last / last_close) < cfg.ATRPCT_MIN:
            continue

        evaluated += 1
        if entry_plan.gu_flag:
            gu_count += 1

        # RS vs TOPIX
        close = df["Close"].astype(float)
        stock_ret20 = (
            float(close.iloc[-1] / close.iloc[-21] - 1.0)
            if len(close) >= 21 else 0.0
        )
        rs = stock_ret20 - idx_ret20
        if rs < 0:
            continue

        feats["rs"] = float(np.clip(rs * 5.0, -1.0, 1.0))
        feats["sector"] = _normalize_sector_signal(s_rank, cfg)

        # RR / EV
        break_line = (
            float(close.rolling(20).max().iloc[-2])
            if len(close) >= 22 else last_close
        )

        rrres: RRResult = compute_rr_ev(
            setup_type=setup_type,
            in_center=entry_plan.in_center,
            in_low=entry_plan.in_low,
            break_line=break_line,
            atr=atr_last,
            market_score=market.market_score,
            momentum_3d=market.momentum_3d,
            is_event_eve=is_eve,
            features=feats,
            cfg=cfg,
        )

        # Filters
        if rrres.rr < cfg.RR_MIN:
            continue

        ev_min = (
            cfg.EV_MIN_R_NEUTRAL
            if 45 <= market.market_score < 60
            else cfg.EV_MIN_R
        )
        if rrres.ev < ev_min:
            continue

        if rrres.exp_days > cfg.EXP_DAYS_MAX:
            continue

        if rrres.r_per_day < cfg.R_PER_DAY_MIN:
            continue

        cand = Candidate(
            ticker=ticker,
            name=name,
            sector=sector,
            setup_type=setup_type,
            in_low=entry_plan.in_low,
            in_high=entry_plan.in_high,
            in_center=entry_plan.in_center,
            stop=rrres.stop,
            tp1=rrres.tp1,
            tp2=rrres.tp2,
            rr=rrres.rr,
            pwin=rrres.pwin,
            ev=rrres.ev,
            adj_ev=rrres.adj_ev,
            exp_days=rrres.exp_days,
            r_per_day=rrres.r_per_day,
            gu_flag=entry_plan.gu_flag,
            monitor_only=entry_plan.monitor_only,
            in_dist_atr=entry_plan.in_dist_atr,
            note=("EVE" if is_eve else ""),
        )

        if cand.gu_flag or cand.monitor_only:
            watchlist.append(cand)
        else:
            candidates.append(cand)

    # --------------------------------------------------------
    # NO-TRADE post-checks
    # --------------------------------------------------------
    no_trade = len(no_trade_reasons) > 0

    if evaluated > 0 and (gu_count / evaluated) >= cfg.NO_TRADE_GU_RATIO_GTE:
        no_trade_reasons.append(f"GU_ratio>={cfg.NO_TRADE_GU_RATIO_GTE:.0%}")
        no_trade = True

    if candidates:
        top_adj = sorted([c.adj_ev for c in candidates], reverse=True)[: min(10, len(candidates))]
        avg_adj = float(np.mean(top_adj)) if top_adj else -999.0
        if avg_adj < cfg.NO_TRADE_AVG_ADJEV_LT_R:
            no_trade_reasons.append(f"avgAdjEV<{cfg.NO_TRADE_AVG_ADJEV_LT_R:.2f}")
            no_trade = True

    # --------------------------------------------------------
    # Return if NO-TRADE
    # --------------------------------------------------------
    if no_trade:
        return ScreeningResult(
            date=today,
            market=market,
            no_trade=True,
            no_trade_reasons=no_trade_reasons,
            sector_ranks=[
                {"sector": s.sector, "rank": s.rank, "ret5": s.ret5}
                for s in sector_ranks[: cfg.SECTOR_TOP_K]
            ],
            picks=[],
            watchlist=sorted(watchlist, key=lambda x: x.adj_ev, reverse=True)[: cfg.MAX_WATCHLIST],
        )

    # --------------------------------------------------------
    # Diversification (Correlation + Sector cap)
    # --------------------------------------------------------
    cand_df = pd.DataFrame([
        {"ticker": c.ticker, "sector": c.sector, "adj_ev": c.adj_ev}
        for c in candidates
    ])

    tickers = cand_df["ticker"].tolist()

    price_df = yf.download(
        tickers,
        period="120d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    rets = pd.DataFrame()
    if price_df is not None and not price_df.empty:
        for t in tickers:
            try:
                if isinstance(price_df.columns, pd.MultiIndex):
                    c = price_df[(t, "Close")].dropna().astype(float)
                else:
                    c = price_df["Close"].dropna().astype(float)
                rets[t] = c.pct_change()
            except Exception:
                continue
        rets = rets.dropna(how="all").tail(cfg.CORR_WINDOW)

    picked = correlation_filter(cand_df, rets, cfg)
    picked_set = set(picked)

    picks = sorted(
        [c for c in candidates if c.ticker in picked_set],
        key=lambda x: x.adj_ev,
        reverse=True,
    )[: cfg.MAX_FINAL_STOCKS]

    # --------------------------------------------------------
    # Final Result
    # --------------------------------------------------------
    return ScreeningResult(
        date=today,
        market=market,
        no_trade=False,
        no_trade_reasons=[],
        sector_ranks=[
            {"sector": s.sector, "rank": s.rank, "ret5": s.ret5}
            for s in sector_ranks[: cfg.SECTOR_TOP_K]
        ],
        picks=picks,
        watchlist=sorted(watchlist, key=lambda x: x.adj_ev, reverse=True)[: cfg.MAX_WATCHLIST],
    )