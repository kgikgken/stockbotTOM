from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.market import MarketContext
from utils.events import EventContext
from utils.features import calc_features
from utils.setup import decide_setup
from utils.entry import build_entry_plan
from utils.rr_ev import evaluate
from utils.diversify import apply_diversify
from utils.sector import get_sector_rank_map

UNIVERSE_PATH = "universe_jpx.csv"

# v2.0 defaults
ADV20_MIN = 200_000_000     # 200M
ADV20_MIN_FALLBACK = 100_000_000
ATR_PCT_MIN = 1.5
ATR_PCT_MAX = 6.0
PRICE_MIN = 200
PRICE_MAX = 15000
EARNINGS_EXCLUDE_DAYS = 3
MAX_PICKS = 5
MAX_WATCH = 10
SECTOR_TOP_N = 5

def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def fetch_history(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.4)
    return None

def _parse_earnings_date(row) -> Optional[pd.Timestamp]:
    d = row.get("earnings_date", None)
    if d is None:
        return None
    try:
        ts = pd.to_datetime(d, errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None

def _is_earnings_near(row, today_date) -> bool:
    ts = _parse_earnings_date(row)
    if ts is None:
        return False
    try:
        delta = abs((ts.date() - today_date).days)
        return delta <= EARNINGS_EXCLUDE_DAYS
    except Exception:
        return False

def run_swing_screen(today_date, mkt: MarketContext, ev_ctx: EventContext) -> Dict:
    # universe
    if not os.path.exists(UNIVERSE_PATH):
        return {"trade_allowed": mkt.trade_allowed, "no_trade_reasons": mkt.no_trade_reasons, "picks": [], "watch": []}
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return {"trade_allowed": mkt.trade_allowed, "no_trade_reasons": mkt.no_trade_reasons, "picks": [], "watch": []}

    # ticker col
    if "ticker" in uni.columns:
        t_col = "ticker"
    elif "code" in uni.columns:
        t_col = "code"
    else:
        return {"trade_allowed": mkt.trade_allowed, "no_trade_reasons": mkt.no_trade_reasons, "picks": [], "watch": []}

    # sector col
    if "sector" in uni.columns:
        sec_col = "sector"
    elif "industry_big" in uni.columns:
        sec_col = "industry_big"
    else:
        sec_col = None

    # sector ranks
    sector_tops, sector_rank_map, top_sector_names = get_sector_rank_map(top_n=SECTOR_TOP_N)

    # index hist for RS
    index_hist = None
    try:
        index_hist = yf.Ticker("^TOPX").history(period="260d", auto_adjust=True)
    except Exception:
        index_hist = None

    candidates: List[dict] = []
    watch: List[dict] = []

    # market NO-TRADEなら候補品質判定用にだけ走らせる（監視も作る）。ただし最終出力は新規見送り。
    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get(sec_col, "不明")) if sec_col else "不明"
        sector_rank = int(sector_rank_map.get(sector, 0))

        # セクター上位5縛り
        if top_sector_names and sector not in top_sector_names:
            continue

        hist = fetch_history(ticker, period="260d")
        if hist is None or len(hist) < 120:
            continue

        feat = calc_features(hist, index_hist=index_hist)

        # Universe filters
        if not (PRICE_MIN <= feat.close <= PRICE_MAX):
            continue

        # ATR%
        if not (np.isfinite(feat.atr_pct) and ATR_PCT_MIN <= feat.atr_pct <= ATR_PCT_MAX):
            continue

        # ADV20（まず200M。少なすぎる場合のfallbackは後段で）
        if not (np.isfinite(feat.adv20) and feat.adv20 >= ADV20_MIN):
            # 100M未満は即除外、100M〜200Mは watch に回す余地
            if np.isfinite(feat.adv20) and feat.adv20 >= ADV20_MIN_FALLBACK:
                # 監視（流動性弱）
                watch.append(dict(
                    ticker=ticker, name=name, sector=sector, setup="-",
                    rr=0.0, r_per_day=0.0, gu=False, action_jp="監視のみ",
                    reject_reason="流動性弱(ADV20<200M)"
                ))
            continue

        # setup
        s = decide_setup(feat)
        if s.setup_type is None:
            continue

        # earnings near => 新規禁止（監視はOK）
        earnings_near = _is_earnings_near(row, today_date)

        entry = build_entry_plan(feat, s.setup_type)

        # 追いかけ禁止：WATCH_ONLYは本命に入れない（監視へ）
        if entry.action == "WATCH_ONLY":
            w = dict(
                ticker=ticker, name=name, sector=sector,
                setup=s.setup_type, rr=0.0, r_per_day=0.0,
                gu=entry.gu_flag, action_jp=entry.action_jp,
                reject_reason="追いかけ禁止/乖離",
            )
            watch.append(w)
            continue

        # evaluate
        evp = evaluate(s.setup_type, feat, entry, sector_rank=sector_rank, mkt=mkt, ev_ctx=ev_ctx)

        if evp.reject_reason:
            watch.append(dict(
                ticker=ticker, name=name, sector=sector, setup=s.setup_type,
                rr=float(evp.rr), r_per_day=float(evp.r_per_day),
                gu=entry.gu_flag, action_jp=entry.action_jp,
                reject_reason=evp.reject_reason
            ))
            continue

        # earnings near => 監視へ（新規禁止）
        if earnings_near:
            watch.append(dict(
                ticker=ticker, name=name, sector=sector, setup=s.setup_type,
                rr=float(evp.rr), r_per_day=float(evp.r_per_day),
                gu=entry.gu_flag, action_jp="監視のみ",
                reject_reason="決算近接(±3営業日)",
            ))
            continue

        candidates.append(dict(
            ticker=ticker,
            name=name,
            sector=sector,
            setup=s.setup_type,
            price=float(feat.close),
            atr=float(feat.atr14),
            gu=bool(entry.gu_flag),
            in_center=float(entry.in_center),
            in_low=float(entry.in_low),
            in_high=float(entry.in_high),
            stop=float(evp.stop),
            tp1=float(evp.tp1),
            tp2=float(evp.tp2),
            rr=float(evp.rr),
            expected_days=float(evp.expected_days),
            r_per_day=float(evp.r_per_day),
            pwin=float(evp.pwin),
            ev=float(evp.ev),
            adj_ev=float(evp.adj_ev),
            action=entry.action,
            action_jp=entry.action_jp,
            hist_close=hist["Close"].astype(float),
        ))

    # 候補品質による NO-TRADE 二次判定
    no_trade_reasons = list(mkt.no_trade_reasons)
    trade_allowed = bool(mkt.trade_allowed)

    if candidates:
        avg_adj = float(np.mean([c["adj_ev"] for c in candidates])) if candidates else 0.0
        if avg_adj < 0.3:
            trade_allowed = False
            no_trade_reasons.append("平均AdjustedEV<0.3")
    else:
        # 候補ゼロなら新規しない
        trade_allowed = False
        no_trade_reasons.append("候補ゼロ")

    # GU比率
    if candidates:
        gu_ratio = float(np.mean([1.0 if c["gu"] else 0.0 for c in candidates]))
        if gu_ratio >= 0.60:
            trade_allowed = False
            no_trade_reasons.append("GU比率>=60%")

    # ソート（AdjEV -> R/day -> RR）
    candidates.sort(key=lambda x: (x["adj_ev"], x["r_per_day"], x["rr"]), reverse=True)

    # 分散
    div = apply_diversify(candidates, max_picks=MAX_PICKS, sector_max=2, corr_max=0.75)
    picks = div.picks
    watch.extend(div.watch_added)

    # watch整形（必要項目だけ）
    watch_sorted: List[dict] = []
    for w in watch:
        watch_sorted.append(w)

    # watchを軽くソート（RR/day優先）
    watch_sorted.sort(key=lambda x: (float(x.get("r_per_day", 0.0)), float(x.get("rr", 0.0))), reverse=True)
    watch_sorted = watch_sorted[:MAX_WATCH]

    # 本命が trade_allowed False の場合でも “情報”として表示はする（新規見送り表示が入る）
    # ただし Action はそのまま出す（買うな、が明確）
    return {
        "trade_allowed": trade_allowed,
        "no_trade_reasons": no_trade_reasons,
        "picks": [
            {k: v for k, v in p.items() if k != "hist_close"} for p in picks
        ],
        "watch": watch_sorted,
        "sector_tops": sector_tops,
    }
