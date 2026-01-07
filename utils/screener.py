# ============================================
# utils/screener.py
# 全体スクリーニング（NO-TRADE機械化、週次制限、イベント時候補数制限 等）
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import csv

import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import (
    jst_now,
    jst_today_str,
    leverage_by_market,
    rr_min_by_market,
)
from utils.events import load_events_csv, nearest_event, macro_risk_on
from utils.market import build_market_context, MarketContext
from utils.sector import top_sectors_5d
from utils.features import compute_features, Features
from utils.setup import detect_setup
from utils.entry import calc_entry_plan
from utils.rr_ev import build_rr_ev
from utils.diversify import pick_with_constraints
from utils.position import load_positions, analyze_positions


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"

# Universeフィルタ
PRICE_MIN = 200
PRICE_MAX = 15000
ADV_MIN_JPY = 200_000_000  # 200M
ATR_PCT_MIN = 1.5
ATR_PCT_MAX = 6.0

# 決算フィルタ
EARNINGS_EXCLUDE_DAYS = 3

# スクリーニング上限
MAX_FINAL_STOCKS = 5
MAX_WATCH = 10

# NO-TRADE閾値
NO_TRADE_MKT_SCORE = 45.0
NO_TRADE_DELTA3D = -5.0
NO_TRADE_DELTA3D_MKT_CAP = 55.0

# 週次新規制限
WEEKLY_NEW_MAX = 3
WEEKLY_STATE_PATH = "weekly_state.csv"

# イベント接近時（マクロ警戒ON）候補数上限
MAX_CANDIDATES_ON_EVENT = 2

# AdjEVの下限
ADJEV_MIN = 0.5


def _read_universe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 必須: ticker
    if "ticker" not in df.columns:
        return pd.DataFrame()
    if "sector" not in df.columns:
        df["sector"] = "不明"
    if "earnings_date" not in df.columns:
        df["earnings_date"] = ""
    return df


def _fetch_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        return df
    return pd.DataFrame()


def _adv20_jpy(df: pd.DataFrame) -> float:
    try:
        if df is None or df.empty:
            return 0.0
        x = df.tail(20)
        turn = (x["Close"] * x["Volume"]).mean()
        return float(turn)
    except Exception:
        return 0.0


def _earnings_blocked(earnings_date_str: str, today_str: str) -> bool:
    """
    earnings_date_str: "YYYY-MM-DD" 前提（空ならFalse）
    """
    s = (earnings_date_str or "").strip()
    if not s:
        return False
    try:
        ed = pd.to_datetime(s).date()
        td = pd.to_datetime(today_str).date()
        d = abs((ed - td).days)
        return d <= EARNINGS_EXCLUDE_DAYS
    except Exception:
        return False


def _load_weekly_state(path: str = WEEKLY_STATE_PATH) -> int:
    """
    今週の新規回数（0-3）を返す
    """
    if not os.path.exists(path):
        return 0
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.empty:
            return 0
        # columns: week_key, count
        week_key = pd.Timestamp(jst_now().date()).strftime("%G-W%V")
        row = df[df["week_key"] == week_key]
        if row.empty:
            return 0
        return int(row.iloc[-1]["count"])
    except Exception:
        return 0


def _save_weekly_state(count: int, path: str = WEEKLY_STATE_PATH) -> None:
    week_key = pd.Timestamp(jst_now().date()).strftime("%G-W%V")
    rows = [{"week_key": week_key, "count": int(count)}]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _no_trade_by_market(ctx: MarketContext) -> Optional[str]:
    """
    NO-TRADE条件
    """
    if ctx.market_score < NO_TRADE_MKT_SCORE:
        return "MarketScore<45"
    if (ctx.delta_3d <= NO_TRADE_DELTA3D) and (ctx.market_score < NO_TRADE_DELTA3D_MKT_CAP):
        return "Δ3d<=-5 & MarketScore<55"
    return None


@dataclass
class ScreeningResult:
    today: str
    market_score: float
    delta_3d: float
    regime: str
    macro_risk: bool
    weekly_new_count: int

    leverage: float
    max_exposure_jpy: float

    sectors_top5: List[Tuple[str, float]]
    nearest_event_text: str

    no_trade: bool
    no_trade_reason: str

    picks: List[dict]
    watch: List[dict]

    positions_text: str
    lot_risk_text: str


def run_screening() -> ScreeningResult:
    now = jst_now()
    today = jst_today_str()

    events = load_events_csv(EVENTS_PATH)
    near = nearest_event(events, now)
    macro_risk = macro_risk_on(events, now, days_ahead=2)

    ctx = build_market_context(macro_risk=macro_risk)

    weekly_new_count = _load_weekly_state(WEEKLY_STATE_PATH)

    no_trade_reason = ""
    no_trade = False

    # NO-TRADE判定（市場）
    x = _no_trade_by_market(ctx)
    if x:
        no_trade = True
        no_trade_reason = x

    # Universe
    uni = _read_universe(UNIVERSE_PATH)

    # positions
    positions = load_positions(POSITIONS_PATH)
    pos_summary, lot_risk_text = analyze_positions(positions, ctx.market_score, ctx.macro_risk)

    # レバ/最大建玉
    lev = leverage_by_market(ctx.market_score, macro_risk=ctx.macro_risk)
    # 資産はpositions側に依存（なければ2,000,000を仮定）
    base_capital = pos_summary.get("capital_jpy", 2_000_000.0)
    max_exposure = float(base_capital) * lev

    # イベント表示
    nearest_event_text = "特になし"
    if near:
        ev, days = near
        nearest_event_text = f"⚠ {ev.name}（{ev.dt_jst.strftime('%Y-%m-%d %H:%M')} JST / {days}日後）"

    # まず候補を作る（全銘柄を母集団にできる想定だが、データ取得はここではuniverseベース）
    candidates: List[dict] = []
    returns_5d: Dict[str, float] = {}

    for _, row in uni.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        sector = str(row.get("sector") or "不明")
        earnings_date = str(row.get("earnings_date") or "")

        df = _fetch_ohlcv(ticker, period="1y")
        if df.empty:
            continue

        f = compute_features(df)
        if f is None:
            continue

        # 5d return for sector ranking
        returns_5d[ticker] = float(f.ret5d)

        # Universeフィルタ
        if not (PRICE_MIN <= f.close <= PRICE_MAX):
            continue
        adv = _adv20_jpy(df)
        if adv < ADV_MIN_JPY:
            continue
        if f.atrp14 < ATR_PCT_MIN:
            continue
        if f.atrp14 > ATR_PCT_MAX:
            # 除外ではなく、弱めにするならここでcontinue
            continue

        # 決算回避（新規は完全回避）
        if _earnings_blocked(earnings_date, today):
            continue

        # Setup
        setup = detect_setup(f)
        if setup.setup_type == "-":
            continue

        # Entry
        ep = calc_entry_plan(setup.setup_type, f)

        # RR/EV
        rrev = build_rr_ev(f, setup.setup_type, ctx.market_score, ctx.macro_risk)
        if rrev is None:
            continue

        # RR下限（地合い連動）
        rr_min = rr_min_by_market(ctx.market_score)
        if rrev.rr < rr_min:
            continue

        # AdjEV 下限
        if rrev.adj_ev < ADJEV_MIN:
            continue

        # GUは原則「監視のみ」
        if f.gu_flag:
            ep.action = "監視のみ"

        candidates.append(
            {
                "ticker": ticker,
                "sector": sector,
                "setup_type": setup.setup_type,
                "setup_reason": setup.reason,
                "close": f.close,
                "atr": f.atr14,
                "gu": f.gu_flag,
                "entry": ep.entry,
                "band_low": ep.band_low,
                "band_high": ep.band_high,
                "action": ep.action,
                "stop": rrev.stop,
                "tp1": rrev.tp1,
                "tp2": rrev.tp2,
                "rr": rrev.rr,
                "ev": rrev.ev,
                "adj_ev": rrev.adj_ev,
                "days": rrev.expected_days,
                "r_day": rrev.r_per_day,
                "close_series": df["Close"].copy(),
            }
        )

    # セクターTop5（補助）
    sectors = top_sectors_5d(uni, returns_5d, top_n=5)

    # ソート（優先順位：速度(R/day)主導 → AdjEV → RR）
    candidates.sort(key=lambda x: (x["r_day"], x["adj_ev"], x["rr"]), reverse=True)

    # マクロ警戒ON（イベント接近）なら候補数を最大2に制限
    max_final = MAX_FINAL_STOCKS
    if ctx.macro_risk:
        max_final = min(max_final, MAX_CANDIDATES_ON_EVENT)

    # 分散制約（セクター上限・相関）
    selected, watch1 = pick_with_constraints(
        candidates=candidates,
        max_final=max_final,
        max_per_sector=2,
        corr_threshold=0.75,
    )

    # watch整形
    watch = []
    for w in watch1:
        if len(watch) >= MAX_WATCH:
            break
        watch.append(w)

    # 最終NO-TRADE（イベント接近は「候補2」+ 原則新規見送りにできる）
    if ctx.macro_risk:
        no_trade = True
        no_trade_reason = "イベント接近"

    # 週次新規制限
    if weekly_new_count >= WEEKLY_NEW_MAX:
        no_trade = True
        no_trade_reason = "週次新規上限"

    # 候補平均AdjEVが弱い場合（裁量ゼロ）
    if selected:
        avg_adj = float(np.mean([x["adj_ev"] for x in selected]))
        if avg_adj < 0.6:
            no_trade = True
            no_trade_reason = "平均AdjEV不足"
    else:
        # 候補なし → NO-TRADE
        no_trade = True
        if not no_trade_reason:
            no_trade_reason = "候補なし"

    return ScreeningResult(
        today=today,
        market_score=ctx.market_score,
        delta_3d=ctx.delta_3d,
        regime=ctx.regime,
        macro_risk=ctx.macro_risk,
        weekly_new_count=weekly_new_count,
        leverage=lev if not no_trade else 1.0,
        max_exposure_jpy=max_exposure if not no_trade else float(base_capital) * 1.0,
        sectors_top5=sectors,
        nearest_event_text=nearest_event_text,
        no_trade=no_trade,
        no_trade_reason=no_trade_reason,
        picks=selected,
        watch=watch,
        positions_text=pos_summary.get("text", "n/a"),
        lot_risk_text=lot_risk_text,
    )