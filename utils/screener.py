from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from utils.util import (
    read_csv_safely,
    pick_ticker_column,
    pick_sector_column,
    yf_history,
    safe_float,
    jst_today_date,
)
from utils.events import build_event_warnings
from utils.sector import top_sectors_5d, sector_rank_map
from utils.features import add_indicators, atr, rel_strength_20d, is_too_perfect
from utils.setup import detect_setup
from utils.entry import calc_entry_zone
from utils.rr_ev import compute_rr_targets, pwin_proxy, ev_and_speed
from utils.diversify import apply_diversify


UNIVERSE_PATH = "universe_jpx.csv"

# 決算±N
EARNINGS_EXCLUDE_DAYS = 3

# Universe（母集団は全銘柄。セクターは理由にしない）
PRICE_MIN = 200
PRICE_MAX = 15000
ADV20_MIN = 200_000_000  # 200M JPY
ATR_PCT_MIN = 0.015
ATR_PCT_MAX = 0.060

# Swing要件
RR_MIN = 1.8
EV_MIN = 0.30
EXP_DAYS_MAX = 5.0
RDAY_MIN = 0.50

# 出力上限
MAX_FINAL = 5
MAX_WATCH = 10


def _earnings_blocked(earnings_date: str, today_date) -> bool:
    if not earnings_date or str(earnings_date).strip() == "":
        return False
    try:
        d = pd.to_datetime(earnings_date, errors="coerce")
        if pd.isna(d):
            return False
        ed = d.date()
        delta = abs((ed - today_date).days)
        return delta <= EARNINGS_EXCLUDE_DAYS
    except Exception:
        return False


def _universe_rows() -> Tuple[pd.DataFrame, str, str]:
    df = read_csv_safely(UNIVERSE_PATH)
    t_col = pick_ticker_column(df) or ""
    s_col = pick_sector_column(df) or ""
    return df, t_col, s_col


def _market_no_trade(market: Dict, avg_adj_ev: float, gu_ratio: float) -> Tuple[bool, List[str]]:
    reasons = list(market.get("no_trade_reasons", [])) if isinstance(market.get("no_trade_reasons"), list) else []
    nt = bool(market.get("no_trade", False))

    # 仕様：上位候補の平均AdjustedEV < 0.3R
    if np.isfinite(avg_adj_ev) and avg_adj_ev < 0.30:
        nt = True
        reasons.append("平均AdjEV<0.3R")

    # 仕様：GU比率 >= 60%
    if np.isfinite(gu_ratio) and gu_ratio >= 0.60:
        nt = True
        reasons.append("GU比率>=60%")

    return nt, reasons


def run_screener(market: Dict) -> Dict:
    today = jst_today_date()
    mkt_score = int(market.get("score", 50))
    delta3d = int(market.get("delta3d", 0))
    regime_mult = float(market.get("regime_mult", 1.0))

    # イベント表示＆リスクday（multは market 側で織り込み済）
    event_lines, _ = build_event_warnings(today)

    # セクター（補助情報として表示）
    sector_top5 = top_sectors_5d(5)
    s_rank = sector_rank_map()

    uni, t_col, s_col = _universe_rows()
    if uni.empty or not t_col:
        return {
            "final": [],
            "watch": [],
            "drops": [],
            "events": event_lines,
            "sector_top5": sector_top5,
            "no_trade": True,
            "no_trade_reasons": ["universe読み込み失敗"],
        }

    cands: List[Dict] = []
    drops: List[Dict] = []
    watch: List[Dict] = []

    # 指数close（RS用：TOPIX優先）
    index_hist = yf_history("^TOPX", period="260d")
    if index_hist is None or index_hist.empty:
        index_hist = yf_history("^N225", period="260d")
    index_close = index_hist["Close"].astype(float) if index_hist is not None and not index_hist.empty else None

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get(s_col, "不明")) if s_col else str(row.get("sector", "不明"))
        earnings = str(row.get("earnings_date", "")).strip()

        # 決算±3日は新規完全回避（監視は可）
        if _earnings_blocked(earnings, today):
            drops.append({"ticker": ticker, "name": name, "sector": sector, "reason": "決算±3日"})
            continue

        hist = yf_history(ticker, period="260d")
        if hist is None or len(hist) < 120:
            drops.append({"ticker": ticker, "name": name, "sector": sector, "reason": "データ不足"})
            continue

        df = add_indicators(hist)
        close = df["Close"].astype(float)
        c = safe_float(close.iloc[-1])

        # Universe 필터：価格
        if not (np.isfinite(c) and PRICE_MIN <= c <= PRICE_MAX):
            drops.append({"ticker": ticker, "name": name, "sector": sector, "reason": "価格帯外"})
            continue

        # ADV20（売買代金）
        adv20 = safe_float(df["adv20"].iloc[-1])
        if not (np.isfinite(adv20) and adv20 >= ADV20_MIN):
            drops.append({"ticker": ticker, "name": name, "sector": sector, "reason": f"流動性弱(ADV20<{ADV20_MIN/1e6:.0f}M)"})
            continue

        # ATR%
        a = atr(hist, 14)
        atrp = float(a / c) if np.isfinite(a) and c > 0 else np.nan
        if not (np.isfinite(atrp) and ATR_PCT_MIN <= atrp <= ATR_PCT_MAX):
            drops.append({"ticker": ticker, "name": name, "sector": sector, "reason": "ボラ不適(ATR%)"})
            continue

        # “完璧すぎない”（急伸は追わない）
        if is_too_perfect(hist):
            drops.append({"ticker": ticker, "name": name, "sector": sector, "reason": "完璧すぎ(急伸)"})
            continue

        # Setup
        setup_info = detect_setup(hist)
        setup = str(setup_info.get("setup", "-"))
        if setup not in ("A", "B"):
            drops.append({"ticker": ticker, "name": name, "sector": sector, "reason": "形不一致"})
            continue

        # Entry + Action（追いかけ禁止はここで完了）
        entry = calc_entry_zone(hist, setup)
        gu = bool(entry.get("gu", False))

        # RR/Targets（固定RR禁止、構造から出る）
        rr_info = compute_rr_targets(hist, setup, entry)
        rr = float(rr_info.get("rr", 0.0))
        if rr < RR_MIN:
            drops.append({"ticker": ticker, "name": name, "sector": sector, "reason": f"RR<{RR_MIN:.1f}"})
            continue

        # RS20
        rs20 = float("nan")
        if index_close is not None:
            rs20 = rel_strength_20d(close, index_close)

        # Sector rank（補助）
        sr = s_rank.get(sector, None)

        # Pwin proxy
        pwin = pwin_proxy(hist, setup, rs20=rs20, sector_rank=sr, adv20=adv20, gu=gu)

        # EV + speed
        sp = ev_and_speed(rr=rr, pwin=pwin, atr_yen=float(entry.get("atr", a)), in_price=float(entry.get("in_center")), tp2=float(rr_info.get("tp2")))
        ev = float(sp["ev"])
        exp_days = float(sp["exp_days"])
        r_day = float(sp["r_day"])

        if ev < EV_MIN:
            drops.append({"ticker": ticker, "name": name, "sector": sector, "reason": f"EV<{EV_MIN:.2f}R"})
            continue
        if exp_days > EXP_DAYS_MAX or r_day < RDAY_MIN:
            drops.append({"ticker": ticker, "name": name, "sector": sector, "reason": "速度不足(R/day)"})
            continue

        adj_ev = ev * regime_mult

        # 監視送り条件（GUなど）
        action = str(entry.get("action", "WATCH_ONLY"))
        if gu:
            action = "WATCH_ONLY"

        cand = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "setup": setup,
            "price_now": float(entry.get("price_now")) if entry.get("price_now") is not None else None,
            "atr": float(entry.get("atr")),
            "gu": "Y" if gu else "N",
            "in_center": float(entry.get("in_center")),
            "in_low": float(entry.get("in_low")),
            "in_high": float(entry.get("in_high")),
            "stop": float(rr_info.get("stop")),
            "tp1": float(rr_info.get("tp1")),
            "tp2": float(rr_info.get("tp2")),
            "rr": float(rr),
            "pwin": float(pwin),
            "ev": float(ev),
            "adj_ev": float(adj_ev),
            "exp_days": float(exp_days),
            "r_day": float(r_day),
            "action": action,
            "sector_rank": sr,
            "adv20": float(adv20),
            "atr_pct": float(atrp * 100.0),
            "_close_series": close,  # diversify用（内部）
        }
        cands.append(cand)

    # ソート：優先順位＝速度(R/day)主導 → AdjEV → RR
    cands.sort(key=lambda x: (x["r_day"], x["adj_ev"], x["rr"]), reverse=True)

    # 多様化フィルタ（sector cap / corr cap）
    selected, dropped2 = apply_diversify(cands, sector_cap=2, corr_cap=0.75)
    # drop理由をwatchに回す（落選の説明）
    for d in dropped2:
        reason = str(d.get("drop_reason", "制約落ち"))
        watch.append({
            "ticker": d["ticker"],
            "name": d["name"],
            "sector": d["sector"],
            "setup": d.get("setup", "-"),
            "rr": float(d.get("rr", 0.0)),
            "r_day": float(d.get("r_day", 0.0)),
            "reason": reason,
            "action": str(d.get("action", "WATCH_ONLY")),
            "gu": str(d.get("gu", "N")),
        })

    final = selected[:MAX_FINAL]

    # GU比率、平均AdjEVで NO-TRADE 最終確定
    gu_ratio = 0.0
    if final:
        gu_ratio = float(np.mean([1.0 if c["gu"] == "Y" else 0.0 for c in final]))
        avg_adj_ev = float(np.mean([c["adj_ev"] for c in final]))
    else:
        avg_adj_ev = 0.0

    no_trade, nt_reasons = _market_no_trade(market, avg_adj_ev=avg_adj_ev, gu_ratio=gu_ratio)

    # NO-TRADEなら action を強制 WATCH_ONLY（裁量ゼロ）
    if no_trade:
        for c in final:
            c["action"] = "WATCH_ONLY"
        # watch側も統一
        for w in watch:
            w["action"] = "WATCH_ONLY"

    # watchは「落とした理由」中心に最大10
    watch.sort(key=lambda x: (x.get("r_day", 0.0), x.get("rr", 0.0)), reverse=True)
    watch = watch[:MAX_WATCH]

    # internal series remove（表示・送信に不要）
    for c in final:
        c.pop("_close_series", None)

    return {
        "final": final,
        "watch": watch,
        "drops": drops[-MAX_WATCH:],  # 全部は多すぎるので末尾のみ
        "events": event_lines,
        "sector_top5": sector_top5,
        "no_trade": bool(no_trade),
        "no_trade_reasons": nt_reasons,
        "avg_rr": float(np.mean([c["rr"] for c in final])) if final else 0.0,
        "avg_ev": float(np.mean([c["ev"] for c in final])) if final else 0.0,
        "avg_adj_ev": float(np.mean([c["adj_ev"] for c in final])) if final else 0.0,
        "avg_r_day": float(np.mean([c["r_day"] for c in final])) if final else 0.0,
    }