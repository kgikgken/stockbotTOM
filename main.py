from __future__ import annotations

import os
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score, calc_market_score_3d_delta
from utils.sector import top_sectors_5d, sector_rank_map
from utils.scoring import trend_gate, detect_setup_type, estimate_pwin
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

PRICE_MIN = 200
PRICE_MAX = 15000
ADV_MIN = 100_000_000
ATR_PCT_MIN = 0.015

EARNINGS_EXCLUDE_BDAYS = 3

MAX_OUTPUT = 5
MAX_CORE = 1      # 2→1
MAX_PER_SECTOR = 2
MAX_CORR = 0.75

MIN_R = 2.2
MIN_EV = 0.4
MIN_ADJ_EV_AVG = 0.6
MIN_R_PER_DAY = 0.5
MAX_EXPECTED_DAYS = 5


def fetch_history(ticker: str, period="260d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty and len(df) >= 120:
                return df
        except Exception:
            time.sleep(0.3)
    return None


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return float("nan")
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")


def _bday_distance(a, b) -> int:
    try:
        s = pd.Timestamp(min(a, b))
        e = pd.Timestamp(max(a, b))
        bd = pd.bdate_range(s, e)
        return int(len(bd) - 1)
    except Exception:
        return int(abs((a - b).days))


def filter_earnings(df: pd.DataFrame, today_date) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df
    d = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    keep = []
    for x in d:
        if pd.isna(x):
            keep.append(True)
        else:
            keep.append(_bday_distance(x, today_date) > EARNINGS_EXCLUDE_BDAYS)
    return df[keep]


def load_event_flags(today_date) -> Tuple[List[str], bool]:
    if not os.path.exists(EVENTS_PATH):
        return ["- 特になし"], False

    try:
        df = pd.read_csv(EVENTS_PATH)
    except Exception:
        return ["- 特になし"], False

    warns = []
    near = False
    for _, r in df.iterrows():
        dt = parse_event_datetime_jst(r.get("datetime"), r.get("date"), r.get("time"))
        if dt is None:
            continue
        delta = (dt.date() - today_date).days
        if delta in (0, 1):
            near = True
            warns.append(f"⚠ {str(r.get('label','')).strip()}（{dt.strftime('%m/%d %H:%M')}）")

    return (warns or ["- 特になし"]), near


def pass_universe_filters(hist: pd.DataFrame) -> bool:
    if hist is None or len(hist) < 120:
        return False

    c = float(hist["Close"].astype(float).iloc[-1])
    if not (np.isfinite(c) and PRICE_MIN <= c <= PRICE_MAX):
        return False

    if "Volume" not in hist.columns:
        return False
    vol = hist["Volume"].astype(float)
    adv20 = float((hist["Close"].astype(float) * vol).rolling(20).mean().iloc[-1])
    if not (np.isfinite(adv20) and adv20 >= ADV_MIN):
        return False

    atr = _atr(hist, 14)
    if not (np.isfinite(atr) and atr > 0):
        return False
    atr_pct = atr / c
    if atr_pct < ATR_PCT_MIN:
        return False

    return True


def corr_20d(hist_a: pd.DataFrame, hist_b: pd.DataFrame) -> float:
    try:
        ca = hist_a["Close"].astype(float).pct_change(fill_method=None).tail(20)
        cb = hist_b["Close"].astype(float).pct_change(fill_method=None).tail(20)
        df = pd.concat([ca, cb], axis=1).dropna()
        if len(df) < 10:
            return 0.0
        return float(df.corr().iloc[0, 1])
    except Exception:
        return 0.0


def run_swing(today_date, mkt: Dict, mkt_delta: int, event_near: bool) -> Tuple[List[Dict], List[Dict], str]:
    if mkt["score"] < 45:
        return [], [], "MarketScore<45"
    if mkt_delta <= -5 and mkt["score"] < 55:
        return [], [], "地合い悪化初動（Δ<=-5）"

    uni = pd.read_csv(UNIVERSE_PATH)
    t_col = "ticker" if "ticker" in uni.columns else "code"

    top_secs = [s for s, _ in top_sectors_5d(5)]
    sec_rank = sector_rank_map(top_secs)

    uni = filter_earnings(uni, today_date)

    candidates: List[Dict] = []
    watchlist: List[Dict] = []

    for _, r in uni.iterrows():
        ticker = str(r.get(t_col, "")).strip()
        if not ticker:
            continue

        sector = str(r.get("sector", r.get("industry_big", "不明")))
        exceptional = bool(r.get("ExceptionalFlag", False)) if "ExceptionalFlag" in uni.columns else False

        if sector not in sec_rank and not exceptional:
            continue

        hist = fetch_history(ticker)
        if hist is None:
            continue

        if not pass_universe_filters(hist):
            continue
        if not trend_gate(hist):
            continue

        setup = detect_setup_type(hist)
        if setup not in ("A", "B"):
            continue

        rr = compute_tp_sl_rr(hist, mkt_score=mkt["score"], setup=setup)
        R = float(rr.get("rr", 0.0))
        if R < MIN_R:
            continue

        atr = float(rr.get("atr", np.nan))
        entry = float(rr.get("entry", np.nan))
        stop = float(rr.get("stop", np.nan))
        tp1 = float(rr.get("tp1", np.nan))
        tp2 = float(rr.get("tp2", np.nan))
        if not (np.isfinite(entry) and np.isfinite(stop) and np.isfinite(tp2) and np.isfinite(atr)):
            continue

        expected_days = (tp2 - entry) / max(atr, 1.0)
        if expected_days > MAX_EXPECTED_DAYS:
            continue

        r_day = R / expected_days if expected_days > 0 else 0.0
        if r_day < MIN_R_PER_DAY:
            continue

        sr = sec_rank.get(sector, 999)
        pwin = estimate_pwin(hist, sector_rank=sr)

        ev = pwin * R - (1 - pwin)
        if ev < MIN_EV:
            continue

        adj = ev
        if mkt["score"] >= 60 and mkt_delta >= 0:
            adj *= 1.05
        if mkt_delta <= -5:
            adj *= 0.70
        if event_near:
            adj *= 0.75

        gu = bool(rr.get("gu_flag", False))
        in_low = float(rr.get("in_low", entry))
        in_high = float(rr.get("in_high", entry))
        dist_above = float(rr.get("dist_above_atr", rr.get("in_dist_atr", 0.0)))

        # 追いかけ禁止は「上方向」だけを判定。下方向は落ちるナイフ側なので原則監視。
        action = "即IN可"
        if gu:
            action = "監視のみ"
        elif price_now < in_low:
            action = "監視のみ"
        elif price_now > in_high or dist_above > 0.8:
            action = "指値待ち"

        item = dict(
            ticker=ticker,
            name=str(r.get("name", ticker)),
            sector=sector,
            setup=setup,
            entry=entry,
            stop=stop,
            tp1=tp1,
            tp2=tp2,
            atr=atr,
            R=R,
            ev=ev,
            adj_ev=adj,
            r_day=float(r_day),
            expected_days=float(expected_days),
            action=action,
            hist=hist,
        )

        if gu or in_dist_above > 1.2 or price_now < in_low:
            watchlist.append(item)
        else:
            candidates.append(item)

    if not candidates:
        return [], watchlist[:10], "候補なし"

    avg_adj = float(np.mean([c["adj_ev"] for c in candidates])) if candidates else 0.0
    if avg_adj < MIN_ADJ_EV_AVG:
        return [], watchlist[:10], f"平均AdjEV<{MIN_ADJ_EV_AVG:.2f}"

    for c in candidates:
        c["rank_score"] = float(c["adj_ev"] * c["r_day"])

    candidates.sort(key=lambda x: x["rank_score"], reverse=True)

    picked: List[Dict] = []
    per_sector = {}

    for c in candidates:
        if len(picked) >= MAX_OUTPUT:
            break

        sec = c["sector"]
        per_sector.setdefault(sec, 0)
        if per_sector[sec] >= MAX_PER_SECTOR:
            continue

        ok = True
        for p in picked:
            if corr_20d(c["hist"], p["hist"]) > MAX_CORR:
                ok = False
                break
        if not ok:
            continue

        picked.append(c)
        per_sector[sec] += 1

    core = picked[:MAX_CORE]
    rest = picked[MAX_CORE:]
    return core + rest, watchlist[:10], "OK"


def _setup_jp(s: str) -> str:
    return "押し目" if s == "A" else ("ブレイク" if s == "B" else "不明")


def build_report(today: str, mkt: Dict, mkt_delta: int, events: List[str],
                 picks: List[Dict], watchlist: List[Dict], reason: str,
                 pos_text: str, total_asset: float) -> str:
    trade_ok = (reason == "OK" and len(picks) > 0)

    lev = 2.0 if mkt["score"] >= 60 else (1.7 if mkt["score"] >= 45 else 1.0)
    max_pos = int(round(total_asset * lev))

    top_secs = top_sectors_5d(5)

    lines: List[str] = []
    lines.append(f"📅 {today} stockbotTOM 日報\n")

    lines.append("◆ 今日の結論（Swing専用）")
    lines.append("✅ 新規可（条件クリア）" if trade_ok else "🚫 新規見送り")
    lines.append(f"- 地合い: {mkt['score']}点 ({mkt.get('comment','')})")
    lines.append(f"- ΔMarketScore_3d: {mkt_delta:+d}")
    lines.append(f"- レバ: {lev:.1f}倍")
    lines.append(f"- MAX建玉: 約{max_pos:,}円\n")

    lines.append("📈 セクター（5日）")
    if top_secs:
        for i, (s, p) in enumerate(top_secs, 1):
            lines.append(f"{i}. {s} ({p:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    if events:
        lines.append("⚠ イベント")
        lines.extend(events)
        lines.append("")

    if not trade_ok:
        lines.append("🚫 本日は新規エントリー禁止")
        lines.append(f"理由: {reason}")
        lines.append("")
        lines.append("📊 ポジション")
        lines.append(pos_text if pos_text else "ノーポジション")
        return "\n".join(lines)

    core = picks[0]
    lines.append("🎯 本命（この1銘柄のみ）")
    lines.append(f"- {core['ticker']} {core['name']} [{core['sector']}] ⭐")
    lines.append(f"  セットアップ:{_setup_jp(core['setup'])}")
    lines.append(f"  IN:{core['entry']:.1f}  STOP:{core['stop']:.1f}")
    lines.append(f"  TP1:{core['tp1']:.1f}  TP2:{core['tp2']:.1f}")
    lines.append(f"  RR:{core['R']:.2f}  AdjEV:{core['adj_ev']:.2f}  R/day:{core['r_day']:.2f}  目安:{core['expected_days']:.1f}日")
    lines.append(f"  行動:{core['action']}\n")

    if len(picks) > 1:
        lines.append("👀 監視・指値（今日は本命以外を触らない）")
        for c in picks[1:]:
            lines.append(f"- {c['ticker']} [{c['sector']}]  {_setup_jp(c['setup'])}  RR:{c['R']:.2f} AdjEV:{c['adj_ev']:.2f} R/day:{c['r_day']:.2f}  行動:{c['action']}")
        lines.append("")

    if watchlist:
        lines.append("🧊 監視のみ（追いかけ/ギャップ/乖離）")
        for w in watchlist[:10]:
            lines.append(f"- {w['ticker']}  {_setup_jp(w['setup'])}  行動:{w['action']}")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text if pos_text else "ノーポジション")

    return "\n".join(lines)


def send_line(text: str) -> None:
    if not WORKER_URL:
        print(text)
        return
    for ch in [text[i:i + 3800] for i in range(0, len(text), 3800)]:
        r = requests.post(WORKER_URL, json={"text": ch}, timeout=20)
        print("[LINE]", r.status_code, str(r.text)[:100])


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    mkt_delta = calc_market_score_3d_delta()

    events, event_near = load_event_flags(today_date)

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=int(mkt["score"]))
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    picks, watchlist, reason = run_swing(today_date, mkt, mkt_delta, event_near)

    report = build_report(
        today=today_str,
        mkt=mkt,
        mkt_delta=mkt_delta,
        events=events,
        picks=picks,
        watchlist=watchlist,
        reason=reason,
        pos_text=pos_text,
        total_asset=total_asset,
    )

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()
