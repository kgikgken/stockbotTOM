from __future__ import annotations

import os
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.scoring import universe_ok, setup_type, in_zone, action_label, pwin_proxy, regime_multiplier, ev_from
from utils.rr import build_trade_plan
from utils.position import load_positions, analyze_positions

UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

EARNINGS_EXCLUDE_DAYS = 3
SECTOR_TOP_N = 5

CORE_MAX = 5
WATCH_MAX = 10

RR_MIN = 2.2
EV_MIN = 0.40
EXP_DAYS_MAX = 5.0
R_PER_DAY_MIN = 0.50

MAX_PER_SECTOR = 2
MAX_CORR = 0.75

def fetch_history(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.4)
    return None

def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    events: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        label = str(row.get("label", "")).strip()
        kind = str(row.get("kind", "")).strip()
        date_str = str(row.get("date", "")).strip()
        time_str = str(row.get("time", "")).strip()
        dt_str = str(row.get("datetime", "")).strip()
        if not label:
            continue
        events.append({"label": label, "kind": kind, "date": date_str, "time": time_str, "datetime": dt_str})
    return events

def build_event_warnings(today_date) -> Tuple[List[str], bool]:
    events = load_events()
    warns: List[str] = []
    has_near = False
    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            has_near = True
            when = "本日" if delta == 0 else ("直近" if delta < 0 else f"{delta}日後")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")
    if not warns:
        warns.append("- 特になし")
    return warns, has_near

def filter_earnings(df: pd.DataFrame, today_date) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df
    try:
        parsed = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    except Exception:
        return df
    keep = []
    for d in parsed:
        if d is None or pd.isna(d):
            keep.append(True)
            continue
        try:
            keep.append(abs((d - today_date).days) > EARNINGS_EXCLUDE_DAYS)
        except Exception:
            keep.append(True)
    return df[keep]

def market_no_trade(mkt_score: int, delta3d: int) -> Tuple[bool, List[str]]:
    reasons = []
    if mkt_score < 45:
        reasons.append("MarketScore<45")
    if delta3d <= -5 and mkt_score < 55:
        reasons.append("Δ3d<=-5 かつ MarketScore<55")
    return (len(reasons) > 0), reasons

def _corr_from_hist(a: pd.DataFrame, b: pd.DataFrame) -> float:
    try:
        ca = a["Close"].astype(float).pct_change(fill_method=None).dropna().tail(20)
        cb = b["Close"].astype(float).pct_change(fill_method=None).dropna().tail(20)
        n = min(len(ca), len(cb))
        if n < 15:
            return 0.0
        ca = ca.iloc[-n:]
        cb = cb.iloc[-n:]
        return float(np.corrcoef(ca.values, cb.values)[0, 1])
    except Exception:
        return 0.0

def run_swing(today_date, mkt: dict) -> Tuple[List[Dict], List[Dict], bool, List[str]]:
    mkt_score = int(mkt.get("score", 50))
    delta3d = int(mkt.get("delta3d", 0))
    no_trade, no_trade_reasons = market_no_trade(mkt_score, delta3d)

    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], [], True, ["universe読めず"]

    if "ticker" in uni.columns:
        t_col = "ticker"
    elif "code" in uni.columns:
        t_col = "code"
    else:
        return [], [], True, ["universeにticker/code列なし"]

    uni_new = filter_earnings(uni, today_date)

    sector_ranking = top_sectors_5d(top_n=SECTOR_TOP_N)
    sector_rank_map = {s: i + 1 for i, (s, _) in enumerate(sector_ranking)}
    top_sectors = set(sector_rank_map.keys())

    _, has_event_near = build_event_warnings(today_date)

    raw: List[Dict] = []
    hist_cache: Dict[str, pd.DataFrame] = {}

    for _, row in uni_new.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        if sector not in top_sectors:
            continue

        hist = fetch_history(ticker, period="260d")
        if hist is None or len(hist) < 120:
            continue

        ok, _ = universe_ok(hist, adv_min=200_000_000)
        if not ok:
            continue

        stype, stinfo = setup_type(hist)
        if stype not in ("A", "B"):
            continue

        cen, low, high = in_zone(hist, stype)
        plan = build_trade_plan(hist, stype, cen, low, high, mkt_score=mkt_score)

        rr = float(plan["rr"])
        if rr < RR_MIN:
            continue

        srank = sector_rank_map.get(sector)
        pwin = pwin_proxy(hist, stype, srank, mkt_score=mkt_score)
        ev = ev_from(pwin, rr)

        mult = regime_multiplier(mkt_score, delta3d, has_event_near=has_event_near)
        adj_ev = ev * mult

        exp_days = float(plan["expected_days"])
        rpd = float(plan["r_per_day"])
        if exp_days > EXP_DAYS_MAX:
            continue
        if rpd < R_PER_DAY_MIN:
            continue
        if adj_ev < EV_MIN:
            continue

        price_now = float(plan["price_now"])
        act, dev = action_label(price_now, cen, float(plan["atr"]), bool(stinfo.get("gu", False)))

        hist_cache[ticker] = hist

        raw.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                sector_rank=srank,
                setup=stype,
                rr=rr,
                ev=ev,
                adj_ev=adj_ev,
                r_per_day=rpd,
                expected_days=exp_days,
                action=act,
                dev=dev,
                in_center=cen,
                in_low=low,
                in_high=high,
                atr=float(plan["atr"]),
                gu="Y" if bool(stinfo.get("gu", False)) else "N",
                stop=float(plan["stop"]),
                tp1=float(plan["tp1"]),
                tp2=float(plan["tp2"]),
                price_now=price_now,
            )
        )

    raw.sort(key=lambda x: (x["adj_ev"], x["r_per_day"], x["rr"]), reverse=True)

    if no_trade:
        return [], raw[:WATCH_MAX], True, no_trade_reasons

    core: List[Dict] = []
    watch: List[Dict] = []
    sector_cnt: Dict[str, int] = {}

    for c in raw:
        if len(core) >= CORE_MAX:
            watch.append(c)
            continue

        sec = c["sector"]
        if sector_cnt.get(sec, 0) >= MAX_PER_SECTOR:
            c2 = c.copy()
            c2["reason"] = "セクター上限"
            watch.append(c2)
            continue

        too_corr = False
        for chosen in core:
            ca = hist_cache.get(c["ticker"])
            cb = hist_cache.get(chosen["ticker"])
            if ca is None or cb is None:
                continue
            corr = _corr_from_hist(ca, cb)
            if corr > MAX_CORR:
                too_corr = True
                c2 = c.copy()
                c2["reason"] = f"相関>{MAX_CORR:.2f}"
                watch.append(c2)
                break
        if too_corr:
            continue

        core.append(c)
        sector_cnt[sec] = sector_cnt.get(sec, 0) + 1

    if core:
        avg_adj = float(np.mean([x["adj_ev"] for x in core]))
        gu_ratio = float(np.mean([1.0 if x["gu"] == "Y" else 0.0 for x in core]))
        if avg_adj < 0.30:
            return [], (core + watch)[:WATCH_MAX], True, ["上位候補の平均AdjEV<0.3R"]
        if gu_ratio >= 0.60:
            return [], (core + watch)[:WATCH_MAX], True, ["GU比率>=60%"]

    return core, watch[:WATCH_MAX], False, []

def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 2.0, "強気（押し目＋一部ブレイク）"
    if mkt_score >= 60:
        return 1.7, "やや強気（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "中立（厳選・押し目中心）"
    if mkt_score >= 45:
        return 1.1, "守り（厳選）"
    return 0.0, "新規禁止"

def _action_jp(a: str) -> str:
    return {"EXEC_NOW": "即IN可", "LIMIT_WAIT": "指値待ち", "WATCH_ONLY": "監視のみ"}.get(a, a)

def build_report(today_str: str, today_date, mkt: Dict, pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))
    delta3d = int(mkt.get("delta3d", 0))

    lev, lev_comment = recommend_leverage(mkt_score)
    max_pos = int(round(total_asset * lev)) if (lev > 0 and np.isfinite(total_asset) and total_asset > 0) else 0

    sectors = top_sectors_5d(top_n=SECTOR_TOP_N)
    events, _ = build_event_warnings(today_date)

    core, watch, is_no_trade, no_trade_reasons = run_swing(today_date, mkt)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")
    if is_no_trade:
        reason = " / ".join(no_trade_reasons) if no_trade_reasons else "条件該当"
        lines.append(f"🚫 新規見送り（{reason}）")
    else:
        lines.append("✅ 新規可（条件クリア）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- ΔMarketScore_3d: {delta3d:+d}")
    if lev > 0:
        lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
        lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    lines.append("📈 セクター（5日）")
    if sectors:
        for i, (s_name, pct) in enumerate(sectors, 1):
            lines.append(f"{i}. {s_name} ({pct:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ イベント")
    lines.extend(events)
    lines.append("")

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    if core:
        rr_avg = float(np.mean([c["rr"] for c in core]))
        ev_avg = float(np.mean([c["ev"] for c in core]))
        adj_avg = float(np.mean([c["adj_ev"] for c in core]))
        rpd_avg = float(np.mean([c["r_per_day"] for c in core]))
        lines.append(f"  候補数:{len(core)}銘柄 / 平均RR:{rr_avg:.2f} / 平均EV:{ev_avg:.2f} / 平均AdjEV:{adj_avg:.2f} / 平均R/day:{rpd_avg:.2f}")
        lines.append("")
        for c in core:
            star = " ⭐" if c["action"] == "EXEC_NOW" else ""
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]{star}")
            lines.append(f"  形:{c['setup']}  RR:{c['rr']:.2f}  AdjEV:{c['adj_ev']:.2f}  R/day:{c['r_per_day']:.2f}")
            lines.append(f"  IN:{c['in_center']:.1f}（帯:{c['in_low']:.1f}〜{c['in_high']:.1f}） 現在:{c['price_now']:.1f}  ATR:{c['atr']:.1f}  GU:{c['gu']}")
            lines.append(f"  STOP:{c['stop']:.1f}  TP1:{c['tp1']:.1f}  TP2:{c['tp2']:.1f}  ExpectedDays:{c['expected_days']:.1f}  行動:{_action_jp(c['action'])}")
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    if watch:
        lines.append("🧠 監視リスト（今日は入らない）")
        for c in watch[:WATCH_MAX]:
            reason = c.get("reason")
            if not reason:
                reason = "追いかけ禁止/乖離" if c.get("action") == "WATCH_ONLY" else "制約落ち"
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}] 形:{c['setup']} RR:{c['rr']:.2f} R/day:{c['r_per_day']:.2f} 理由:{reason} 行動:{_action_jp(c['action'])} GU:{c['gu']}")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")
    return "\n".join(lines)

def send_line(text: str) -> None:
    if not WORKER_URL:
        print(text)
        return
    chunk_size = 3800
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]
    for ch in chunks:
        requests.post(WORKER_URL, json={"text": ch}, timeout=20)

def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    mkt_score = int(mkt.get("score", 50))

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=mkt_score)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    report = build_report(today_str, today_date, mkt, pos_text, total_asset)
    print(report)
    send_line(report)

if __name__ == "__main__":
    main()
