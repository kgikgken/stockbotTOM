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
from utils.scoring import score_stock, calc_inout_for_stock
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions

# Swing-only add-on
from utils.qualify import evaluate_runner, select_al3_top1


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

EARNINGS_EXCLUDE_DAYS = 3

SWING_SCORE_MIN = 70.0
SWING_RR_MIN = 1.8
SWING_EV_R_MIN = 0.40

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
    near = False

    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            near = True
            when = f"{delta}日後" if delta > 0 else ("本日" if delta == 0 else "直近")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")

    if not warns:
        warns.append("- 特になし")
    return warns, near


def filter_earnings(df: pd.DataFrame, today_date) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df
    try:
        parsed = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    except Exception:
        return df

    df = df.copy()
    df["earnings_date_parsed"] = parsed

    keep = []
    for d in df["earnings_date_parsed"]:
        if d is None or pd.isna(d):
            keep.append(True)
            continue
        try:
            delta = abs((d - today_date).days)
            keep.append(delta > EARNINGS_EXCLUDE_DAYS)
        except Exception:
            keep.append(True)
    return df[keep]


def expected_r_from_in_rank(in_rank: str, rr: float) -> float:
    if rr <= 0:
        return -999.0

    if in_rank == "強IN":
        win = 0.45
    elif in_rank == "通常IN":
        win = 0.40
    elif in_rank == "弱めIN":
        win = 0.33
    else:
        win = 0.25

    lose = 1.0 - win
    return float(win * rr - lose * 1.0)


def run_swing_candidates(today_date, mkt_score: int) -> List[Dict]:
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    if "ticker" in uni.columns:
        t_col = "ticker"
    elif "code" in uni.columns:
        t_col = "code"
    else:
        return []

    uni = filter_earnings(uni, today_date)

    cands: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker, period="260d")
        if hist is None or len(hist) < 120:
            continue

        base_score = score_stock(hist)
        if base_score is None or not np.isfinite(base_score) or float(base_score) < SWING_SCORE_MIN:
            continue

        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score)
        rr = float(rr_info.get("rr", 0.0))
        if rr < SWING_RR_MIN:
            continue

        ev_r = expected_r_from_in_rank(in_rank, rr)
        if ev_r < SWING_EV_R_MIN:
            continue

        runner_label, runner_strength = evaluate_runner(hist)

        entry = float(rr_info["entry"])
        price_now = _safe_float(hist["Close"].iloc[-1], np.nan)
        gap_pct = np.nan
        if np.isfinite(price_now) and price_now > 0 and np.isfinite(entry) and entry > 0:
            gap_pct = (price_now / entry - 1.0) * 100.0

        cands.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                score=float(base_score),
                in_rank=in_rank,
                rr=rr,
                ev_r=float(ev_r),
                runner_label=runner_label,
                runner_strength=float(runner_strength),
                entry=entry,
                price_now=float(price_now) if np.isfinite(price_now) else np.nan,
                gap_pct=float(gap_pct) if np.isfinite(gap_pct) else np.nan,
                tp_pct=float(rr_info["tp_pct"]),
                sl_pct=float(rr_info["sl_pct"]),
                tp_price=float(rr_info["tp_price"]),
                sl_price=float(rr_info["sl_price"]),
            )
        )

    cands.sort(key=lambda x: (x["runner_strength"] * x["ev_r"] * x["rr"], x["score"]), reverse=True)
    return cands


def recommend_leverage(mkt_score: int, event_near: bool, has_al3: bool) -> Tuple[float, str]:
    if not has_al3:
        return 0.0, "新規なし（AL3不在）"

    lev = 2.3
    comment = "攻め（AL3一点集中）"

    if mkt_score < 50:
        lev = 2.0
        comment = "攻め（AL3一点集中 / 地合い<50で2.0x）"
    elif mkt_score >= 70:
        lev = 2.5
        comment = "強攻め（AL3一点集中 / 地合い>=70で2.5x）"

    if event_near:
        lev = min(lev, 2.0)
        comment += " / イベント近接で2.0x上限"

    return float(lev), comment


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


def _fmt_pct(p: float) -> str:
    return f"{p*100:+.1f}%"


def build_report(today_str: str, today_date, mkt: Dict, pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    sectors = top_sectors_5d(top_n=SECTOR_TOP_N)
    events, event_near = build_event_warnings(today_date)

    cands = run_swing_candidates(today_date, mkt_score)
    swing = select_al3_top1(cands)
    has_al3 = bool(swing)

    lev, lev_comment = recommend_leverage(mkt_score, event_near, has_al3)
    max_pos = calc_max_position(total_asset, lev)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing集中 / AL3一点）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- 推奨レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append(f"- イベント近接: {'YES' if event_near else 'NO'}")
    lines.append("")

    lines.append("📈 セクター（5日）")
    if sectors:
        for i, (s_name, pct) in enumerate(sectors):
            lines.append(f"{i+1}. {s_name} ({pct:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ イベント")
    for ev in events:
        lines.append(ev)
    lines.append("")

    lines.append("🏆 Swing（AL3一点集中）")
    if swing:
        c = swing[0]
        lines.append(f"  候補数:1銘柄（AL3） / RR:{c['rr']:.2f}R / EV:{c['ev_r']:.2f}R")
        lines.append("")
        lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
        lines.append(f"  AL:3  Score:{c['score']:.1f}  IN:{c['in_rank']}  Runner:{c['runner_label']}（走行強度:{c['runner_strength']:.1f}）")
        lines.append(f"  RR:{c['rr']:.2f}R  EV:{c['ev_r']:.2f}R")
        if np.isfinite(c.get('price_now', np.nan)) and np.isfinite(c.get('gap_pct', np.nan)):
            lines.append(f"  押し目基準IN:{c['entry']:.1f} / 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
        else:
            lines.append(f"  押し目基準IN:{c['entry']:.1f}")
        lines.append(f"  TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})")
        lines.append("")
    else:
        lines.append("- 該当なし（AL3不在）")
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
        r = requests.post(WORKER_URL, json={"text": ch}, timeout=20)
        print("[LINE RESULT]", r.status_code, str(r.text)[:200])


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    mkt_score = int(mkt.get("score", 50))

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=mkt_score)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        pos_text=pos_text,
        total_asset=total_asset,
    )

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()
