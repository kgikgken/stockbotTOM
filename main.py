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
from utils.day import score_daytrade_candidate
from utils.qualify import runner_strength, pullback_quality, decide_al_for_swing, al3_score, enforce_single_al3


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

EARNINGS_EXCLUDE_DAYS = 3

SWING_MAX_FINAL = 3
SWING_SCORE_MIN = 70.0
SWING_RR_MIN = 1.8
SWING_EV_R_MIN = 0.40

DAY_MAX_FINAL = 3
DAY_SCORE_MIN = 60.0
DAY_RR_MIN = 1.5

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


def fetch_intraday(ticker: str, period: str = "5d", interval: str = "5m") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
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


def build_event_warnings_and_flag(today_date) -> Tuple[List[str], bool]:
    events = load_events()
    warns: List[str] = []
    event_near = False

    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue

        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            event_near = True
            when = "直近" if delta < 0 else ("本日" if delta == 0 else f"{delta}日後")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")

    if not warns:
        warns.append("- 特になし")
    return warns, event_near


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


def base_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 2.0, "強め（押し目＋一部ブレイク）"
    if mkt_score >= 60:
        return 1.7, "やや強め（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "中立（押し目メイン）"
    if mkt_score >= 40:
        return 1.1, "弱め（新規ロット小さめ）"
    return 1.0, "かなり弱い（守り）"


def leverage_caps(mkt_score: int, event_near: bool) -> float:
    cap = 2.5
    if mkt_score < 50:
        cap = min(cap, 2.0)
    if event_near:
        cap = min(cap, 2.0)
    return float(cap)


def leverage_for_candidate(al: int, mkt_score: int, event_near: bool) -> float:
    base, _ = base_leverage(mkt_score)
    cap = leverage_caps(mkt_score, event_near)

    if al >= 3:
        return float(min(cap, max(base, 2.0)))
    if al == 2:
        return float(min(cap, max(base, 1.3)))
    return float(min(cap, base))


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


def run_swing(today_date, mkt_score: int, event_near: bool) -> List[Dict]:
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

    MIN_SCORE = float(SWING_SCORE_MIN)
    RR_MIN = float(SWING_RR_MIN)
    EV_MIN = float(SWING_EV_R_MIN)

    if mkt_score >= 70:
        MIN_SCORE -= 3.0
        RR_MIN -= 0.1
    elif mkt_score <= 45:
        MIN_SCORE += 3.0
        RR_MIN += 0.1

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

        base_sc = score_stock(hist)
        if base_sc is None or not np.isfinite(base_sc) or base_sc < MIN_SCORE:
            continue

        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue
        if mkt_score <= 45 and in_rank == "弱めIN":
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score, for_day=False)
        rr = float(rr_info.get("rr", 0.0))
        if rr < RR_MIN:
            continue

        ev_r = expected_r_from_in_rank(in_rank, rr)
        if ev_r < EV_MIN:
            continue

        entry = float(rr_info["entry"])
        price_now = _safe_float(hist["Close"].iloc[-1], np.nan)

        gap_pct = np.nan
        if np.isfinite(price_now) and price_now > 0 and np.isfinite(entry) and entry > 0:
            gap_pct = (price_now / entry - 1.0) * 100.0

        r_sc, r_label = runner_strength(hist)
        pb_sc = pullback_quality(float(gap_pct) if np.isfinite(gap_pct) else np.nan, in_rank)

        al = decide_al_for_swing(r_sc, r_label, rr, ev_r, in_rank)
        lev_cand = leverage_for_candidate(al, mkt_score, event_near)

        c = dict(
            ticker=ticker,
            name=name,
            sector=sector,
            score=float(base_sc),
            rr=float(rr),
            ev_r=float(ev_r),
            in_rank=in_rank,
            entry=entry,
            price_now=float(price_now) if np.isfinite(price_now) else np.nan,
            gap_pct=float(gap_pct) if np.isfinite(gap_pct) else np.nan,
            tp_pct=float(rr_info["tp_pct"]),
            sl_pct=float(rr_info["sl_pct"]),
            tp_price=float(rr_info["tp_price"]),
            sl_price=float(rr_info["sl_price"]),
            runner_label=str(r_label),
            runner_strength=float(r_sc),
            al=int(al),
            lev=float(lev_cand),
            pullback_quality=float(pb_sc),
        )
        c["al3_score"] = float(al3_score(c["runner_strength"], c["rr"], c["ev_r"], c["pullback_quality"]))
        cands.append(c)

    cands.sort(key=lambda x: (x["score"], x["ev_r"], x["rr"], x["runner_strength"]), reverse=True)
    cands = enforce_single_al3(cands, event_near=event_near)

    return cands[:SWING_MAX_FINAL]


def run_day(today_date, mkt_score: int, swing_picks: List[Dict]) -> List[Dict]:
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

    exclude = set(str(c.get("ticker")) for c in (swing_picks or []) if c.get("ticker"))
    out: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker or ticker in exclude:
            continue

        hist_d = fetch_history(ticker, period="180d")
        if hist_d is None or len(hist_d) < 80:
            continue

        day_sc = score_daytrade_candidate(hist_d, mkt_score=mkt_score)
        if not np.isfinite(day_sc) or day_sc < DAY_SCORE_MIN:
            continue

        hist_i = fetch_intraday(ticker, period="5d", interval="5m")
        if hist_i is None or len(hist_i) < 50:
            continue

        rr_info = compute_tp_sl_rr(hist_d, mkt_score=mkt_score, for_day=True)
        rr = float(rr_info.get("rr", 0.0))
        eff_rr = float(rr_info.get("effective_rr", rr))
        if rr < DAY_RR_MIN:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        price_now = _safe_float(hist_i["Close"].iloc[-1], np.nan)
        entry = float(rr_info["entry"])

        gap_pct = np.nan
        if np.isfinite(price_now) and price_now > 0 and np.isfinite(entry) and entry > 0:
            gap_pct = (price_now / entry - 1.0) * 100.0

        if np.isfinite(gap_pct) and gap_pct > 5.0:
            continue

        out.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                score=float(day_sc),
                rr=rr,
                effective_rr=eff_rr,
                entry=entry,
                price_now=float(price_now) if np.isfinite(price_now) else np.nan,
                gap_pct=float(gap_pct) if np.isfinite(gap_pct) else np.nan,
                tp_pct=float(rr_info["tp_pct"]),
                sl_pct=float(rr_info["sl_pct"]),
                tp_price=float(rr_info["tp_price"]),
                sl_price=float(rr_info["sl_price"]),
            )
        )

    out.sort(key=lambda x: (x["score"], x["effective_rr"], x["rr"]), reverse=True)
    return out[:DAY_MAX_FINAL]


def _fmt_pct(p: float) -> str:
    return f"{p*100:+.1f}%"


def build_report(today_str: str, today_date, mkt: Dict, pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    events, event_near = build_event_warnings_and_flag(today_date)

    swing = run_swing(today_date, mkt_score, event_near=event_near)
    day = run_day(today_date, mkt_score, swing_picks=swing)

    base_lev, base_lev_comment = base_leverage(mkt_score)
    cap = leverage_caps(mkt_score, event_near)

    lev = base_lev
    lev_comment = base_lev_comment
    if swing:
        lev = float(max([float(c.get("lev", base_lev)) for c in swing] + [base_lev]))
        lev = float(min(lev, cap))
        if event_near:
            lev_comment = f"攻め（AL3一点のみ）/イベント近接で{cap:.1f}x上限"
        else:
            lev_comment = "条件付き攻め（AL3一点集中）"
    else:
        lev = float(min(lev, cap))
        if event_near:
            lev_comment = f"{base_lev_comment} / イベント近接で{cap:.1f}x上限"

    max_pos = calc_max_position(total_asset, lev)

    sectors = top_sectors_5d(top_n=SECTOR_TOP_N)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（vAB_prime+ / 大勝ちモード）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- 推奨レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append(f"- イベント近接: {'YES' if event_near else 'NO'}")
    lines.append("補足: イベント近接→SwingはAL3一点のみ許可。" if event_near else "補足: AL3は1日1銘柄（一点集中）。")
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

    lines.append("🏆 Swing（数日〜2週）Core候補（Runner→押し目）")
    if swing:
        rr_vals = [c["rr"] for c in swing]
        ev_vals = [c["ev_r"] for c in swing]
        lines.append(f"  候補数:{len(swing)}銘柄 / 平均RR:{float(np.mean(rr_vals)):.2f}R / 平均EV:{float(np.mean(ev_vals)):.2f}R")
        lines.append("")
        for c in swing:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  AL:{c['al']} 推奨レバ:{c['lev']:.1f}x  Score:{c['score']:.1f}  IN:{c['in_rank']}  Runner:{c['runner_label']}")
            lines.append(f"  RR:{c['rr']:.2f}R  EV:{c['ev_r']:.2f}R  走行強度:{c['runner_strength']:.1f}")
            if np.isfinite(c.get('price_now', np.nan)) and np.isfinite(c.get('gap_pct', np.nan)):
                lines.append(f"  押し目基準IN:{c['entry']:.1f} / 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
            else:
                lines.append(f"  押し目基準IN:{c['entry']:.1f}")
            lines.append(f"  TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})")
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    lines.append("⚡ Day（デイトレ）候補（追い禁止 + 実効RR表示 / Swing採用銘柄は除外）")
    if day:
        rr_vals = [c["rr"] for c in day]
        eff_vals = [c.get("effective_rr", c["rr"]) for c in day]
        lines.append(f"  候補数:{len(day)}銘柄 / 平均RR:{float(np.mean(rr_vals)):.2f}R（実効:{float(np.mean(eff_vals)):.2f}R）")
        lines.append("")
        for c in day:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R（実効:{c.get('effective_rr', c['rr']):.2f}R）")
            if np.isfinite(c.get("price_now", np.nan)) and np.isfinite(c.get("gap_pct", np.nan)):
                lines.append(f"  Day基準IN:{c['entry']:.1f} / 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
            else:
                lines.append(f"  Day基準IN:{c['entry']:.1f}")
            lines.append(f"  TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})")
            lines.append("")
    else:
        lines.append("- 該当なし")
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

    report = build_report(today_str, today_date, mkt, pos_text, total_asset)
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()
