
from __future__ import annotations

import os
import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.scoring import add_indicators, score_stock, calc_in_rank, trend_strength
from utils.rr import compute_tp_sl_rr
from utils.qualify import trend_gate
from utils.position import load_positions, analyze_positions


# =====================
# 設定
# =====================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

EARNINGS_EXCLUDE_DAYS = 3

# Swing（順張り専用 / 逆張り排除）
AL3_TOP_N = 5
SCORE_MIN = 72.0
RR_MIN = 1.8
EV_MIN = 0.30

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


# =====================
# events.csv
# =====================
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


def build_event_warnings(today_date) -> List[str]:
    events = load_events()
    warns: List[str] = []
    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        delta = (dt.date() - today_date).days
        if -1 <= delta <= 2:
            when = "本日" if delta == 0 else ("直近" if delta < 0 else f"{delta}日後")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")
    if not warns:
        warns.append("- 特になし")
    return warns


def is_event_near(today_date) -> bool:
    events = load_events()
    for ev in events:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        delta = (dt.date() - today_date).days
        if -1 <= delta <= 2:
            return True
    return False


# =====================
# 決算フィルタ
# =====================
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
            keep.append(abs((d - today_date).days) > EARNINGS_EXCLUDE_DAYS)
        except Exception:
            keep.append(True)
    return df[keep]


# =====================
# EV（暫定）
# =====================
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
    return float(win * rr - (1.0 - win) * 1.0)


# =====================
# AL3（攻める価値指数）
# =====================
def al3_score(score: float, rr: float, ev_r: float, trend_str: float) -> float:
    s = 0.0
    s += (trend_str / 100.0) * 2.2
    s += float(np.clip(ev_r, -1.0, 3.0)) * 1.2
    s += float(np.clip(rr, 0.0, 8.0)) * 0.25
    s += (float(np.clip(score, 0.0, 100.0)) / 100.0) * 1.0
    return float(s)


# =====================
# Swing（順張り専用）
# =====================
def run_swing(today_date, mkt_score: int) -> List[Dict]:
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

    out: List[Dict] = []
    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker, period="260d")
        if hist is None or len(hist) < 120:
            continue

        ok, _diag = trend_gate(hist)
        if not ok:
            continue

        df = add_indicators(hist)

        sc = score_stock(df)
        if sc is None or not np.isfinite(sc) or sc < SCORE_MIN:
            continue

        inr = calc_in_rank(df)
        if inr == "様子見":
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score)
        rr = float(rr_info["rr"])
        if not np.isfinite(rr) or rr < RR_MIN:
            continue

        ev = expected_r_from_in_rank(inr, rr)
        if not np.isfinite(ev) or ev < EV_MIN:
            continue

        tr = trend_strength(df)
        al3 = al3_score(sc, rr, ev, tr)

        price_now = _safe_float(hist["Close"].iloc[-1], np.nan)
        entry = float(rr_info["entry"])
        gap_pct = np.nan
        if np.isfinite(price_now) and price_now > 0 and np.isfinite(entry) and entry > 0:
            gap_pct = (price_now / entry - 1.0) * 100.0

        out.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                score=float(sc),
                in_rank=inr,
                rr=float(rr),
                ev_r=float(ev),
                trend=float(tr),
                al3=float(al3),
                entry=float(entry),
                price_now=float(price_now) if np.isfinite(price_now) else np.nan,
                gap_pct=float(gap_pct) if np.isfinite(gap_pct) else np.nan,
                tp_pct=float(rr_info["tp_pct"]),
                sl_pct=float(rr_info["sl_pct"]),
                tp_price=float(rr_info["tp_price"]),
                sl_price=float(rr_info["sl_price"]),
            )
        )

    out.sort(key=lambda x: (x["al3"], x["ev_r"], x["rr"]), reverse=True)
    return out[:AL3_TOP_N]


# =====================
# レバ（Swing専念）
# =====================
def recommend_leverage(mkt_score: int, event_near: bool) -> tuple[float, str]:
    if mkt_score >= 70:
        lev = 2.5
        txt = "強気（順張り押し目）"
    elif mkt_score >= 60:
        lev = 2.2
        txt = "攻め（順張り押し目）"
    elif mkt_score >= 50:
        lev = 2.0
        txt = "中立（AL3のみ攻め）"
    else:
        lev = 1.6
        txt = "弱め（厳選AL3のみ）"

    if event_near:
        lev = min(lev, 2.0)
        txt += " / イベント近接で2.0x上限"
    return float(lev), txt


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


def _fmt_pct(p: float) -> str:
    return f"{p*100:+.1f}%"


def build_report(today_str: str, today_date, mkt: Dict, pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    event_near = is_event_near(today_date)

    lev, lev_comment = recommend_leverage(mkt_score, event_near)
    max_pos = calc_max_position(total_asset, lev)

    sectors = top_sectors_5d(top_n=SECTOR_TOP_N)
    events = build_event_warnings(today_date)
    swing = run_swing(today_date, mkt_score)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専念 / TrendGate）")
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

    lines.append(f"🏆 Swing（AL3 Top{AL3_TOP_N} / 逆張り排除）")
    if swing:
        evs = [c["ev_r"] for c in swing]
        rrs = [c["rr"] for c in swing]
        lines.append(f"  候補数:{len(swing)}銘柄 / 平均RR:{float(np.mean(rrs)):.2f}R / 平均EV:{float(np.mean(evs)):.2f}R")
        lines.append("")
        for i, c in enumerate(swing, start=1):
            star = " ⭐" if i == 1 else ""
            lines.append(f"{i}. {c['ticker']} {c['name']}{star} [{c['sector']}]")
            lines.append(f"   AL3:{c['al3']:.2f}  Score:{c['score']:.1f}  IN:{c['in_rank']}  Trend:{c['trend']:.1f}")
            lines.append(f"   RR:{c['rr']:.2f}R  EV:{c['ev_r']:.2f}R")
            if np.isfinite(c.get('price_now', np.nan)) and np.isfinite(c.get('gap_pct', np.nan)):
                lines.append(f"   押し目IN:{c['entry']:.1f} / 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
            else:
                lines.append(f"   押し目IN:{c['entry']:.1f}")
            lines.append(f"   TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})")
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")
    return "\\n".join(lines)


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
