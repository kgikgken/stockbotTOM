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
from utils.scoring import score_stock
from utils.position import load_positions, analyze_positions
from utils.day import score_daytrade_candidate
from utils.rr import compute_tp_sl_rr
from utils.qualify import qualify_swing, day_event_ok


# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算前後の除外
EARNINGS_EXCLUDE_DAYS = 3

# Swing/Day 出力数
SWING_MAX_FINAL = 3
DAY_MAX_FINAL = 3

# vAB：Day は事件がない日は0銘柄OK
DAY_REQUIRE_EVENT = True

# 表示
SECTOR_TOP_N = 5


# ============================================================
# 便利
# ============================================================
def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _fmt_pct(p: float) -> str:
    return f"{p*100:+.1f}%"


def fetch_history(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.35)
    return None


def fetch_intraday(ticker: str, period: str = "5d", interval: str = "5m") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.35)
    return None


# ============================================================
# events.csv
# ============================================================
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
        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            when = "直近" if delta < 0 else ("本日" if delta == 0 else f"{delta}日後")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev['label']}（{dt_disp} / {when}）")
    if not warns:
        warns.append("- 特になし")
    return warns


def has_near_event(today_date) -> bool:
    for ev in load_events():
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        delta = (dt.date() - today_date).days
        if -1 <= delta <= 2:
            return True
    return False


# ============================================================
# 決算フィルタ
# ============================================================
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


# ============================================================
# レバ（vAB）
# ============================================================
def recommend_base_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 2.5, "強気（攻め）"
    if mkt_score >= 60:
        return 2.2, "やや強気（攻め）"
    if mkt_score >= 50:
        return 1.7, "中立（選別して攻め）"
    if mkt_score >= 40:
        return 1.3, "弱め（守り）"
    return 1.0, "弱い（守り）"


def cap_leverage(mkt_score: int, lev: float) -> Tuple[float, str]:
    note = ""
    lev = float(min(lev, 2.5))
    if mkt_score < 50 and lev > 2.0:
        lev = 2.0
        note = "地合い<50で2.0x上限"
    return lev, note


def lev_by_al(al: int) -> float:
    if al >= 3:
        return 2.3
    if al == 2:
        return 1.7
    return 1.3


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# Swing（vAB）
# ============================================================
def run_swing_vab(today_date, mkt_score: int) -> List[Dict]:
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []
    t_col = "ticker" if "ticker" in uni.columns else ("code" if "code" in uni.columns else None)
    if not t_col:
        return []

    uni = filter_earnings(uni, today_date)

    cands: List[Dict] = []
    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        hist = fetch_history(ticker, period="260d")
        if hist is None or len(hist) < 220:
            continue

        ok, reason, payload = qualify_swing(hist, mkt_score=mkt_score)
        if not ok:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        sc = score_stock(hist)
        sc = float(sc) if sc is not None and np.isfinite(sc) else 0.0

        al = int(payload["al"])
        lev = lev_by_al(al)

        cands.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                score=sc,
                al=al,
                lev=lev,
                in_rank=payload["in_rank"],
                rr=float(payload["rr"]),
                ev_r=float(payload["ev_r"]),
                resistance=int(payload["resistance"]),
                entry=float(payload["entry"]),
                price_now=float(payload["price_now"]),
                gap_pct=float(payload["gap_pct"]),
                tp_pct=float(payload["tp_pct"]),
                sl_pct=float(payload["sl_pct"]),
                tp_price=float(payload["tp_price"]),
                sl_price=float(payload["sl_price"]),
                trail_pct=float(payload["trail_pct"]),
                trail_price=float(payload["trail_price"]),
            )
        )

    cands.sort(key=lambda x: (x["al"], x["ev_r"], x["rr"], x["score"]), reverse=True)
    return cands[:SWING_MAX_FINAL]


# ============================================================
# Day（vAB）
# ============================================================
def run_day_vab(today_date, mkt_score: int, exclude_tickers: set[str]) -> List[Dict]:
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []
    t_col = "ticker" if "ticker" in uni.columns else ("code" if "code" in uni.columns else None)
    if not t_col:
        return []

    uni = filter_earnings(uni, today_date)

    out: List[Dict] = []
    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker or ticker in exclude_tickers:
            continue

        hist_d = fetch_history(ticker, period="260d")
        if hist_d is None or len(hist_d) < 120:
            continue

        if DAY_REQUIRE_EVENT:
            ok_event, _ = day_event_ok(hist_d)
            if not ok_event:
                continue

        day_score = score_daytrade_candidate(hist_d, mkt_score=mkt_score)
        if not np.isfinite(day_score) or day_score < 60.0:
            continue

        rr_info = compute_tp_sl_rr(hist_d, mkt_score=mkt_score, for_day=True)
        rr = float(rr_info.get("rr", 0.0))
        if not np.isfinite(rr) or rr < 1.5:
            continue

        hist_i = fetch_intraday(ticker, period="5d", interval="5m")
        if hist_i is None or len(hist_i) < 30:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        price_now = _safe_float(hist_i["Close"].iloc[-1], np.nan)
        entry = float(rr_info.get("entry", 0.0))

        gap_pct = np.nan
        if np.isfinite(price_now) and price_now > 0 and np.isfinite(entry) and entry > 0:
            gap_pct = (price_now / entry - 1.0) * 100.0

        out.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                score=float(day_score),
                rr=float(rr),
                rr_eff=float(rr * 0.70),
                entry=float(entry),
                price_now=float(price_now) if np.isfinite(price_now) else np.nan,
                gap_pct=float(gap_pct) if np.isfinite(gap_pct) else np.nan,
                tp_pct=float(rr_info.get("tp_pct", 0.0)),
                sl_pct=float(rr_info.get("sl_pct", 0.0)),
                tp_price=float(rr_info.get("tp_price", 0.0)),
                sl_price=float(rr_info.get("sl_price", 0.0)),
            )
        )

    out.sort(key=lambda x: (x["score"], x["rr_eff"]), reverse=True)
    return out[:DAY_MAX_FINAL]


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print(text)
        return
    chunk_size = 3800
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]
    for ch in chunks:
        r = requests.post(WORKER_URL, json={"text": ch}, timeout=20)
        print("[LINE RESULT]", r.status_code, str(r.text)[:200])


# ============================================================
# レポート
# ============================================================
def build_report(today_str: str, today_date, mkt: Dict, pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    base_lev, base_comment = recommend_base_leverage(mkt_score)
    base_lev, cap_note = cap_leverage(mkt_score, base_lev)
    max_pos = calc_max_position(total_asset, base_lev)

    sectors = top_sectors_5d(top_n=SECTOR_TOP_N)
    events = build_event_warnings(today_date)
    near_event = has_near_event(today_date)

    swing = run_swing_vab(today_date, mkt_score)

    # イベント近接はAL3一点のみ許可
    if near_event and swing:
        swing = [c for c in swing if c.get("al", 0) >= 3][:1]

    exclude = set([c["ticker"] for c in swing]) if swing else set()
    day = run_day_vab(today_date, mkt_score, exclude_tickers=exclude)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（vAB / 大勝ちモード）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    if cap_note:
        lines.append(f"- 推奨レバ: {base_lev:.1f}倍（{base_comment} / {cap_note}）")
    else:
        lines.append(f"- 推奨レバ: {base_lev:.1f}倍（{base_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append(f"- イベント近接: {'YES' if near_event else 'NO'}")
    if near_event:
        lines.append("補足: イベント近接→AL3一点のみ許可。")
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

    # --- SWING ---
    lines.append("🏆 Swing（数日〜2週）Core候補（vAB：走行能力A→押し目B）")
    if swing:
        rr_vals = [c["rr"] for c in swing if np.isfinite(c["rr"])]
        ev_vals = [c["ev_r"] for c in swing if np.isfinite(c["ev_r"])]
        avg_rr = float(np.mean(rr_vals)) if rr_vals else 0.0
        avg_ev = float(np.mean(ev_vals)) if ev_vals else 0.0
        lines.append(f"  候補数:{len(swing)}銘柄 / 平均RR:{avg_rr:.2f}R / 平均EV:{avg_ev:.2f}R")
        lines.append("")
        for c in swing:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  AL:{c['al']} 推奨レバ:{c['lev']:.1f}x  Score:{c['score']:.1f}  IN:{c['in_rank']}")
            lines.append(f"  RR:{c['rr']:.2f}R  EV:{c['ev_r']:.2f}R  抵抗:{c['resistance']}")
            if np.isfinite(c.get('price_now', np.nan)) and np.isfinite(c.get('gap_pct', np.nan)):
                lines.append(f"  押し目基準IN:{c['entry']:.1f} / 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
            else:
                lines.append(f"  押し目基準IN:{c['entry']:.1f}")
            lines.append(f"  初期SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})")
            lines.append(f"  TRAIL:{_fmt_pct(-c['trail_pct'])} ({c['trail_price']:.1f})")
            lines.append(f"  参考TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})")
            lines.append("")
    else:
        lines.append("- 該当なし（vABは“走る銘柄だけ”残す）")
        lines.append("")

    # --- DAY ---
    lines.append("⚡ Day（デイトレ）候補（vAB：事件がない日は0銘柄OK / Swing採用銘柄は除外）")
    if day:
        rr_vals = [c["rr"] for c in day if np.isfinite(c["rr"])]
        rr_eff_vals = [c["rr_eff"] for c in day if np.isfinite(c["rr_eff"])]
        avg_rr = float(np.mean(rr_vals)) if rr_vals else 0.0
        avg_rr_eff = float(np.mean(rr_eff_vals)) if rr_eff_vals else 0.0
        lines.append(f"  候補数:{len(day)}銘柄 / 平均RR:{avg_rr:.2f}R（実効:{avg_rr_eff:.2f}R）")
        lines.append("")
        for c in day:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R（実効:{c['rr_eff']:.2f}R）")
            if np.isfinite(c.get("price_now", np.nan)) and np.isfinite(c.get("gap_pct", np.nan)):
                lines.append(f"  Day基準IN:{c['entry']:.1f} / 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
            else:
                lines.append(f"  Day基準IN:{c['entry']:.1f}")
            lines.append(f"  TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})")
            lines.append("")
    else:
        lines.append("- 該当なし（事件条件を満たさず/または除外）")
        lines.append("")

    # --- POS ---
    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
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
