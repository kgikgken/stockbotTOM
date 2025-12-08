from __future__ import annotations
import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.util import jst_today_str


# ============================================================
# 基本設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

SCREENING_TOP_N = 10
MAX_FINAL_STOCKS = 3
EARNINGS_EXCLUDE_DAYS = 3


# ============================================================
# 日付処理
# ============================================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


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

    events = []
    for _, row in df.iterrows():
        d = str(row.get("date", "")).strip()
        label = str(row.get("label", "")).strip()
        kind = str(row.get("kind", "")).strip()
        if not d or not label:
            continue
        events.append({"date": d, "label": label, "kind": kind})
    return events


def build_event_warnings(today: datetime.date) -> List[str]:
    events = load_events()
    warns = []
    for ev in events:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except Exception:
            continue

        delta = (d - today).days
        if -1 <= delta <= 2:
            when = "本日" if delta == 0 else ("直近" if delta < 0 else f"{delta}日後")
            warns.append(f"- {ev['label']}（{when}）")
    return warns


# ============================================================
# Universe
# ============================================================
def load_universe(path: str = UNIVERSE_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "ticker" not in df.columns:
        return None

    df["ticker"] = df["ticker"].astype(str)

    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce").dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


def in_earnings_window(row: pd.Series, today: datetime.date) -> bool:
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    return abs((d - today).days) <= EARNINGS_EXCLUDE_DAYS


# ============================================================
# yfinance
# ============================================================
def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(1)
    return None


# ============================================================
# テクニカル
# ============================================================
def calc_ma(close: pd.Series, window: int) -> float:
    if len(close) < window:
        return float(close.iloc[-1])
    return float(close.rolling(window).mean().iloc[-1])


def calc_rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) <= period + 1:
        return 50.0
    diff = close.diff(1)
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    v = float(rsi.iloc[-1])
    return v if np.isfinite(v) else 50.0


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) <= period + 1:
        return 0.0
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else 0.0


def calc_volatility(close: pd.Series, window: int = 20) -> float:
    if len(close) < window + 1:
        return 0.03
    ret = close.pct_change(fill_method=None)
    v = ret.rolling(window).std().iloc[-1]
    return float(v) if np.isfinite(v) else 0.03


# ============================================================
# 地合い / レバ
# ============================================================
def recommend_leverage(mkt_score: int) -> float:
    if mkt_score >= 70: return 1.8
    if mkt_score >= 60: return 1.5
    if mkt_score >= 50: return 1.3
    if mkt_score >= 40: return 1.1
    return 1.0


def calc_max_position(total_asset: float, lev: float) -> int:
    if total_asset <= 0: return 0
    return int(round(total_asset * lev))


def dynamic_min_score(mkt_score: int) -> float:
    if mkt_score >= 70: return 72.0
    if mkt_score >= 60: return 75.0
    if mkt_score >= 50: return 78.0
    if mkt_score >= 40: return 80.0
    return 82.0


# ============================================================
# セクター強度
# ============================================================
def build_sector_strength_map() -> Dict[str, float]:
    secs = top_sectors_5d()
    strength = {}
    for rank, (name, chg) in enumerate(secs[:5]):
        strength[name] = (6 - rank) + max(chg, 0) * 0.3
    return strength


# ============================================================
# スクリーニング
# ============================================================
def score_candidate(
    ticker: str,
    name: str,
    sector: str,
    hist: pd.DataFrame,
    score_raw: float,
    mkt_score: int,
    sector_strength: Dict[str, float],
) -> Dict:

    close = hist["Close"].astype(float)
    price = float(close.iloc[-1])

    ma5 = calc_ma(close, 5)
    ma20 = calc_ma(close, 20)
    ma60 = calc_ma(close, 60)
    rsi = calc_rsi(close, 14)
    atr = calc_atr(hist)
    vola20 = calc_volatility(close, 20)

    setup_score = 0.0
    if ma5 > ma20 > ma60: setup_score += 12.0
    if 40 <= rsi <= 65: setup_score += 10.0
    if vola20 < 0.02: setup_score += 5.0

    regime_score = (mkt_score - 50) * 0.12
    regime_score += sector_strength.get(sector, 0.0)

    total_score = score_raw * 0.7 + setup_score * 1.0 + regime_score * 0.6

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "price": price,
        "score": float(total_score),
        "ma5": ma5,
        "ma20": ma20,
        "atr": atr,
        "vola20": vola20,
        "hist": hist,
    }


def compute_entry_price(close, ma5, ma20, atr):
    price = float(close.iloc[-1])
    target = ma20 - atr * 0.5
    if price > ma5 > ma20:
        target = ma20 + (ma5 - ma20) * 0.3
    if target > price:
        target = price * 0.995
    return round(float(target), 1)


def calc_candidate_tp_sl(vola20: float, mkt_score: int) -> Tuple[float, float]:
    v = abs(vola20)
    tp = 0.08; sl = -0.04
    if v < 0.015: tp, sl = 0.06, -0.03
    elif v > 0.03: tp, sl = 0.1, -0.05
    if mkt_score >= 70: tp += 0.02
    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))
    return tp, sl


def run_screening(today: datetime.date, mkt_score: int) -> List[Dict]:
    df = load_universe()
    if df is None:
        return []
    min_score = dynamic_min_score(mkt_score)
    sector_strength = build_sector_strength_map()

    cands = []
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if in_earnings_window(row, today):
            continue
        name = str(row.get("name", ticker))
        sector = str(row.get("sector", ""))
        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue
        base = score_stock(hist)
        if base is None or base < min_score:
            continue
        info = score_candidate(
            ticker, name, sector, hist, base, mkt_score, sector_strength)
        close = info["hist"]["Close"].astype(float)
        entry = compute_entry_price(close, info["ma5"], info["ma20"], info["atr"])
        tp_pct, sl_pct = calc_candidate_tp_sl(info["vola20"], mkt_score)
        price = float(info["price"])
        gap_ratio = abs(price - entry) / price if price > 0 else 1

        entry_type = "today" if gap_ratio <= 0.01 else "soon"
        RR = round(abs(tp_pct / sl_pct), 2)  # R計算

        cands.append({
            **info,
            "entry": entry,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "RR": RR,
            "entry_type": entry_type,
        })

    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands[:MAX_FINAL_STOCKS]


# ============================================================
# 行動ロジック
# ============================================================
def behavior_decision(final_list: List[Dict]) -> Tuple[str, str]:
    if not final_list:
        return "Wait", "波がない日。指値だけ置いて待つ。"

    today = [c for c in final_list if c["entry_type"] == "today"]
    if today:
        best = today[0]
        if best["RR"] >= 1.8:
            return "Core IN", f"{best['ticker']} RR={best['RR']}R 本命波"
        else:
            return "Light IN", f"{best['ticker']} RR={best['RR']}R ロット小さく"

    return "Wait", "押し目待ち。焦って建てない。"


# ============================================================
# レポート表示
# ============================================================
def build_report(
    today_str: str,
    today_date: datetime.date,
    mkt: Dict,
    total_asset: float,
    pos_text: str,
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = mkt.get("comment", "")
    rec_lev = recommend_leverage(mkt_score)
    est_asset = total_asset
    max_pos = calc_max_position(total_asset, rec_lev)

    secs = top_sectors_5d()
    sec_text = "\n".join([
        f"{i+1}. {name} ({chg:+.2f}%)"
        for i, (name, chg) in enumerate(secs[:3])
    ])

    events = build_event_warnings(today_date)
    if not events:
        events = ["- 特になし"]

    final_list = run_screening(today_date, mkt_score)
    today_list = [c for c in final_list if c["entry_type"] == "today"]
    soon_list = [c for c in final_list if c["entry_type"] == "soon"]

    behavior, advice = behavior_decision(final_list)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM")
    lines.append("")
    lines.append("🎯 **結論**")
    lines.append(f"地合い: {mkt_score}点")
    lines.append(f"レバ: {rec_lev}x")
    lines.append(f"コメント: {mkt_comment}")
    lines.append(f"資産: {est_asset:,.0f}円 / MAX建玉:{max_pos:,.0f}円")
    lines.append("")
    lines.append("📈 **セクター（5日）**")
    lines.append(sec_text)
    lines.append("")
    lines.append("⚠ **イベント**")
    lines.extend(events)
    lines.append("")

    lines.append(f"🏆 **Core候補（Aランク）** 最大{MAX_FINAL_STOCKS}銘柄")
    lines.append("")
    lines.append("🔥 今日IN（押し目入り）")
    if today_list:
        for c in today_list:
            lines.append(f"{c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"Score:{c['score']:.1f} 現値:{c['price']:.1f}")
            lines.append(f"IN:{c['entry']:.1f} TP:+{c['tp_pct']*100:.1f}% SL:{c['sl_pct']*100:.1f}% RR:{c['RR']:.2f}R")
            lines.append("")
    else:
        lines.append("- なし")
    lines.append("")

    lines.append("⏳ 数日以内IN（押し目待ち）")
    if soon_list:
        for c in soon_list:
            lines.append(f"{c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"Score:{c['score']:.1f} 現値:{c['price']:.1f}")
            lines.append(f"IN:{c['entry']:.1f} TP:+{c['tp_pct']*100:.1f}% SL:{c['sl_pct']*100:.1f}% RR:{c['RR']:.2f}R")
            lines.append("")
    else:
        lines.append("- なし")
    lines.append("")

    lines.append("📊 **ポジション**")
    lines.append(pos_text)
    lines.append("")

    lines.append("🐳 **今日の行動**")
    lines.append(f"{behavior}: {advice}")

    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print(text)
        return

    chunks = [text[i:i+3900] for i in range(0, len(text), 3900)] or [""]
    for ch in chunks:
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=15)
            print("LINE:", r.status_code)
        except Exception as e:
            print("LINE error:", e)


# ============================================================
# main
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = calc_market_score()
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, *_ = analyze_positions(pos_df)

    if not np.isfinite(total_asset) or total_asset <= 0:
        total_asset = 2_000_000.0

    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        total_asset=total_asset,
        pos_text=pos_text,
    )
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()