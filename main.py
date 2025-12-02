from __future__ import annotations

import os
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


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
WORKER_URL = os.getenv("WORKER_URL")

EARNINGS_EXCLUDE_DAYS = 3


# ============================
# JST date
# ============================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


# ============================
# Event warnings
# ============================
EVENT_CALENDAR: List[Dict[str, str]] = []


def build_event_warnings(today: datetime.date) -> List[str]:
    warns = []
    for ev in EVENT_CALENDAR:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except:
            continue
        delta = (d - today).days

        if -1 <= delta <= 2:
            if delta > 0:
                when = f"{delta}日後"
            elif delta == 0:
                when = "本日"
            else:
                when = "直近"
            warns.append(f"⚠ {ev['label']}（{when}）: ポジションサイズ注意")

    return warns


# ============================
# Universe loader
# ============================
def load_universe(path: str = UNIVERSE_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except:
        return None

    if "ticker" not in df.columns:
        return None

    df["ticker"] = df["ticker"].astype(str)

    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


def in_earnings_window(row: pd.Series, today: datetime.date) -> bool:
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False

    try:
        delta = abs((d - today).days)
        return delta <= EARNINGS_EXCLUDE_DAYS
    except:
        return False


# ============================
# yfinance
# ============================
def fetch_history(ticker: str, period="130d") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# ============================
# Leverage / Lots
# ============================
def calc_target_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 2.0, "攻め（Aランク3銘柄フル）"
    if mkt_score >= 50:
        return 1.3, "標準（押し目のみ）"
    if mkt_score >= 40:
        return 1.0, "守り寄り"
    return 0.8, "守り優先"


def calc_lot_for_stock(price, total_asset, lev, slots=3):
    if not (np.isfinite(price) and price > 0):
        return 0
    if not (np.isfinite(total_asset) and total_asset > 0):
        return 0

    per_value = total_asset * lev / slots
    raw_shares = per_value // price
    lots_100 = int(raw_shares // 100)
    return max(lots_100 * 100, 0)


# ============================
# TP / SL
# ============================
def calc_candidate_tp_sl(price, vola20, mkt_score):
    if price <= 0:
        return 0.0, 0.0, price, price

    v = vola20 if vola20 is not None else 0.04

    if v < 0.02:
        tp, sl = 0.08, -0.03
    elif v > 0.06:
        tp, sl = 0.12, -0.06
    else:
        tp, sl = 0.10, -0.04

    if mkt_score >= 70:
        tp += 0.02
    elif mkt_score < 45:
        tp -= 0.02
        sl = max(sl, -0.03)

    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))

    return tp, sl, price * (1 + tp), price * (1 + sl)


# ============================
# Screening
# ============================
def run_screening(today, mkt_score, total_asset, lev):
    df = load_universe()
    if df is None:
        return [], []

    A_list, B_list = [], []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        sc = score_stock(hist)
        if sc is None or not np.isfinite(sc):
            continue

        close = hist["Close"].astype(float)
        price = float(close.iloc[-1])

        ret = close.pct_change()
        vola20 = float(ret.rolling(20).std().iloc[-1]) if len(ret) > 20 else None

        tp_pct, sl_pct, tp_price, sl_price = calc_candidate_tp_sl(
            price, vola20, mkt_score
        )

        lot = calc_lot_for_stock(price, total_asset, lev)

        info = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": float(sc),
            "price": price,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "lot": lot,
        }

        if sc >= 85:
            A_list.append(info)
        elif sc >= 75:
            B_list.append(info)

    A_list.sort(key=lambda x: x["score"], reverse=True)
    B_list.sort(key=lambda x: x["score"], reverse=True)
    return A_list, B_list


def select_primary_targets(A, B, n=3):
    if len(A) >= n:
        return A[:n], []
    if len(A) > 0:
        need = n - len(A)
        return A + B[:need], B[need:]
    return B[:n], B[n:]


# ============================
# Report builders
# ============================
def build_core_report(today_str, today, mkt, total_asset):
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = mkt.get("comment", "")
    lev, lev_label = calc_target_leverage(mkt_score)

    secs = top_sectors_5d()
    if secs:
        sec_text = "\n".join([f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)])
    else:
        sec_text = "算出不可（データ不足）"

    A_list, B_list = run_screening(today, mkt_score, total_asset, lev)
    primary, _ = select_primary_targets(A_list, B_list, n=3)

    events = build_event_warnings(today)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推奨レバ: 約{lev:.1f}倍（{lev_label}）")
    lines.append(f"- 推定運用資産ベース: 約{int(total_asset):,}円")

    for ev in events:
        lines.append(ev)

    lines.append("\n◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sec_text + "\n")

    lines.append("◆ Core候補 Aランク（本命押し目・最大3銘柄）")
    if not primary:
        lines.append("Aランク該当なし")
    else:
        for r in primary:
            lines.append(
                f"- {r['ticker']} {r['name']} Score:{r['score']:.1f} 現値:{r['price']:.1f}\n"
                f"    ・IN目安: {r['price']:.1f}\n"
                f"    ・利確目安: +{r['tp_pct']*100:.1f}%（{r['tp_price']:.1f}）\n"
                f"    ・損切り目安: {r['sl_pct']*100:.1f}%（{r['sl_price']:.1f}）\n"
                f"    ・推奨ロット: {r['lot']}株"
            )

    return "\n".join(lines)


def build_position_report(today_str, pos_text):
    return f"📊 {today_str} ポジション分析\n\n{pos_text}"


# ============================
# LINE
# ============================
def send_line(text: str):
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定")
        print(text)
        return

    try:
        r = requests.post(WORKER_URL, json={"text": text}, timeout=10)
        print("[LINE RESULT]", r.status_code, r.text)
    except Exception as e:
        print("[ERROR] LINE送信失敗:", e)
        print(text)


# ============================
# Main
# ============================
def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = calc_market_score()

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, _, _, _ = analyze_positions(pos_df)

    core = build_core_report(today_str, today_date, mkt, total_asset)
    posrep = build_position_report(today_str, pos_text)

    print(core)
    print("\n==============================\n")
    print(posrep)

    send_line(core)
    send_line(posrep)


if __name__ == "__main__":
    main()