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


# ============================================================
# JST 日付
# ============================================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


# ============================================================
# Universe 読み込み
# ============================================================
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
    except:
        return False
    return delta <= EARNINGS_EXCLUDE_DAYS


# ============================================================
# 株価取得
# ============================================================
def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period)
    except:
        return None
    if df is None or df.empty:
        return None
    return df


# ============================================================
# TP / SL 計算
# ============================================================
def calc_candidate_tp_sl(
    price: float,
    vola20: Optional[float],
    mkt_score: int,
) -> Tuple[float, float, float, float]:

    if price <= 0 or not np.isfinite(price):
        return 0.0, 0.0, price, price

    v = float(vola20) if (vola20 is not None and np.isfinite(vola20)) else 0.04

    if v < 0.02:
        tp = 0.08
        sl = -0.03
    elif v > 0.06:
        tp = 0.12
        sl = -0.06
    else:
        tp = 0.10
        sl = -0.04

    if mkt_score >= 70:
        tp += 0.02
    elif mkt_score < 45:
        tp -= 0.02
        sl = max(sl, -0.03)

    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))

    tp_price = price * (1.0 + tp)
    sl_price = price * (1.0 + sl)

    return tp, sl, tp_price, sl_price


# ============================================================
# スクリーニング本体
# ============================================================
def run_screening(
    today: datetime.date,
    mkt_score: int,
) -> Tuple[List[Dict], List[Dict]]:
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return [], []

    A_list: List[Dict] = []
    B_list: List[Dict] = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        sc = score_stock(hist)
        if sc is None or not np.isfinite(sc):
            continue

        close = hist["Close"].astype(float)
        price = float(close.iloc[-1])

        ret = close.pct_change(fill_method=None)
        vola20 = float(ret.rolling(20).std().iloc[-1]) if len(ret) >= 20 else None

        tp_pct, sl_pct, tp_price, sl_price = calc_candidate_tp_sl(
            price, vola20, mkt_score
        )

        info = {
            "ticker": ticker,
            "name": name,
            "score": float(sc),
            "price": price,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }

        if sc >= 85:
            A_list.append(info)
        elif sc >= 75:
            B_list.append(info)

    A_list.sort(key=lambda x: x["score"], reverse=True)
    B_list.sort(key=lambda x: x["score"], reverse=True)

    return A_list, B_list


def select_primary_targets(
    A_list: List[Dict],
    B_list: List[Dict],
    max_names: int = 3,
) -> Tuple[List[Dict], List[Dict]]:

    if len(A_list) >= max_names:
        return A_list[:max_names], []

    if len(A_list) > 0:
        need = max_names - len(A_list)
        return A_list + B_list[:need], B_list[need:]

    return B_list[:max_names], B_list[max_names:]


# ============================================================
# レポート構築
# ============================================================
def build_core_report(
    today_str: str,
    today_date: datetime.date,
    mkt: Dict,
    total_asset: float,
) -> str:

    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))

    secs = top_sectors_5d()
    if secs:
        sec_text = "\n".join(
            [f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)]
        )
    else:
        sec_text = "算出不可（データ不足）"

    A_list, B_list = run_screening(
        today=today_date,
        mkt_score=mkt_score,
    )

    primary, rest_B = select_primary_targets(A_list, B_list, max_names=3)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推定運用資産ベース: 約{int(total_asset):,}円\n")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sec_text + "\n")

    lines.append("◆ Core候補 Aランク（本命押し目・最大3銘柄)")
    if not primary:
        lines.append("本命Aランク候補なし\n")
    else:
        for r in primary:
            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{r['price']:.1f}"
            )
            lines.append(
                f"    ・IN目安: {r['price']:.1f}\n"
                f"    ・利確目安: +{r['tp_pct']*100:.1f}%（{r['tp_price']:.1f}）\n"
                f"    ・損切り目安: {r['sl_pct']*100:.1f}%（{r['sl_price']:.1f}）\n"
            )

    return "\n".join(lines)


def build_position_report(today_str: str, pos_text: str) -> str:
    lines = []
    lines.append(f"📊 {today_str} ポジション分析\n")
    lines.append("◆ ポジションサマリ")
    lines.append(pos_text.strip())
    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str):
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定 → printのみ")
        print(text)
        return

    try:
        r = requests.post(WORKER_URL, json={"text": text}, timeout=10)
        print("[LINE RESULT]", r.status_code, r.text)
    except Exception as e:
        print("[ERROR] LINE送信失敗:", e)
        print(text)


# ============================================================
# Entry
# ============================================================
def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = calc_market_score()

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev, risk_info = analyze_positions(pos_df)

    core_report = build_core_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        total_asset=total_asset,
    )

    pos_report = build_position_report(today_str=today_str, pos_text=pos_text)

    # --- 文末 MAX建て玉 ---
    target_lev = mkt.get("leverage", 1.3)
    max_position = int(total_asset * target_lev)

    tail_lines = []
    tail_lines.append("◆ 本日の建て玉最大金額")
    tail_lines.append(f"- 推奨レバ: {target_lev:.1f}倍")
    tail_lines.append(f"- 運用資産ベース: 約{int(total_asset):,}円")
    tail_lines.append(f"- 今日のMAX建て玉: 約{max_position:,}円")

    tail_report = "\n".join(tail_lines)

    # print
    print(core_report)
    print("\n" + "=" * 40 + "\n")
    print(pos_report)
    print("\n" + "=" * 40 + "\n")
    print(tail_report)

    # LINE送信（3通）
    send_line(core_report)
    send_line(pos_report)
    send_line(tail_report)


if __name__ == "__main__":
    main()