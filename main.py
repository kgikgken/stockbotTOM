from __future__ import annotations

import os
import time
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date
from utils.market import enhance_market_score
from utils.scoring import score_stock, calc_inout_for_stock
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions


# ============================================================
# 設定（Swing集中・AL3 Top5 固定）
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
WORKER_URL = os.getenv("WORKER_URL")

AL3_MIN_SCORE = 75.0          # AL3最低スコア（強化済み）
AL3_MIN_RR = 2.5              # 最低RR
AL3_MIN_EV = 0.5              # 最低EV
AL3_MAX_DISPLAY = 5           # Top5固定

DEFAULT_ASSET = 2_000_000
MAX_LEVERAGE = 2.0


# ============================================================
# 共通
# ============================================================
def fetch_history(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return None


def expected_ev(in_rank: str, rr: float) -> float:
    if rr <= 0:
        return -999.0
    win = 0.45 if in_rank == "強IN" else 0.40 if in_rank == "通常IN" else 0.30
    return win * rr - (1 - win)


# ============================================================
# AL3スコア（最重要）
# ============================================================
def calc_al3_score(c: Dict) -> float:
    """
    勝ちに直結する要素のみ
    """
    score = 0.0
    score += c["rr"] * 0.35
    score += c["ev"] * 0.45
    score += (c["score"] / 100.0) * 0.20
    return score


# ============================================================
# Swing候補生成
# ============================================================
def build_swing_candidates(today_date) -> List[Dict]:
    uni = pd.read_csv(UNIVERSE_PATH)
    t_col = "ticker" if "ticker" in uni.columns else "code"

    out: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 120:
            continue

        score = score_stock(hist)
        if score is None or score < AL3_MIN_SCORE:
            continue

        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=50)
        rr = rr_info["rr"]
        if rr < AL3_MIN_RR:
            continue

        ev = expected_ev(in_rank, rr)
        if ev < AL3_MIN_EV:
            continue

        out.append(
            dict(
                ticker=ticker,
                name=row.get("name", ticker),
                score=score,
                in_rank=in_rank,
                rr=rr,
                ev=ev,
                entry=rr_info["entry"],
                tp_pct=rr_info["tp_pct"],
                sl_pct=rr_info["sl_pct"],
            )
        )

    for c in out:
        c["al3"] = calc_al3_score(c)

    out.sort(key=lambda x: x["al3"], reverse=True)
    return out[:AL3_MAX_DISPLAY]


# ============================================================
# 乗り換え判定（重要）
# ============================================================
def judge_switch(current_rr: float, current_ev: float, new: Dict) -> bool:
    """
    RR +0.8R もしくは EV +0.4R 以上でのみ乗り換え
    """
    if new["rr"] >= current_rr + 0.8:
        return True
    if new["ev"] >= current_ev + 0.4:
        return True
    return False


# ============================================================
# レポート
# ============================================================
def build_report(
    today_str: str,
    mkt: Dict,
    swings: List[Dict],
    pos_text: str,
    total_asset: float,
) -> str:

    max_pos = int(total_asset * MAX_LEVERAGE)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing集中 / AL3 Top5）")
    lines.append(f"- 地合い: {mkt['score']}点 ({mkt['comment']})")
    lines.append(f"- 推奨レバ: {MAX_LEVERAGE:.1f}倍（AL3のみ）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    lines.append("🏆 Swing（AL3 Top5）")
    if not swings:
        lines.append("- 該当なし（今日は“やらない”日）")
    else:
        for i, c in enumerate(swings, 1):
            star = " ⭐" if i == 1 else ""
            lines.append(
                f"{i}. {c['ticker']} {c['name']}{star}\n"
                f"   AL3:{c['al3']:.2f}  Score:{c['score']:.1f}  IN:{c['in_rank']}\n"
                f"   RR:{c['rr']:.2f}R  EV:{c['ev']:.2f}R\n"
                f"   押し目IN:{c['entry']:.1f}  TP:{c['tp_pct']*100:.1f}%  SL:{c['sl_pct']*100:.1f}%"
            )

    lines.append("")
    lines.append("📊 ポジション")
    lines.append(pos_text if pos_text else "ノーポジション")

    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return
    requests.post(WORKER_URL, json={"text": text}, timeout=20)


# ============================================================
# Main
# ============================================================
def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=mkt["score"])
    if not np.isfinite(total_asset) or total_asset <= 0:
        total_asset = DEFAULT_ASSET

    swings = build_swing_candidates(today_date)

    report = build_report(
        today_str=today_str,
        mkt=mkt,
        swings=swings,
        pos_text=pos_text,
        total_asset=total_asset,
    )

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()