from __future__ import annotations

import os
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from util import jst_today_str, jst_today_date
from market import enhance_market_score, market_delta_3d
from sector import top_sectors_5d
from scoring import screen_swing
from position import load_positions, analyze_positions

# =========================
# 設定（届く構成は維持）
# =========================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
WORKER_URL = os.getenv("WORKER_URL")

MAX_FINAL = 5
SECTOR_TOP_N = 5

# =========================
def build_report(today_str, today_date, mkt, pos_text, total_asset) -> str:
    mkt_score = mkt["score"]
    delta3 = market_delta_3d()

    trade_ok = not (
        mkt_score < 45 or
        (delta3 <= -5 and mkt_score < 55)
    )

    lev = 1.7 if mkt_score >= 50 else 1.3
    max_pos = int(total_asset * lev)

    sectors = top_sectors_5d(SECTOR_TOP_N)
    swing = screen_swing(today_date, mkt_score)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論（Swing専用）")
    lines.append("✅ 新規可（条件クリア）" if trade_ok else "🚫 本日は新規見送り")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt['comment']})")
    lines.append(f"- ΔMarketScore_3d: {delta3:+d}")
    lines.append(f"- レバ: {lev:.1f}倍")
    lines.append(f"- MAX建玉: 約{max_pos:,}円\n")

    lines.append("📈 セクター（5日）")
    for i, (s, p) in enumerate(sectors, 1):
        lines.append(f"{i}. {s} ({p:+.2f}%)")
    lines.append("")

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    if swing:
        rr_avg = np.mean([c["rr"] for c in swing])
        ev_avg = np.mean([c["ev"] for c in swing])
        rday_avg = np.mean([c["r_per_day"] for c in swing])
        lines.append(
            f"  候補数:{len(swing)}銘柄 / 平均RR:{rr_avg:.2f} / 平均EV:{ev_avg:.2f} / 平均R/day:{rday_avg:.2f}\n"
        )

        for i, c in enumerate(swing[:MAX_FINAL], 1):
            star = " ⭐" if i <= 2 else ""
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}] {star}")
            lines.append(
                f"  形:{c['setup']}  RR:{c['rr']:.2f}  AdjEV:{c['ev']:.2f}  R/day:{c['r_per_day']:.2f}"
            )
            lines.append(
                f"  IN:{c['entry']:.1f}（帯:{c['in_low']:.1f}〜{c['in_high']:.1f}） "
                f"現在:{c['price_now']:.1f}  ATR:{c['atr']:.1f}  GU:{'Y' if c['gu'] else 'N'}"
            )
            lines.append(
                f"  STOP:{c['stop']:.1f}  TP1:{c['tp1']:.1f}  TP2:{c['tp2']:.1f} "
                f"ExpectedDays:{c['exp_days']:.1f}  行動:{c['action']}\n"
            )
    else:
        lines.append("- 該当なし\n")

    lines.append("📊 ポジション")
    lines.append(pos_text if pos_text else "ノーポジション")

    return "\n".join(lines)


# =========================
def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return

    for ch in [text[i:i+3800] for i in range(0, len(text), 3800)]:
        try:
            requests.post(WORKER_URL, json={"text": ch}, timeout=20)
            time.sleep(0.3)
        except Exception:
            pass


def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df)

    report = build_report(today_str, today_date, mkt, pos_text, total_asset)
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()