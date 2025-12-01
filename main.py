from __future__ import annotations

import os
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str
from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.scoring import score_stock
from utils.position import load_positions, analyze_positions


# ============================================================
# 基本設定
# ============================================================

UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
WORKER_URL = os.getenv("WORKER_URL")


# ============================================================
# yfinance ラッパ
# ============================================================

def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    """株価ヒストリを安全に取得"""
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def calc_20d_vola_from_hist(df: pd.DataFrame) -> Optional[float]:
    """ヒストリから20日ボラティリティを算出"""
    try:
        close = df["Close"].astype(float)
        ret = close.pct_change(fill_method=None)
        vola = float(ret.rolling(20).std().iloc[-1])
        if not np.isfinite(vola) or vola <= 0:
            return None
        return vola
    except Exception:
        return None


# ============================================================
# 候補銘柄の IN / 利確 / 損切り目安
# ============================================================

def calc_entry_and_risk_for_candidate(
    score: float,
    market_score: int,
    price: float,
    vola20: Optional[float],
) -> Dict[str, str]:
    """
    Aランク / Bランク候補用
    vola20 が None の場合はデフォルト値で代用
    """
    # デフォルトボラ（そこそこ動く銘柄）
    base_vola = 0.025
    if vola20 is None or not np.isfinite(vola20) or vola20 <= 0:
        vola = base_vola
    else:
        # 変な値はクリップ
        vola = float(np.clip(vola20, 0.01, 0.08))

    # 地合いで IN 条件
    if score >= 80:
        # Aランク
        cls = "A"
        if market_score >= 60:
            entry = "寄り〜前日終値付近でIN可"
        elif market_score >= 40:
            entry = "日中の押し目（前日安値〜5日線付近）でIN"
        else:
            entry = "地合い弱いので本日IN見送り推奨"
        # 利確・損切（ボラに応じて 4〜6%, 2.5〜3.5%）
        tp_pct = float(np.clip(0.03 + 1.6 * vola, 0.04, 0.06))
        sl_pct = float(np.clip(0.015 + 1.0 * vola, 0.025, 0.035))
    else:
        # Bランク
        cls = "B"
        if market_score >= 65:
            entry = "寄りIN可（ロット控えめ推奨）"
        elif market_score >= 45:
            entry = "押し目限定IN（前日安値〜5日線付近）"
        else:
            entry = "地合い弱め → 新規INは慎重に/見送り推奨"
        # 利確・損切（2.5〜4%, 2〜2.5%）
        tp_pct = float(np.clip(0.02 + 1.2 * vola, 0.025, 0.04))
        sl_pct = float(np.clip(0.015 + 0.6 * vola, 0.02, 0.025))

    tp_price = price * (1.0 + tp_pct)
    sl_price = price * (1.0 - sl_pct)

    line = (
        f"IN目安: {entry} / "
        f"利確:+{tp_pct*100:.1f}%（{tp_price:.0f}円） / "
        f"損切:-{sl_pct*100:.1f}%（{sl_price:.0f}円）"
    )

    return {
        "class": cls,
        "entry": entry,
        "tp_pct": f"{tp_pct*100:.1f}",
        "sl_pct": f"{sl_pct*100:.1f}",
        "tp_price": f"{tp_price:.0f}",
        "sl_price": f"{sl_price:.0f}",
        "line": line,
    }


# ============================================================
# スクリーニング本体
# ============================================================

def run_screening(market_score: int) -> Tuple[List[Dict], List[Dict]]:
    """
    universe_jpx.csv 全銘柄をスコアリングして
    Aランク / Bランク候補リストを返す
    """
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], []

    if "ticker" not in uni.columns:
        return [], []

    A_list: List[Dict] = []
    B_list: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row["ticker"])
        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        score = score_stock(hist)
        if score is None:
            continue

        price = float(hist["Close"].iloc[-1])
        vola20 = calc_20d_vola_from_hist(hist)
        risk = calc_entry_and_risk_for_candidate(score, market_score, price, vola20)

        info = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": float(score),
            "price": price,
            "risk": risk,
        }

        if score >= 80:
            A_list.append(info)
        elif score >= 70:
            B_list.append(info)

    A_list = sorted(A_list, key=lambda x: x["score"], reverse=True)
    B_list = sorted(B_list, key=lambda x: x["score"], reverse=True)
    return A_list, B_list


# ============================================================
# レポート生成
# ============================================================

def build_report() -> str:
    today = jst_today_str()

    # ---- 地合い ----
    mkt = calc_market_score()
    mkt_score: int = int(mkt["score"])
    mkt_comment: str = mkt["comment"]

    # ---- セクターTOP3 ----
    secs = top_sectors_5d()
    if secs:
        sector_lines = [
            f"{i+1}. {name} ({chg:+.2f}%)"
            for i, (name, chg) in enumerate(secs[:3])
        ]
        sector_text = "\n".join(sector_lines)
    else:
        sector_text = "算出不可（データ不足）"

    # ---- スクリーニング ----
    A_list, B_list = run_screening(mkt_score)

    # ---- ポジション ----
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev, risk_lines = analyze_positions(pos_df)

    # ---- レポート組み立て ----
    lines: List[str] = []

    lines.append(f"📅 {today} stockbotTOM 日報")
    lines.append("")
    # 結論
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append("")
    # セクター
    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sector_text)
    lines.append("")

    # Core A
    lines.append("◆ Core候補 Aランク（本命押し目）")
    if not A_list:
        lines.append("本命Aランクなし。")
    else:
        for r in A_list:
            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{r['price']:.1f}"
            )
            lines.append(f"  {r['risk']['line']}")
    lines.append("")

    # Core B
    lines.append("◆ Core候補 Bランク（押し目候補）")
    if not B_list:
        lines.append("Bランク候補なし。")
    else:
        for r in B_list:
            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{r['price']:.1f}"
            )
            lines.append(f"  {r['risk']['line']}")
    lines.append("")

    # ポジション
    lines.append("◆ ポジション分析")
    lines.append(pos_text)
    lines.append("")

    # ポジションの利確/損切り目安（銘柄ごと）
    if risk_lines:
        lines.append("◆ 保有ポジションの利確 / 損切り目安")
        lines.extend(risk_lines)
        lines.append("")

    return "\n".join(lines)


# ============================================================
# LINE 送信
# ============================================================

def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定。以下メッセージのみ出力:")
        print(text)
        return

    try:
        r = requests.post(WORKER_URL, json={"text": text}, timeout=15)
        print("[LINE RESULT]", r.status_code, r.text)
    except Exception as e:
        print("[ERROR] LINE送信失敗:", e)
        print(text)


# ============================================================
# main
# ============================================================

def main() -> None:
    text = build_report()
    print(text)
    send_line(text)


if __name__ == "__main__":
    main()