from __future__ import annotations

"""
日本株スイングトレード用 朝イチスクリーニング & 戦略通知ボット（完全版）
機能:
- universe_jpx.csv を読み込み（全銘柄ユニバース想定）
- yfinance で日足 OHLCV を取得
- テクニカル評価を算出
- Core スコア（0〜100）を算出
- 地合い（0〜100）
- セクターTOP3（5日騰落率）
- positions.csv から保有ポジション分析
- data/equity.json による推定運用資産（LINE通知で更新前提）
- Cloudflare Worker 経由で LINE にテキスト送信
"""

import os
import math
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone

from utils import (
    safe_price,
    calc_market_score,
    calc_sector_strength,
    calc_core_score,
    load_positions,
    load_equity,
    estimate_total_equity,
)


JST = timezone(timedelta(hours=9))


# ============================================================
# Core 候補抽出
# ============================================================

def load_universe(csv_path: str = "universe_jpx.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["ticker"])
        df["ticker"] = df["ticker"].astype(str)
        return df
    except Exception:
        return pd.DataFrame()


def screen_core_candidates(universe: pd.DataFrame) -> list[dict]:
    """Core Aランクのみ抽出（score ≥ 75）"""

    results = []

    tickers = universe["ticker"].astype(str).tolist()
    if len(tickers) == 0:
        return []

    # 一括取得（高速化）
    try:
        hist = yf.download(
            tickers=tickers,
            period="60d",
            interval="1d",
            group_by="ticker",
            progress=False,
        )
    except Exception:
        hist = None

    for t in tickers:
        try:
            h = hist[t] if hist is not None else yf.Ticker(t).history(period="60d")
        except Exception:
            continue

        if h is None or len(h) < 40:
            continue

        score = calc_core_score(h)
        if score >= 75:
            results.append(
                {
                    "ticker": t,
                    "score": score,
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ============================================================
# メッセージ生成
# ============================================================

def build_message(
    market_info: dict,
    sectors: list,
    core_list: list,
    positions: list,
    total_equity: float,
    total_position_value: float,
):
    lines = []

    today = datetime.now(JST).strftime("%Y-%m-%d")
    lines.append(f"📅 {today} stockbotTOM 日報\n")

    # --------------------------------------------------------
    # 地合い
    # --------------------------------------------------------
    ms = market_info
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {ms['score']}点")
    lines.append(f"- コメント: {ms['comment']}\n")

    # --------------------------------------------------------
    # セクターTOP3
    # --------------------------------------------------------
    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    if len(sectors) == 0:
        lines.append("算出できるセクターデータがありません。\n")
    else:
        for i, s in enumerate(sectors, 1):
            lines.append(f"{i}位: {s['sector']}（{s['perf']:.2f}%）")
        lines.append("")

    # --------------------------------------------------------
    # Core候補
    # --------------------------------------------------------
    lines.append("◆ Core候補 Aランク（本命押し目）")
    if len(core_list) == 0:
        lines.append("本命Aランク条件なし。\n")
    else:
        for c in core_list:
            lines.append(f"- {c['ticker']}（Score {c['score']}）")
        lines.append("")

    # --------------------------------------------------------
    # ポジション分析
    # --------------------------------------------------------
    lines.append("◆ ポジション分析")
    lines.append(f"推定運用資産: {total_equity:,.0f}円")
    lines.append(f"推定ポジション総額: {total_position_value:,.0f}円（レバ約 {total_position_value/total_equity:.2f}倍）")

    if len(positions) == 0:
        lines.append("保有ポジションなし。\n")
    else:
        for p in positions:
            if p["current"] is None:
                lines.append(f"- {p['ticker']}: データ取得失敗")
            else:
                pct = (p["current"] - p["price"]) / p["price"] * 100
                lines.append(
                    f"- {p['ticker']}: 現値 {p['current']} / 取得 {p['price']} / 損益 {pct:.2f}%"
                )
        lines.append("")

    return "\n".join(lines)


# ============================================================
# メイン処理
# ============================================================

def main():
    universe = load_universe()
    positions = load_positions()
    equity_json = load_equity()

    # --------------------------------------------------------
    # 地合い
    # --------------------------------------------------------
    market_info = calc_market_score()

    # --------------------------------------------------------
    # セクタートップ
    # --------------------------------------------------------
    sectors = calc_sector_strength()

    # --------------------------------------------------------
    # Core候補 Aランク
    # --------------------------------------------------------
    core_list = screen_core_candidates(universe)

    # --------------------------------------------------------
    # ポジション評価
    # --------------------------------------------------------
    total_equity = estimate_total_equity(equity_json, positions)
    total_position_value = 0

    for p in positions:
        cur = safe_price(p["ticker"])
        p["current"] = cur
        if cur is not None:
            total_position_value += cur * p["qty"]

    # --------------------------------------------------------
    # LINE メッセージ作成
    # --------------------------------------------------------
    msg = build_message(
        market_info,
        sectors,
        core_list,
        positions,
        total_equity,
        total_position_value,
    )

    print(msg)

    # --------------------------------------------------------
    # Cloudflare Worker へ送信
    # --------------------------------------------------------
    url = os.getenv("WORKER_URL", "")
    if url:
        try:
            import requests
            requests.post(url, json={"text": msg}, timeout=10)
        except Exception as e:
            print("[WARN] LINE送信エラー:", e)


if __name__ == "__main__":
    main()