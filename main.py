from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import pandas as pd
import yfinance as yf

from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.rr import compute_rr

# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"  # いまは未使用（将来のイベント拡張用）

# 決算フィルタ: ±N日
EARNINGS_EXCLUDE_DAYS = 3

# 候補数（地合いで 1〜3 に可変）
MAX_CORE_CANDIDATES = 3


# ============================================================
# 日付ユーティリティ（JST）
# ============================================================
def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date() -> datetime.date:
    return jst_now().date()


# ============================================================
# Earnings 付近の銘柄を除外
# ============================================================
def filter_earnings(df: pd.DataFrame, today: datetime.date) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df

    try:
        ed = pd.to_datetime(df["earnings_date"], errors="coerce")
    except Exception:
        return df

    today_ts = pd.Timestamp(today)
    delta_days = (ed - today_ts).dt.days.abs()

    # earnings_date NaN → フィルタしない
    mask = (delta_days.isna()) | (delta_days > EARNINGS_EXCLUDE_DAYS)
    return df[mask]


# ============================================================
# スクリーニング条件
# ============================================================
def determine_score_threshold(mkt_score: int) -> float:
    """
    地合いに応じて必要スコアを可変
    """
    base = 60.0
    if mkt_score <= 45:
        base += 5.0       # 弱いときはより厳しく
    elif mkt_score >= 60:
        base -= 5.0       # 強いときは少し緩め
    return base


def determine_max_candidates(mkt_score: int) -> int:
    """
    地合いに応じて候補数を 1〜3 に調整
    """
    if mkt_score <= 40:
        return 1
    elif mkt_score <= 50:
        return 2
    else:
        return MAX_CORE_CANDIDATES


# ============================================================
# スクリーニング本体
# ============================================================
def run_screening(today: datetime.date, mkt_score: int) -> List[Dict[str, Any]]:
    # universe 読み込み
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    # 決算前後を除外
    uni = filter_earnings(uni, today)

    score_min = determine_score_threshold(mkt_score)
    max_candidates = determine_max_candidates(mkt_score)

    results: List[Dict[str, Any]] = []

    for _, row in uni.iterrows():
        ticker = str(row.get("ticker") or row.get("code") or "").strip()
        if not ticker:
            continue

        # 株価ヒストリ取得
        try:
            hist = yf.download(
                ticker,
                period="90d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception:
            continue

        if hist is None or len(hist) < 60:
            continue

        # スコアリング
        try:
            score = score_stock(ticker, hist)
        except Exception:
            continue

        if score < score_min:
            continue

        # RR 計算
        try:
            rr_info = compute_rr(hist, mkt_score)
        except Exception:
            continue

        rr = float(rr_info.get("rr", 0.0))
        if rr < 2.0:
            continue

        results.append(
            dict(
                ticker=ticker,
                sector=str(row.get("sector", "")),
                score=float(score),
                rr=rr,
                entry=float(rr_info.get("entry", 0.0)),
                tp_pct=float(rr_info.get("tp_pct", 0.0)),
                sl_pct=float(rr_info.get("sl_pct", 0.0)),
                in_label=str(rr_info.get("in_label", "")),
            )
        )

    # スコア→RR の優先順でソート
    results.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)

    return results[:max_candidates]


# ============================================================
# レポート生成
# ============================================================
def build_report(
    today_str: str,
    today_date: datetime.date,
    mkt_dict: Dict[str, Any],
    pos_text: str,
    total_asset: float,
) -> str:
    mkt_score = int(mkt_dict.get("score", 50))
    mkt_comment = str(mkt_dict.get("comment", "中立"))

    # レバ推奨（地合いに応じて）
    if mkt_score <= 40:
        lever = 1.0
    elif mkt_score <= 55:
        lever = 1.3
    elif mkt_score <= 70:
        lever = 1.6
    else:
        lever = 2.0

    core_list = run_screening(today_date, mkt_score)
    sectors = top_sectors_5d()

    lines: List[str] = []

    # 見出し
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")

    # 今日の結論
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- レバ: {lever:.1f}倍（{mkt_comment}）")
    max_notional = int(total_asset * lever)
    lines.append(f"- MAX建玉: 約{max_notional:,}円\n")

    # セクター
    if sectors:
        lines.append("📈 セクター（5日）")
        for i, sec in enumerate(sectors[:5], start=1):
            lines.append(f"{i}. {sec['sector']} ({sec['chg']:+.2f}%)")
        lines.append("")

    # イベント（今はプレースホルダ）
    lines.append("⚠ イベント")
    lines.append("- 特になし\n")

    # Core候補
    lines.append("🏆 Core候補（最大3銘柄）")
    if core_list:
        for c in core_list:
            lines.append(f"- {c['ticker']} [{c['sector']}]")
            in_label = c.get("in_label") or ""
            if in_label:
                lines.append(
                    f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R IN:{in_label}"
                )
            else:
                lines.append(
                    f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R"
                )
            lines.append(
                f"  IN:{c['entry']:.1f} TP:{c['tp_pct']*100:+.1f}% SL:{c['sl_pct']*100:.1f}%"
            )

        rr_values = [c["rr"] for c in core_list]
        avg_rr = sum(rr_values) / len(rr_values)
        lines.append(
            f"\n  候補数:{len(core_list)}銘柄 / 平均RR:{avg_rr:.2f}R "
            f"(最小:{min(rr_values):.2f}R 最大:{max(rr_values):.2f}R)\n"
        )
    else:
        lines.append("- 該当なし\n")

    # ポジション
    lines.append("📊 ポジション")
    lines.append(pos_text)

    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(msg: str) -> None:
    worker_url = os.getenv("WORKER_URL")
    if not worker_url:
        # デバッグ用
        print(msg)
        return

    import requests

    try:
        requests.post(worker_url, json={"message": msg}, timeout=10)
    except Exception:
        # 失敗してもログだけ残す
        print(msg)


# ============================================================
# Main
# ============================================================
def main() -> None:
    today_date = jst_today_date()
    today_str = jst_today_str()

    # 地合い
    mkt_dict = enhance_market_score()

    # ポジション
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df)

    # レポート & 送信
    report = build_report(today_str, today_date, mkt_dict, pos_text, total_asset)
    send_line(report)


if __name__ == "__main__":
    main()