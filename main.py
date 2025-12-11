from __future__ import annotations

import os
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.scoring import score_stock, calc_inout_for_stock
from utils.position import load_positions, analyze_positions

# ============================================================
# 基本設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算前後の除外日数
EARNINGS_EXCLUDE_DAYS = 3

# スクリーニング関連
SCORE_MIN_BASE = 70.0      # Aランク基準
RR_MIN_BASE = 1.8          # 最低RR
EV_R_MIN_BASE = 0.4        # 期待値R下限
MAX_FINAL_STOCKS_BASE = 3  # 地合い良ければ最大3銘柄


# ============================================================
# 決算フィルタ
# ============================================================
def filter_earnings(df: pd.DataFrame, today_date) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df

    df = df.copy()
    try:
        parsed = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    except Exception:
        return df

    df["earnings_date_parsed"] = parsed

    mask = []
    for d in df["earnings_date_parsed"]:
        if d is None or pd.isna(d):
            mask.append(True)
            continue
        try:
            delta = abs((d - today_date).days)
            mask.append(delta > EARNINGS_EXCLUDE_DAYS)
        except Exception:
            mask.append(True)

    return df[mask]


# ============================================================
# イベント関連
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
        date_str = str(row.get("date", "")).strip()
        time_str = str(row.get("time_jst", "")).strip()
        label = str(row.get("label", "")).strip()
        kind = str(row.get("kind", "")).strip()
        if not date_str or not label:
            continue
        events.append(
            {
                "date": date_str,
                "time_jst": time_str,
                "label": label,
                "kind": kind,
            }
        )
    return events


def build_event_warnings(today_date) -> List[str]:
    events = load_events()
    warns: List[str] = []
    for ev in events:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except Exception:
            continue

        delta = (d - today_date).days
        if delta > 2 or delta < -1:
            continue

        if delta > 1:
            when = f"{delta}日後"
        elif delta == 1:
            when = "明日"
        elif delta == 0:
            when = "本日"
        else:  # -1
            when = "昨日"

        time_part = f" {ev['time_jst']}" if ev.get("time_jst") else ""
        warns.append(f"⚠ {ev['label']}（{ev['date']}{time_part} JST / {when}）")

    if not warns:
        warns.append("- 特になし")

    return warns


# ============================================================
# 市場レバレッジ
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 2.0, "強気（押し目＋一部ブレイク）"
    if mkt_score >= 60:
        return 1.7, "やや強気（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "中立（押し目メイン）"
    if mkt_score >= 40:
        return 1.1, "やや守り（新規ロット小さめ）"
    return 1.0, "守り（新規かなり絞る）"


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# 期待値R
# ============================================================
def expected_r_from_in_rank(in_rank: str, rr: float) -> float:
    if rr <= 0:
        return -1.0

    if in_rank == "強IN":
        win = 0.45
    elif in_rank == "通常IN":
        win = 0.40
    elif in_rank == "弱めIN":
        win = 0.33
    else:
        win = 0.25

    lose = 1.0 - win
    ev_r = win * rr - lose * 1.0
    return float(ev_r)


# ============================================================
# 価格履歴取得
# ============================================================
def fetch_history(ticker: str, period: str = "130d") -> pd.DataFrame | None:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    return None


# ============================================================
# スクリーニング本体
# ============================================================
def run_screening(today_date, mkt_score: int) -> List[Dict]:
    try:
        df = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    # ticker カラム吸収
    if "ticker" in df.columns:
        t_col = "ticker"
    elif "code" in df.columns:
        t_col = "code"
    else:
        return []

    df = filter_earnings(df, today_date)

    # 地合い連動の基準
    min_score = SCORE_MIN_BASE
    rr_min = RR_MIN_BASE
    ev_min = EV_R_MIN_BASE

    if mkt_score >= 70:
        min_score -= 3
    elif mkt_score <= 45:
        min_score += 3

    # 地合いが弱いときはRR / EV閾値も少し上げる
    if mkt_score <= 45:
        rr_min += 0.2
        ev_min += 0.05

    candidates: List[Dict] = []

    for _, row in df.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        base_score = score_stock(hist)
        if base_score is None or base_score < min_score:
            continue

        in_rank, tp_pct, sl_pct = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue

        # 地合いが弱いとき「弱めIN」は除外
        if mkt_score <= 45 and in_rank == "弱めIN":
            continue

        # entry, tp/sl price
        close = hist["Close"].astype(float)
        entry = float(close.iloc[-1])

        tp_price = entry * (1.0 + tp_pct / 100.0)
        sl_price = entry * (1.0 + sl_pct / 100.0)

        rr = (tp_pct / 100.0) / abs(sl_pct / 100.0) if sl_pct < 0 else 0.0
        ev_r = expected_r_from_in_rank(in_rank, rr)

        if rr < rr_min or ev_r < ev_min:
            continue

        candidates.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                score=float(base_score),
                in_rank=in_rank,
                rr=float(rr),
                entry=float(entry),
                tp_pct=float(tp_pct),
                sl_pct=float(sl_pct),
                tp_price=float(tp_price),
                sl_price=float(sl_price),
                ev_r=float(ev_r),
            )
        )

    # スコア → EV_R → RR でソート
    candidates.sort(
        key=lambda x: (x["score"], x["ev_r"], x["rr"]),
        reverse=True,
    )

    # 地合いに応じて銘柄数調整
    max_n = MAX_FINAL_STOCKS_BASE
    if mkt_score < 45:
        max_n = 2
    if mkt_score < 40:
        max_n = 1

    return candidates[:max_n]


# ============================================================
# レポート構築
# ============================================================
def build_report(today_str: str, today_date, mkt: Dict,
                 pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    lev, lev_comment = recommend_leverage(mkt_score)
    max_pos = calc_max_position(total_asset, lev)

    sectors = top_sectors_5d()
    cand = run_screening(today_date, mkt_score)

    # 候補統計
    if cand:
        rr_vals = [c["rr"] for c in cand]
        avg_rr = float(np.mean(rr_vals))
        min_rr = float(min(rr_vals))
        max_rr = float(max(rr_vals))
        cand_header = f"  候補数:{len(cand)}銘柄 / 平均RR:{avg_rr:.2f}R (最小:{min_rr:.2f}R 最大:{max_rr:.2f}R)"
    else:
        cand_header = "  候補数:0銘柄"

    events = build_event_warnings(today_date)

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")
    lines.append("📈 セクター（5日）")
    if sectors:
        for i, (s_name, pct) in enumerate(sectors[:5]):
            if i == 0:
                lines.append(f"1. {s_name} ({pct:+.2f}%)")
            else:
                lines.append(f"{i+1}. {s_name} ({pct:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")
    lines.append("⚠ イベント")
    for ev in events:
        lines.append(ev)
    lines.append("")
    lines.append("🏆 Core候補（最大3銘柄）")
    if cand:
        for c in cand:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(
                f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R IN:{c['in_rank']} EV:{c['ev_r']:.2f}R"
            )
            lines.append(
                f"  IN:{c['entry']:.1f} "
                f"TP:+{c['tp_pct']:.1f}% ({c['tp_price']:.1f}) "
                f"SL:{c['sl_pct']:.1f}% ({c['sl_price']:.1f})"
            )
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")
    lines.append(cand_header)
    lines.append("")
    lines.append("📊 ポジション")
    lines.append(pos_text)

    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定。以下をprintのみ。")
        print(text)
        return

    chunk_size = 3800
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=15)
            print("[LINE RESULT]", r.status_code, str(r.text)[:200])
        except Exception as e:
            print("[ERROR] LINE送信に失敗:", e)
            print(ch)


# ============================================================
# main
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    # 地合い
    mkt = enhance_market_score()

    # ポジション
    pos_df = load_positions()
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=int(mkt.get("score", 50)))
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    # レポート
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