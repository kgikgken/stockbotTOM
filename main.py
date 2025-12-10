from __future__ import annotations

import os
from datetime import datetime, timedelta
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
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算前後の除外
EARNINGS_EXCLUDE_DAYS = 3

# スクリーニング関連
MAX_FINAL_STOCKS_BASE = 3      # 地合い次第で 1〜3 に変動
SCORE_MIN_BASE = 70.0          # Aランク基準
RR_MIN_BASE = 1.8
EV_R_MIN_BASE = 0.4            # 期待値 (R)の下限

# ------------------------------------------------------------


# ============================================================
# イベント系
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
        label = str(row.get("label", "")).strip()
        kind = str(row.get("kind", "")).strip()
        if not date_str or not label:
            continue
        events.append({"date": date_str, "label": label, "kind": kind})
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
        if -1 <= delta <= 2:
            if delta > 0:
                when = f"{delta}日後"
            elif delta == 0:
                when = "本日"
            else:
                when = "直近"
            warns.append(f"⚠ {ev['label']}（{when}）")
    if not warns:
        warns.append("- 特になし")
    return warns


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
# ATR / Entry price
# ============================================================

def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) <= period + 1:
        return 0.0

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]

    if atr is None or not np.isfinite(atr):
        return 0.0
    return float(atr)


def compute_entry_price(close: pd.Series, ma5: float, ma20: float, atr: float) -> float:
    """
    “今日から3〜10日スイングで勝ちやすい” IN価格
    - ベースは MA20 付近
    - ATR の 0.5倍だけ下方向にずらす（押し目をしっかり待つ）
    - 直近安値を割りすぎないように補正
    - 強トレンド時は少しだけ浅めに
    """
    price = float(close.iloc[-1])
    last_low = float(close.iloc[-5:].min())

    target = ma20

    if atr and atr > 0:
        target = target - atr * 0.5

    if price > ma5 > ma20:
        target = ma20 + (ma5 - ma20) * 0.3

    if target > price:
        target = price * 0.995

    if target < last_low:
        target = last_low * 1.02

    return round(float(target), 1)


# ============================================================
# レバレッジ
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
# RR / EV 計算
# ============================================================

def expected_r_from_in_rank(in_rank: str, rr: float) -> float:
    """
    in_rankごとのざっくり勝率を想定して EV_R を計算
    """
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
# スクリーニング本体
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


def run_screening(today_date, mkt_score: int) -> List[Dict]:
    try:
        df = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    # tickerカラム名の吸収
    if "ticker" in df.columns:
        t_col = "ticker"
    elif "code" in df.columns:
        t_col = "code"
    else:
        return []

    df = filter_earnings(df, today_date)

    MIN_SCORE = SCORE_MIN_BASE
    RR_MIN = RR_MIN_BASE
    EV_MIN = EV_R_MIN_BASE

    # 地合いで少しだけ基準可変
    if mkt_score >= 70:
        MIN_SCORE -= 3
    elif mkt_score <= 45:
        MIN_SCORE += 3

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
        if base_score is None or base_score < MIN_SCORE:
            continue

        in_rank, tp_pct, sl_pct = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue

        # 弱い地合いで「弱めIN」は切る
        if mkt_score <= 45 and in_rank == "弱めIN":
            continue

        close = hist["Close"].astype(float)
        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        atr = calc_atr(hist)

        entry = compute_entry_price(close, ma5, ma20, atr)

        tp_price = entry * (1.0 + tp_pct / 100.0)
        sl_price = entry * (1.0 + sl_pct / 100.0)

        rr = (tp_pct / 100.0) / abs(sl_pct / 100.0) if sl_pct < 0 else 0.0
        ev_r = expected_r_from_in_rank(in_rank, rr)

        if rr < RR_MIN or ev_r < EV_MIN:
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

    # ソート：Score → EV_R → RR
    candidates.sort(
        key=lambda x: (x["score"], x["ev_r"], x["rr"]),
        reverse=True,
    )

    # 地合いで本数を調整
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
            lines.append(
                f"- {c['ticker']} {c['name']} [{c['sector']}]"
            )
            lines.append(
                f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R IN:{c['in_rank']}"
            )
            lines.append(
                f"  IN:{c['entry']:.1f} TP:+{c['tp_pct']:.1f}% SL:{c['sl_pct']:.1f}%"
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
            print("[LINE RESULT]", r.status_code, r.text[:200])
        except Exception as e:
            print("[ERROR] LINE送信に失敗:", e)
            print(ch)


# ============================================================
# Main
# ============================================================

def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    # 地合い（SOX/NVDA込み）
    mkt = enhance_market_score()

    # ポジション
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=int(mkt.get("score", 50)))
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    # レポート構築
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