from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.rr import compute_rr
from utils.util import jst_today_str, jst_today_date

# ============================================================
# 基本設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算フィルタ: ±N日
EARNINGS_EXCLUDE_DAYS = 3

# 候補数
SCREENING_TOP_N = 30         # 内部でまず30銘柄まで評価
MAX_FINAL_STOCKS = 3         # 最終的にLINEに出すのは最大3銘柄

# RRフィルタ
MIN_RR = 1.8                 # これ未満は候補から除外


# ============================================================
# 日付 / イベント関連
# ============================================================
def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    """
    events.csv:
      date,label,kind
      2025-12-12,FOMC,macro
    """
    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    out: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        d = str(row.get("date", "")).strip()
        label = str(row.get("label", "")).strip()
        kind = str(row.get("kind", "")).strip()
        if not d or not label:
            continue
        out.append({"date": d, "label": label, "kind": kind})
    return out


def build_event_warnings(today: date) -> List[str]:
    """
    イベント2日前〜翌日までを警告表示
    """
    events = load_events()
    warns: List[str] = []

    for ev in events:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except Exception:
            continue

        delta = (d - today).days
        if -1 <= delta <= 2:
            if delta > 0:
                when = f"{delta}日後"
            elif delta == 0:
                when = "本日"
            else:
                when = "直近"
            warns.append(f"⚠ {ev['label']}（{when}）")

    return warns


# ============================================================
# Universe / 決算フィルタ
# ============================================================
def load_universe(path: str = UNIVERSE_PATH) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"[WARN] universe file not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read universe: {e}")
        return None

    # ticker列前提（ユーザーのuniverse_jpx.csv仕様）
    if "ticker" not in df.columns:
        print("[WARN] universe has no 'ticker' column")
        return None

    df["ticker"] = df["ticker"].astype(str)

    # 決算日パース（あれば）
    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


def in_earnings_window(row: pd.Series, today: date) -> bool:
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    try:
        delta = abs((d - today).days)
    except Exception:
        return False
    return delta <= EARNINGS_EXCLUDE_DAYS


# ============================================================
# レバ / 建て玉
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    """
    地合いスコア → レバ＆コメント
    """
    if mkt_score >= 70:
        return 1.8, "強め（押し目＋一部ブレイク）"
    if mkt_score >= 60:
        return 1.5, "やや強め（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "中立（押し目メイン）"
    if mkt_score >= 40:
        return 1.0, "弱め（新規は厳選）"
    return 0.7, "守り（新規ほぼ見送り）"


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# 最低スコアライン（地合い連動）
# ============================================================
def dynamic_min_score(mkt_score: int) -> float:
    """
    地合いが弱いほどフィルタを厳しく、強いほど少し緩く。
    """
    base = 60.0  # Bランクの下限イメージ
    if mkt_score >= 70:
        return base - 5.0   # 少し緩め
    if mkt_score >= 60:
        return base - 2.0
    if mkt_score >= 50:
        return base
    if mkt_score >= 40:
        return base + 5.0   # 少し厳しく
    return base + 8.0       # かなり厳しく


# ============================================================
# 株価履歴
# ============================================================
def fetch_history(ticker: str, period: str = "130d") -> pd.DataFrame | None:
    for attempt in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f"[WARN] fetch history failed {ticker} (try {attempt+1}): {e}")
    return None


# ============================================================
# スクリーニング本体
# ============================================================
def run_screening(today: date, mkt_score: int) -> List[Dict]:
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return []

    min_score = dynamic_min_score(mkt_score)

    candidates: List[Dict] = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        # 決算前後は除外
        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        # ベーススコア（形・流動性など）
        base_score = score_stock(hist)
        if not np.isfinite(base_score):
            continue

        # 最低スコアライン未達は除外
        if base_score < min_score:
            continue

        # RR＋INランク
        rr_info = compute_rr(hist, mkt_score)
        rr = float(rr_info["rr"])
        entry = float(rr_info["entry"])
        tp_pct = float(rr_info["tp_pct"])     # 0.156 → +15.6%
        sl_pct = float(rr_info["sl_pct"])     # 0.048 → -4.8% 表示時にマイナス付与
        in_rank = str(rr_info["in_rank"])

        # INランク「様子見」は除外（通知する価値なし）
        if in_rank == "様子見":
            continue

        # RRフィルタ
        if rr < MIN_RR:
            continue

        candidates.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "score": float(base_score),
                "rr": rr,
                "entry": entry,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "in_rank": in_rank,
            }
        )

    # スコア → RR の順でソート
    candidates.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)

    # 上位N銘柄に絞る
    top = candidates[:SCREENING_TOP_N]

    # 最終的にLINEに出すのは最大MAX_FINAL_STOCKS
    return top[:MAX_FINAL_STOCKS]


# ============================================================
# レポート構築
# ============================================================
def build_report(
    today_str: str,
    today_date: date,
    mkt: Dict,
    pos_text: str,
    total_asset: float,
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    rec_lev, lev_comment = recommend_leverage(mkt_score)
    est_asset = total_asset if np.isfinite(total_asset) and total_asset > 0 else 2_000_000.0
    max_pos = calc_max_position(est_asset, rec_lev)

    # セクター（上位5）
    secs = top_sectors_5d()
    sec_lines: List[str] = []
    for i, (name, chg) in enumerate(secs[:5]):
        sec_lines.append(f"{i+1}. {name} ({chg:+.2f}%)")

    # イベント
    ev_lines = build_event_warnings(today_date)
    if not ev_lines:
        ev_lines = ["- 特になし"]

    # スクリーニング
    core_list = run_screening(today_date, mkt_score)

    # RRサマリ
    if core_list:
        rr_vals = [c["rr"] for c in core_list]
        rr_avg = float(np.mean(rr_vals))
        rr_min = float(np.min(rr_vals))
        rr_max = float(np.max(rr_vals))
    else:
        rr_avg = rr_min = rr_max = 0.0

    lines: List[str] = []

    # ヘッダー
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- レバ: {rec_lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    # セクター
    lines.append("📈 セクター（5日）")
    lines.extend(sec_lines or ["データ不足"])
    lines.append("")

    # イベント
    lines.append("⚠ イベント")
    lines.extend(ev_lines)
    lines.append("")

    # Core候補
    lines.append(f"🏆 Core候補（最大{MAX_FINAL_STOCKS}銘柄）")
    if core_list:
        for c in core_list:
            lines.append(
                f"- {c['ticker']} {c['name']} [{c['sector']}]"
            )
            lines.append(
                f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R IN:{c['in_rank']}"
            )
            lines.append(
                f"  IN:{c['entry']:.1f} TP:+{c['tp_pct']*100:.1f}% SL:-{c['sl_pct']*100:.1f}%"
            )
            lines.append("")
        lines.append(
            f"  候補数:{len(core_list)}銘柄 / 平均RR:{rr_avg:.2f}R "
            f"(最小:{rr_min:.2f}R 最大:{rr_max:.2f}R)"
        )
    else:
        lines.append("- 該当なし")
    lines.append("")

    # ポジション
    lines.append("📊 ポジション")
    lines.append(pos_text.strip() or "ノーポジション")

    return "\n".join(lines)


# ============================================================
# LINE送信（分割対応）
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定。コンソール出力のみ。")
        print(text)
        return

    chunk_size = 3900
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=15)
            print("[LINE RESULT]", r.status_code, r.text[:200])
        except Exception as e:
            print("[ERROR] LINE送信失敗:", e)
            print(ch)


# ============================================================
# Entry
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    # 地合い（SOX/NVDA補正込み）
    mkt = enhance_market_score()

    # ポジション / 推定資産
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=int(mkt.get("score", 50)))

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