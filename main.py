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

# Aランク最大数（あなたの指定）
MAX_A = 5
MAX_TOTAL = 5

# 決算フィルタ
EARNINGS_EXCLUDE_DAYS = 3


# ============================================================
# 日付ユーティリティ
# ============================================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


# ============================================================
# イベント（あとで強化可能）
# ============================================================
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
            when = (
                f"{delta}日後" if delta > 0 else ("本日" if delta == 0 else "直近")
            )
            warns.append(f"⚠ {ev['label']}（{when}）: ポジションサイズ注意")

    return warns


# ============================================================
# Universe 読み込み
# ============================================================
def load_universe(path: str = UNIVERSE_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print("[WARN] universe file missing")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("[WARN] cannot read universe:", e)
        return None

    if "ticker" not in df.columns:
        print("[WARN] universe has no ticker col")
        return None

    df["ticker"] = df["ticker"].astype(str)

    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


def in_earnings_window(row, today):
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    try:
        if abs((d - today).days) <= EARNINGS_EXCLUDE_DAYS:
            return True
    except:
        pass
    return False


# ============================================================
# データ取得
# ============================================================
def fetch_history(ticker: str, period: str = "130d"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# ============================================================
# TP / SL 設計
# ============================================================
def calc_candidate_tp_sl(price, vola20, mkt_score):
    if price <= 0:
        return 0, 0, price, price

    v = float(vola20) if np.isfinite(vola20) else 0.04

    if v < 0.02:
        tp, sl = 0.08, -0.03
    elif v > 0.06:
        tp, sl = 0.12, -0.06
    else:
        tp, sl = 0.10, -0.04

    # 市場の地合いで微調整
    if mkt_score >= 70:
        tp += 0.02
    elif mkt_score < 45:
        tp -= 0.02
        sl = max(sl, -0.03)

    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))
    return tp, sl, price * (1 + tp), price * (1 + sl)


# ============================================================
# スクリーニング本体
# ============================================================
def run_screening(today, mkt_score, total_asset):
    df = load_universe()
    if df is None:
        return [], []

    A, B = [], []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        # 決算直近は除外
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
        vola20 = float(close.pct_change().rolling(20).std().iloc[-1])

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

        # --- スコア帯で A / B 判定 ---
        if sc >= 85:
            A.append(info)
        elif sc >= 75:
            B.append(info)

    # スコア順
    A.sort(key=lambda x: x["score"], reverse=True)
    B.sort(key=lambda x: x["score"], reverse=True)

    return A, B(
    # ============================================================
# A/B 最終選定（最大5銘柄）
# ============================================================
def select_primary(A_list, B_list, max_total=5):
    """
    ・Aを優先して max_total まで埋める
    ・Aがmax_total以上 → A上位max_totalのみ
    ・Aが少ない場合 → Bから補充
    """
    A_sel = A_list[:max_total]
    remain = max_total - len(A_sel)

    if remain > 0:
        B_sel = B_list[:remain]
    else:
        B_sel = []

    return A_sel, B_sel


# ============================================================
# レポート生成：Core候補
# ============================================================
def build_core_report(today_str, today_date, mkt, total_asset):
    mkt_score = int(mkt["score"])
    mkt_comment = mkt["comment"]

    # 推奨レバ（あなた仕様）
    if mkt_score >= 70:
        target_lev = 1.8
        lev_label = "強め（押し目＋一部ブレイク可）"
    elif mkt_score >= 60:
        target_lev = 1.5
        lev_label = "やや強め（押し目メイン）"
    elif mkt_score >= 50:
        target_lev = 1.3
        lev_label = "標準（押し目のみ）"
    else:
        target_lev = 1.1
        lev_label = "守り寄り（ロット下げ）"

    # トップセクター
    secs = top_sectors_5d()
    if secs:
        sector_text = "\n".join([
            f"{i+1}. {s[0]} ({s[1]:+.2f}%)"
            for i, s in enumerate(secs)
        ])
    else:
        sector_text = "算出不可（データ不足）"

    # スクリーニング
    A_list, B_list = run_screening(
        today=today_date,
        mkt_score=mkt_score,
        total_asset=total_asset,
    )

    # A優先で5つ
    A_sel, B_sel = select_primary(A_list, B_list, max_total=5)

    # イベント警告
    warns = build_event_warnings(today_date)

    # ---------------------------
    # レポート本文
    # ---------------------------
    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")

    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推奨レバ: 約{target_lev:.1f}倍（{lev_label}） / Aランク最大5銘柄")
    lines.append(f"- 推定運用資産ベース: 約{int(total_asset):,}円")
    if warns:
        lines.extend(warns)
    lines.append("")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sector_text)
    lines.append("")

    # ---------------------------
    # Aランク
    # ---------------------------
    lines.append("◆ Core候補 Aランク（本命・最大5銘柄）")
    if not A_sel:
        lines.append("Aランク候補なし（無理IN禁止）。\n")
    else:
        for r in A_sel:
            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{r['price']:.1f}"
            )
            lines.append(
                f"    ・IN目安: {r['price']:.1f}"
                f"\n    ・利確目安: +{r['tp_pct']*100:.1f}%（{r['tp_price']:.1f}）"
                f"\n    ・損切り目安: {r['sl_pct']*100:.1f}%（{r['sl_price']:.1f}）"
            )
            lines.append("")

    # ---------------------------
    # Bランク
    # ---------------------------
    lines.append("◆ Core候補 Bランク（押し目候補）")
    if len(A_sel) >= 5:
        lines.append("Aランク5銘柄で枠が埋まったため、Bランクは省略。\n")
    else:
        if not B_sel:
            lines.append("Bランク候補なし。\n")
        else:
            for r in B_sel:
                lines.append(
                    f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{r['price']:.1f}"
                )
            lines.append("")

    return "\n".join(lines)


# ============================================================
# ポジションレポート
# ============================================================
def build_position_report(today_str, pos_text):
    lines = []
    lines.append(f"📊 {today_str} ポジション分析\n")
    lines.append("◆ ポジションサマリ")
    lines.append(pos_text.strip())
    return "\n".join(lines)


# ============================================================
# LINE送信（ワーカー互換・分割）
# ============================================================
def send_line(text):
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定 → printのみ")
        print(text)
        return

    chunk = 3800
    parts = [text[i:i + chunk] for i in range(0, len(text), chunk)]

    for p in parts:
        try:
            r = requests.post(WORKER_URL, json={"text": p}, timeout=12)
            print("[LINE RESULT]", r.status_code, r.text)
        except Exception as e:
            print("[ERROR] LINE送信失敗:", e)
            print(p)


# ============================================================
# MAIN
# ============================================================
def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    # 市場
    mkt = calc_market_score()

    # ポジション
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev, risk = analyze_positions(pos_df)

    # レポート作成
    core = build_core_report(today_str, today_date, mkt, total_asset)
    posrep = build_position_report(today_str, pos_text)

    # print
    print(core)
    print("\n" + "=" * 60 + "\n")
    print(posrep)

    # 送信（2通）
    send_line(core)
    send_line(posrep)


if __name__ == "__main__":
    main()