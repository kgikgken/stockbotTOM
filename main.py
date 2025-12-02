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

# 決算日フィルタ：±N日を除外
EARNINGS_EXCLUDE_DAYS = 3

# イベントカレンダーCSV（任意）
EVENTS_PATH = "events.csv"


# ========================
# 日付系
# ========================
def jst_today_date() -> datetime.date:
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).date()


# ========================
# データ取得（安全版）
# ========================
def fetch_history(ticker: str, period="130d") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


# ========================
# RSI
# ========================
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ========================
# イベント読み込み & テキスト生成
# ========================
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
        if not date_str or not label:
            continue
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            continue
        impact = str(row.get("impact", "")).strip()
        events.append({"date": d, "label": label, "impact": impact})
    return events


def build_event_text(today: datetime.date) -> str:
    """
    events.csv から重要イベントを読み取り、
    今日を基準に -1〜+2日 のものを警告として出す。
    """
    events = load_events()
    if not events:
        return "特筆すべきイベントなし（通常モード）"

    warns: List[str] = []
    for ev in events:
        d = ev["date"]
        delta = (d - today).days
        if -1 <= delta <= 2:
            if delta > 1:
                when = f"{delta}日後"
            elif delta == 1:
                when = "明日"
            elif delta == 0:
                when = "本日"
            else:
                when = "直近"

            impact = f" [{ev['impact']}]" if ev.get("impact") else ""
            warns.append(f"⚠ {ev['label']}{impact}（{when}）: ポジションサイズ注意")

    if not warns:
        return "特筆すべきイベントなし（通常モード）"

    return "\n".join(warns)


# ========================
# 建て玉最大金額
# ========================
def calc_max_position(total_asset: float, leverage: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0):
        return 0
    if not (np.isfinite(leverage) and leverage > 0):
        return int(total_asset)
    return int(total_asset * leverage)


# ========================
# スクリーニング
# ========================
def run_screening(
    today: datetime.date,
    mkt_score: int,
) -> Tuple[List[Dict], List[Dict]]:
    """
    世界最高トレーダー仕様のスクリーニング。
    - 決算±3日は除外
    - 20MA > 60MA の上昇トレンド優先
    - 出来高フィルタ
    - RSI 35〜70 を高評価（過熱/ド底は減点）
    - セクター強度で加点
    戻り値: (A_list, B_list)  ※まだ数は絞らない
    """
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], []

    if "ticker" not in uni.columns:
        return [], []

    # earnings_dateパース
    if "earnings_date" in uni.columns:
        uni["earnings_date_parsed"] = pd.to_datetime(
            uni["earnings_date"], errors="coerce"
        ).dt.date
    else:
        uni["earnings_date_parsed"] = pd.NaT

    # セクター強度（top_sectors_5dの順位で加点）
    sectors_5d = top_sectors_5d()
    sector_bonus: Dict[str, float] = {}
    for rank, (name, chg) in enumerate(sectors_5d, start=1):
        bonus = max(0, 4 - rank)  # 1位:3, 2位:2, 3位:1
        sector_bonus[name] = float(bonus)

    results: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        # 決算前後フィルタ
        ed = row.get("earnings_date_parsed")
        if isinstance(ed, datetime) or isinstance(ed, datetime.date):
            try:
                delta = abs((ed - today).days)
                if delta <= EARNINGS_EXCLUDE_DAYS:
                    continue
            except Exception:
                pass

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        close = hist["Close"].astype(float)
        vol = hist["Volume"].astype(float)

        # 出来高フィルタ（過去20日平均）
        avg_vol20 = vol.rolling(20).mean().iloc[-1]
        if not np.isfinite(avg_vol20) or avg_vol20 < 50000:
            # 流動性低すぎる銘柄は除外
            continue

        # トレンド（20MA > 60MA を高評価）
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        if not (np.isfinite(ma20) and np.isfinite(ma60)):
            continue
        uptrend = ma20 > ma60

        # RSI
        rsi_series = calc_rsi(close, period=14)
        rsi_last = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

        # ベーススコア
        base_score = score_stock(hist)
        if base_score is None or not np.isfinite(base_score):
            continue
        score = float(base_score)

        # トレンド加点/減点
        if uptrend:
            score += 5.0
        else:
            score -= 5.0

        # RSI調整
        if 40 <= rsi_last <= 65:
            score += 3.0
        elif rsi_last > 75 or rsi_last < 30:
            score -= 5.0

        # セクター強度加点
        score += sector_bonus.get(sector, 0.0)

        price = float(close.iloc[-1])

        # ボラから TP / SL を決定
        vola20 = close.pct_change().rolling(20).std().iloc[-1]
        if not np.isfinite(vola20):
            vola20 = 0.02

        if vola20 < 0.015:
            tp, sl = 6, -3
        elif vola20 < 0.03:
            tp, sl = 8, -4
        else:
            tp, sl = 12, -6

        # 地合いで微調整（弱いときは少し守り）
        if mkt_score < 45:
            tp = max(tp - 2, 4)
            sl = min(sl, -3)

        results.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "score": score,
                "price": price,
                "tp": tp,
                "sl": sl,
            }
        )

    # スコア順でソート
    results.sort(key=lambda x: x["score"], reverse=True)

    A_list: List[Dict] = []
    B_list: List[Dict] = []

    for r in results:
        if r["score"] >= 90:
            A_list.append(r)
        elif r["score"] >= 82:
            B_list.append(r)

    return A_list, B_list


# ========================
# ライン送信（三分割）
# ========================
def send_line(text: str):
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定 → print のみ")
        print(text)
        return

    # 分割送信
    chunk_size = 3900
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    for ch in chunks:
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=10)
            print("[LINE RESULT]", r.status_code, r.text)
        except Exception as e:
            print("[ERROR] LINE送信失敗:", e)


# ========================
# レポート生成
# ========================
def build_report():
    today_str = jst_today_str()
    today_date = jst_today_date()

    # ---- 市場 ----
    mkt = calc_market_score()
    mkt_score = int(mkt["score"])
    mkt_comment = mkt["comment"]
    lev = float(mkt["leverage"])
    est_asset = float(mkt["asset"])

    # ---- max建て玉 ----
    max_pos = calc_max_position(est_asset, lev)

    # ---- トップセクター ----
    secs = top_sectors_5d()
    if secs:
        sector_text = "\n".join(
            [f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)]
        )
    else:
        sector_text = "算出不可（データ不足）"

    # ---- イベント ----
    event_text = build_event_text(today_date)

    # ---- スクリーニング ----
    A_all, B_all = run_screening(today_date, mkt_score)

    # 最大3銘柄に絞るロジック
    A_list = A_all[:3]
    if len(A_list) >= 3:
        B_list: List[Dict] = []
    else:
        need = 3 - len(A_list)
        B_list = B_all[:need]

    # ---- ポジション ----
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev_used, risk_info = analyze_positions(pos_df)

    # ========================
    #  assemble
    # ========================
    lines: List[str] = []

    lines.append(f"📅 {today_str} stockbotTOM 日報\n")

    # 戦略サマリ（E）
    lines.append("◆ 今日の戦略まとめ")
    lines.append(
        f"- 地合い: {mkt_score}点（{mkt_comment}） / 推奨レバ: 約{lev:.1f}倍"
    )
    lines.append(
        f"- 本命候補: Aランク {len(A_list)}銘柄 / 補欠Bランク {len(B_list)}銘柄"
    )
    lines.append(f"- 今日のMAX建て玉: 約{max_pos:,}円\n")

    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推奨レバ: 約{lev:.1f}倍（標準（押し目のみ））")
    lines.append(f"- 推定運用資産ベース: 約{int(est_asset):,}円\n")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sector_text + "\n")

    lines.append("◆ 今日のイベント・警戒情報")
    lines.append(f"{event_text}\n")

    lines.append("◆ Core候補 Aランク（本命押し目・最大3銘柄）")
    if not A_list:
        lines.append("Aランクなし\n")
    else:
        for r in A_list:
            p = r["price"]
            tp_price = p * (1 + r["tp"] / 100)
            sl_price = p * (1 + r["sl"] / 100)

            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{p:.1f}\n"
                f"    ・IN目安: {p:.1f}\n"
                f"    ・利確目安: {r['tp']}%（{tp_price:.1f}）\n"
                f"    ・損切り目安: {r['sl']}%（{sl_price:.1f}）"
            )
        lines.append("")

    lines.append("◆ Core候補 Bランク（押し目候補・ロット控えめ）")
    if not B_list:
        if A_list:
            lines.append("Aランクで枠が埋まっているため、Bランク表示は省略。\n")
        else:
            lines.append("Bランク候補なし。\n")
    else:
        for r in B_list:
            p = r["price"]
            tp_price = p * (1 + r["tp"] / 100)
            sl_price = p * (1 + r["sl"] / 100)

            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{p:.1f}\n"
                f"    ・利確目安: {r['tp']}%（{tp_price:.1f}） / "
                f"損切り目安: {r['sl']}%（{sl_price:.1f}）"
            )
        lines.append("")

    lines.append("📊 ポジション分析")
    lines.append(pos_text + "\n")

    lines.append("◆ 本日の建て玉最大金額")
    lines.append(f"- 推奨レバ: {lev:.1f}倍")
    lines.append(f"- 運用資産ベース: 約{int(est_asset):,}円")
    lines.append(f"- 今日のMAX建て玉: 約{max_pos:,}円")

    return "\n".join(lines)


# ========================
# Entry
# ========================
def main():
    text = build_report()
    print(text)
    send_line(text)


if __name__ == "__main__":
    main()