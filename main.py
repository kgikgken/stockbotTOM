from __future__ import annotations
import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.util import jst_today_str


UNIVERSE_PATH = "universe_jpx.csv"
WORKER_URL = os.getenv("WORKER_URL")


# ========================
# データ取得（安全版）
# ========================
def fetch_history(ticker: str, period="130d"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# ========================
# スクリーニング
# ========================
def run_screening():
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except:
        return []

    results = []

    for _, row in uni.iterrows():
        ticker = str(row["ticker"])
        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))
        ed = row.get("earnings_date", "")

        # 決算フィルタ
        if isinstance(ed, str) and "-" in ed:
            try:
                ed_dt = pd.to_datetime(ed)
                today = pd.to_datetime(jst_today_str())
                if abs((ed_dt - today).days) <= 3:
                    continue
            except:
                pass

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        score = score_stock(hist)
        if score is None:
            continue

        price = float(hist["Close"].iloc[-1])
        vola20 = hist["Close"].pct_change().rolling(20).std().iloc[-1]

        if vola20 is None or np.isnan(vola20):
            vola20 = 0.02

        # ボラティリティ分類
        if vola20 < 0.015:
            vola_class = "low"
            tp, sl = 6, -3
        elif vola20 < 0.03:
            vola_class = "mid"
            tp, sl = 8, -4
        else:
            vola_class = "high"
            tp, sl = 12, -6

        results.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": score,
            "price": price,
            "tp": tp,
            "sl": sl,
        })

    # スコア順に並び替え
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


# ========================
# 建て玉最大金額
# ========================
def calc_max_position(total_asset, leverage):
    return int(total_asset * leverage)


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
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

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
    today = jst_today_str()

    # ---- 市場 ----
    mkt = calc_market_score()
    mkt_score = mkt["score"]
    mkt_comment = mkt["comment"]
    lev = mkt["leverage"]
    est_asset = mkt["asset"]

    # ---- max建て玉 ----
    max_pos = calc_max_position(est_asset, lev)

    # ---- トップセクター ----
    secs = top_sectors_5d()
    if secs:
        sector_text = "\n".join([f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)])
    else:
        sector_text = "算出不可（データ不足）"

    # ---- イベント ----
    event_text = "特筆すべきイベントなし（通常モード）"

    # ---- スクリーニング ----
    results = run_screening()

    A_list = results[:3]   # ←最大3銘柄
    B_list = []            # Aが満たない場合のみ使うが今回は0固定

    # ---- ポジション ----
    pos_df = load_positions("positions.csv")
    pos_text, total_asset, total_pos, lev_used, risk_info = analyze_positions(pos_df)

    # ========================
    #  assemble
    # ========================
    lines = []

    lines.append(f"📅 {today} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推奨レバ: 約{lev:.1f}倍（Aランク最大3銘柄）")
    lines.append(f"- 推定運用資産ベース: 約{est_asset:,}円\n")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sector_text + "\n")

    lines.append("◆ 今日のイベント・警戒情報")
    lines.append(f"- {event_text}\n")

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

    lines.append("📊 ポジション分析")
    lines.append(pos_text + "\n")

    lines.append("◆ 本日の建て玉最大金額")
    lines.append(f"- 推奨レバ: {lev:.1f}倍")
    lines.append(f"- 運用資産ベース: 約{est_asset:,}円")
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