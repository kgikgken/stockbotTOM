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

# スコア分類
A_MIN = 85.0
B_MIN = 80.0
MAX_PICKS = 5   # A+B 合計最大5


# ========================
# 株価データ
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
# TP/SL をスコアとボラから決定
# ========================
def decide_tp_sl(score: float, vola: float):
    if vola < 0.015:
        tp, sl = 0.06, -0.03
    elif vola < 0.03:
        tp, sl = 0.08, -0.04
    else:
        tp, sl = 0.12, -0.06

    if score >= 90:
        tp += 0.02
    if score < 83:
        tp -= 0.01

    tp = float(np.clip(tp, 0.04, 0.15))
    sl = float(np.clip(sl, -0.08, -0.02))
    return tp, sl


# ========================
# スクリーニング（決算 ±3日弾く + A/B スコア分類）
# ========================
def run_screening():
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except:
        return [], []

    A = []
    B = []

    for _, row in uni.iterrows():
        ticker = str(row["ticker"])
        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))
        ed = row.get("earnings_date", "")

        # 決算日前後 ±3日は除外（空欄は除外しない）
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
        score = float(score)

        price = float(hist["Close"].iloc[-1])
        vola20 = hist["Close"].pct_change().rolling(20).std().iloc[-1]
        if vola20 is None or np.isnan(vola20):
            vola20 = 0.02

        tp, sl = decide_tp_sl(score, vola20)

        rec = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": score,
            "price": price,
            "tp": tp,
            "sl": sl,
        }

        if score >= A_MIN:
            A.append(rec)
        elif score >= B_MIN:
            B.append(rec)

    A.sort(key=lambda x: x["score"], reverse=True)
    B.sort(key=lambda x: x["score"], reverse=True)

    return A, B


# ========================
# A優先で最大5銘柄
# ========================
def select_candidates(A, B):
    A_sel = A[:MAX_PICKS]
    remain = MAX_PICKS - len(A_sel)
    B_sel = B[:remain] if remain > 0 else []
    return A_sel, B_sel


# ========================
# 建て玉最大金額
# ========================
def calc_max_position(total_asset, lev):
    return int(total_asset * lev)


# ========================
# LINE送信（あなたのWorker対応版・絶対に壊さない）
# ========================
def send_line(text: str):
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定 → print のみ")
        print(text)
        return

    chunk = 3800
    chunks = [text[i:i + chunk] for i in range(0, len(text), chunk)]

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

    # 市場
    mkt = calc_market_score()
    mkt_score = mkt["score"]
    mkt_comment = mkt["comment"]
    lev = mkt["leverage"]
    est_asset = mkt["asset"]

    max_pos = calc_max_position(est_asset, lev)

    # セクター
    secs = top_sectors_5d()
    if secs:
        sector_text = "\n".join([f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)])
    else:
        sector_text = "算出不可"

    # イベント（自動判断は後で追加予定）
    event_text = "特筆すべきイベントなし（通常モード）"

    # スクリーニング
    A_all, B_all = run_screening()
    A_list, B_list = select_candidates(A_all, B_all)

    # ポジション
    pos_df = load_positions("positions.csv")
    pos_text, total_asset, total_pos, lev_used, risk_info = analyze_positions(pos_df)

    # assemble
    lines = []

    lines.append(f"📅 {today} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推奨レバ: 約{lev:.1f}倍（Aランク最大{MAX_PICKS}銘柄）")
    lines.append(f"- 推定運用資産ベース: 約{est_asset:,}円\n")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sector_text + "\n")

    lines.append("◆ 今日のイベント・警戒情報")
    lines.append(f"- {event_text}\n")

    lines.append(f"◆ Core候補 Aランク（最大{MAX_PICKS}銘柄）")
    if not A_list:
        lines.append("Aランクなし\n")
    else:
        for r in A_list:
            p = r["price"]
            tp_price = p * (1 + r["tp"])
            sl_price = p * (1 + r["sl"])
            lines.append(
                f"- {r['ticker']} {r['name']} Score:{r['score']:.1f} 現値:{p:.1f}\n"
                f"    ・IN目安: {p:.1f}\n"
                f"    ・利確目安: {r['tp']*100:.1f}%（{tp_price:.1f}）\n"
                f"    ・損切り目安: {r['sl']*100:.1f}%（{sl_price:.1f}）"
            )
        lines.append("")

    lines.append("◆ Core候補 Bランク")
    if not B_list:
        if A_list:
            lines.append("Aランクで枠が埋まっているため省略。\n")
        else:
            lines.append("Bランク候補なし。\n")
    else:
        for r in B_list:
            p = r["price"]
            tp_price = p * (1 + r["tp"])
            sl_price = p * (1 + r["sl"])
            lines.append(
                f"- {r['ticker']} {r['name']} Score:{r['score']:.1f} 現値:{p:.1f}\n"
                f"    ・利確目安: {r['tp']*100:.1f}%（{tp_price:.1f}） / 損切り目安: {r['sl']*100:.1f}%（{sl_price:.1f}）"
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