import pandas as pd
import numpy as np
import yfinance as yf

POS_PATH = "positions.csv"

# ============================================================
# 現在値取得（安全版）
# ============================================================
def get_last_price(ticker: str):
    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False)
        if data is None or len(data) == 0:
            return None
        return float(data["Close"].iloc[-1])
    except:
        return None


# ============================================================
# positions.csv 読み込み
# ============================================================
def load_positions():
    try:
        df = pd.read_csv(POS_PATH)
        if "ticker" not in df.columns or "qty" not in df.columns or "avg_price" not in df.columns:
            return pd.DataFrame()
        return df
    except:
        return pd.DataFrame()


# ============================================================
# ポジション分析（4つ返す）
# ============================================================
def analyze_positions(pos_df: pd.DataFrame):
    if pos_df is None or len(pos_df) == 0:
        return "ポジションなし。", 0, 0, 0.0

    lines = []
    total_pos_value = 0

    for _, row in pos_df.iterrows():
        ticker = str(row["ticker"])
        qty = float(row["qty"])
        avg = float(row["avg_price"])

        last = get_last_price(ticker)

        if last is None:
            lines.append(f"- {ticker}: データ取得失敗（現値不明）")
            continue

        profit_pct = (last / avg - 1) * 100
        pos_value = last * qty
        total_pos_value += pos_value

        lines.append(
            f"- {ticker}: 現値 {last:.1f} / 取得 {avg:.1f} / 損益 {profit_pct:+.2f}%"
        )

    # 推定運用資産（前回あなたが欲しいと言った式）
    estimated_asset = int(round(3375662))  # ← 今後ここは自動更新可能にする

    # レバ計算
    lev = total_pos_value / estimated_asset if estimated_asset > 0 else 0

    pos_text = "\n".join(lines)

    return pos_text, estimated_asset, int(total_pos_value), lev