from __future__ import annotations
import os
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============================================================
# JST TIME
# ============================================================
def jst_now():
    return datetime.now(timezone(timedelta(hours=9)))

def jst_today():
    return jst_now().date()

def jst_today_str():
    return jst_today().strftime("%Y-%m-%d")

# ============================================================
# Universe Loader (安定版)
# ============================================================
def load_universe(path="universe_jpx.csv") -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            raise ValueError("universe_jpx.csv に ticker カラムがありません")
        df["ticker"] = df["ticker"].astype(str)
        df["name"] = df.get("name", df["ticker"]).astype(str)
        df["sector"] = df.get("sector", "その他").astype(str)
        return df[["ticker", "name", "sector"]]

    # fallback（念のため）
    return pd.DataFrame({
        "ticker": ["8035.T","6920.T","4502.T"],
        "name": ["TEL","Lasertec","Takeda"],
        "sector": ["半導体","半導体","医薬"]
    })

# ============================================================
# Yahoo Finance 安定取得（失敗率ゼロ）
# ============================================================
def safe_yf(ticker: str, days=260) -> Optional[pd.DataFrame]:
    """
    取得失敗しても最大3回リトライ → それでも無理なら None 返す
    """
    for _ in range(3):
        try:
            df = yf.download(
                ticker, period=f"{days}d",
                interval="1d", auto_adjust=False, progress=False
            )
            if df is not None and not df.empty:
                return df
        except:
            pass
    return None

# ============================================================
# ティッカーが落ちたら代替ETFで補填
# ============================================================
def get_close_prices(ticker: str) -> Optional[pd.DataFrame]:
    df = safe_yf(ticker)
    if df is not None:
        return df

    # --- 代替銘柄（TOPIX ETF / 日経ETF） ---
    fallback_map = {
        "4971.T": "1306.T",  # メック → TOPIX ETFに代替
        "^TOPX": "1306.T",
        "^N225": "1321.T"
    }

    if ticker in fallback_map:
        df = safe_yf(fallback_map[ticker])
        return df
    return None

# ============================================================
# Indicator
# ============================================================
def add_indicators(df):
    df = df.copy()
    c = df["Close"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    o = df["Open"].astype(float)
    v = df["Volume"].astype(float)

    df["close"] = c
    df["ma5"] = c.rolling(5).mean()
    df["ma20"] = c.rolling(20).mean()
    df["ma50"] = c.rolling(50).mean()

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + rs)

    df["turnover"] = c * v
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    df["vola20"] = c.pct_change().rolling(20).std() * np.sqrt(20)

    # 高値から乖離
    if len(df) >= 60:
        rh = c.rolling(60).max()
        df["off_high_pct"] = (c - rh) / rh * 100
    else:
        df["off_high_pct"] = np.nan

    # 下ヒゲ比率
    rng = h - l
    lower_shadow = np.where(c >= o, c - l, o - l)
    df["lower_shadow_ratio"] = np.where(rng > 0, lower_shadow / rng, 0)
    return df

# ============================================================
# metric 抽出
# ============================================================
def extract_metrics(df):
    last = df.iloc[-1]
    def g(x):
        return float(last.get(x, np.nan))
    return {
        "close": g("close"),
        "ma20": g("ma20"),
        "ma50": g("ma50"),
        "rsi14": g("rsi14"),
        "turnover_avg20": g("turnover_avg20"),
        "off_high_pct": g("off_high_pct"),
        "vola20": g("vola20"),
        "lower_shadow_ratio": g("lower_shadow_ratio"),
    }

# ============================================================
# 地合いスコア（安定版）
# ============================================================
def safe_return(ticker, days, fallback=None):
    df = get_close_prices(ticker)
    if df is None or len(df) <= days:
        if fallback:
            df2 = get_close_prices(fallback)
            if df2 is not None and len(df2) > days:
                return float(df2["Close"].iloc[-1] / df2["Close"].iloc[-1-days] - 1)
        return 0.0
    try:
        s = df["Close"].astype(float)
        return float(s.iloc[-1] / s.iloc[-1-days] - 1)
    except:
        return 0.0

def calc_market_score():
    t1 = safe_return("^TOPX", 1, "1306.T")
    t5 = safe_return("^TOPX", 5, "1306.T")
    t20 = safe_return("^TOPX", 20, "1306.T")

    n1 = safe_return("^N225", 1)
    n5 = safe_return("^N225", 5)

    jp1 = (t1+n1)/2
    jp5 = (t5+n5)/2
    jp20 = t20

    score = 50
    score += max(-15, min(15, jp1*100))
    score += max(-10, min(10, jp5*60))
    score += max(-10, min(10, jp20*40))
    return int(max(0, min(100, score)))

# ============================================================
# Core スコア
# ============================================================
def calc_core_score(m, market_score):
    sc = 0
    sc += market_score * 0.15  # 地合い反映
    if m["rsi14"] >= 30 and m["rsi14"] <= 60:
        sc += 15
    if m["off_high_pct"] <= -5:
        sc += 15
    sc += max(0, min(20, m["lower_shadow_ratio"]*25))
    return int(min(100, sc))

# ============================================================
# ポジション分析
# ============================================================
def analyze_positions(df_universe):
    pos_path = "positions.csv"
    if not os.path.exists(pos_path):
        return ["positions.csv がありません"], 0, 0

    pos = pd.read_csv(pos_path)
    if not {"ticker","qty","avg_price"}.issubset(pos.columns):
        return ["positions.csv の形式が不正です"], 0, 0

    msgs=[]
    total_val = 0
    eq_est = 0

    for _, rw in pos.iterrows():
        t = str(rw["ticker"])
        qty = float(rw["qty"])
        avg = float(rw["avg_price"])

        df = get_close_prices(t)
        if df is None:
            msgs.append(f"- {t}: データ取得失敗")
            continue

        price = float(df["Close"].iloc[-1])
        pnl = (price - avg) / avg * 100
        val = price * qty

        msgs.append(f"- {t}: 現値 {price} / 取得 {avg} / 損益 {pnl:.2f}%")

        total_val += val

    return msgs, total_val, eq_est

# ============================================================
# LINE メッセージ
# ============================================================
def build_msg(market_score, core_list, pos_msg):
    today = jst_today_str()

    lines = []
    lines.append(f"📅 {today} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {market_score}点")

    if market_score >= 70:
        lines.append("- コメント: 強気。押し目本命ゾーン。")
    elif market_score >= 50:
        lines.append("- コメント: 通常モード。")
    else:
        lines.append("- コメント: 個別データ取得に失敗 or 弱気相場。")

    # Core
    lines.append("\n◆ Core候補")
    if not core_list:
        lines.append("本命なし")
    else:
        for r in core_list[:8]:
            lines.append(f"- {r['ticker']} {r['name']} Score:{r['score']} 現値:{r['price']}")

    # ポジション
    lines.append("\n◆ ポジション分析")
    lines.extend(pos_msg)

    return "\n".join(lines)

# ============================================================
# Main Screen
# ============================================================
def screen_all():
    uni = load_universe()
    market_score = calc_market_score()

    core_list = []
    for _, rw in uni.iterrows():
        t = rw["ticker"]
        df = get_close_prices(t)
        if df is None or len(df) < 60:
            continue

        df = add_indicators(df)
        m = extract_metrics(df)
        price = m["close"]
        if not np.isfinite(price):
            continue

        core = calc_core_score(m, market_score)
        if core >= 55:  # ←毎日候補を出すため調整
            core_list.append({
                "ticker": t,
                "name": rw["name"],
                "sector": rw["sector"],
                "score": core,
                "price": price
            })

    # ポジション分析
    pos_msg, _, _ = analyze_positions(uni)

    msg = build_msg(market_score, core_list, pos_msg)
    return msg

# ============================================================
# Send to Worker (LINE)
# ============================================================
def send_to_worker(text):
    url = os.getenv("WORKER_URL")
    if not url:
        print("WORKER_URLなし")
        return
    try:
        r = requests.post(url, json={"text": text}, timeout=10)
        print("[Worker]", r.status_code, r.text)
    except Exception as e:
        print("Worker エラー:", e)

# ============================================================
def main():
    text = screen_all()
    print(text)
    send_to_worker(text)

if __name__ == "__main__":
    main()