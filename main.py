from __future__ import annotations
import os
import math
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "MIN_PRICE": 300.0,
    "MIN_TURNOVER": 1e8,
    "CORE_SCORE_MIN": 75.0,
    "VOL_LOW_TH": 0.02,
    "VOL_HIGH_TH": 0.06,
    "TP_MIN": 0.06,
    "TP_MAX": 0.15,
    "SL_UPPER": -0.03,
    "SL_LOWER": -0.06,
}

# ============================================================
# Utility
# ============================================================
def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))

def jst_today() -> date:
    return jst_now().date()

# ============================================================
# Universe
# ============================================================
def load_universe(path="universe_jpx.csv") -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["ticker"] = df["ticker"].astype(str)
        df["name"]   = df.get("name", df["ticker"]).astype(str)
        df["sector"] = df.get("sector", "その他").astype(str)
        return df[["ticker","name","sector"]]

    # fallback
    return pd.DataFrame({
        "ticker":["8035.T","6920.T"],
        "name":["TokyoElectron","Lasertec"],
        "sector":["半導体","半導体"]
    })

# ============================================================
# OHLCV（強化版）
# ============================================================
def fetch_ohlcv(ticker: str, period="260d") -> Optional[pd.DataFrame]:
    """堅牢版 – 3段階で日足取得"""
    # --- ① 通常
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
        if df is not None and not df.empty:
            return df
    except: pass

    # --- ② fallback（90日）
    try:
        df = yf.download(ticker, period="90d", interval="1d", auto_adjust=False, progress=False)
        if df is not None and not df.empty:
            return df
    except: pass

    # --- ③ fallback（5日）
    try:
        df = yf.download(ticker, period="5d", interval="1d", auto_adjust=False, progress=False)
        if df is not None and not df.empty:
            return df
    except: pass

    print(f"[WARN] 強化版でも取得失敗: {ticker}")
    return None

# ============================================================
# Indicators
# ============================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol   = df["Volume"].astype(float)

    df["close"] = close
    df["ma5"]  = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    # RSI14
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    df["vola20"] = close.pct_change().rolling(20).std() * np.sqrt(20)

    if len(close) >= 60:
        rh = close.rolling(60).max()
        df["off_high_pct"] = (close - rh) / rh * 100
        tail = close.tail(60)
        idx = int(np.argmax(tail.values))
        df["days_since_high60"] = (len(tail) - 1) - idx
    else:
        df["off_high_pct"] = np.nan
        df["days_since_high60"] = np.nan

    df["trend_slope20"] = df["ma20"].pct_change()

    rng = high - low
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    df["lower_shadow_ratio"] = np.where(rng > 0, lower_shadow / rng, 0.0)

    return df

def extract_metrics(df: pd.DataFrame) -> Dict[str,float]:
    last = df.iloc[-1]
    keys = ["close","ma5","ma20","ma50","rsi14","turnover_avg20","off_high_pct",
            "vola20","trend_slope20","lower_shadow_ratio","days_since_high60"]
    return {k: float(last.get(k, np.nan)) for k in keys}

# ============================================================
# Market Score（安全版）
# ============================================================
def safe_download_ret(ticker: str, days: int) -> float:
    try:
        df = yf.download(ticker, period="90d", interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty or len(df) <= days:
            return 0.0
        c = df["Close"].astype(float)
        return float(c.iloc[-1] / c.iloc[-(days + 1)] - 1)
    except:
        return 0.0

def calc_market_score() -> int:
    topix1  = safe_download_ret("1306.T", 1)
    topix5  = safe_download_ret("1306.T", 5)
    topix20 = safe_download_ret("1306.T", 20)

    nk1 = safe_download_ret("^N225", 1)
    nk5 = safe_download_ret("^N225", 5)

    jp1  = (topix1 + nk1) / 2
    jp5  = (topix5 + nk5) / 2
    jp20 = topix20

    score = 50
    score += max(-15, min(15, jp1 * 100))
    score += max(-10, min(10, jp5 * 50))
    score += max(-10, min(10, jp20 * 20))

    return int(min(100, max(0, score)))

# ============================================================
# Scoring
# ============================================================
def calc_trend_score(m):
    close, ma20, ma50, slope, off = m["close"], m["ma20"], m["ma50"], m["trend_slope20"], m["off_high_pct"]
    s=0
    if np.isfinite(slope):
        if slope>=0.01: s+=8
        elif slope>0:  s+=4 + slope/0.01*4
        else: s+=max(0,4 + slope*50)

    if close>ma20>ma50: s+=8
    elif close>ma20:    s+=4
    elif ma20>ma50:     s+=2

    if np.isfinite(off):
        if off>=-5: s+=4
        elif off>=-15: s+=4 - abs(off+5)*0.2

    return int(min(20,max(0,s)))

def calc_pullback_score(m):
    rsi, off, days, shadow = m["rsi14"], m["off_high_pct"], m["days_since_high60"], m["lower_shadow_ratio"]
    s=0
    if 30<=rsi<=45: s+=7
    elif 20<=rsi<30 or 45<rsi<=55: s+=4
    else: s+=1

    if -12<=off<=-5: s+=6
    elif -20<=off<-12: s+=3
    else: s+=1

    if 2<=days<=10: s+=4
    elif 1<=days<2 or 10<days<=20: s+=2

    if shadow>=0.5: s+=3
    elif shadow>=0.3: s+=1

    return int(min(20,max(0,s)))

def calc_liquidity_score(m):
    t, v = m["turnover_avg20"], m["vola20"]
    s=0
    if t>=10e8: s+=16
    elif t>=1e8: s+=16*(t-1e8)/9e8

    if v<0.02: s+=4
    elif v<0.06: s+=4*(0.06-v)/0.04

    return int(min(20,max(0,s)))

def calc_core_score(m, market_score, sector_score):
    return int(min(100, 
        min(20, market_score*0.2) +
        min(20, sector_score*0.2) +
        calc_trend_score(m) +
        calc_pullback_score(m) +
        calc_liquidity_score(m)
    ))

# ============================================================
# TP/SL
# ============================================================
def classify_volatility(v):
    if not np.isfinite(v): return "mid"
    if v<CONFIG["VOL_LOW_TH"]: return "low"
    if v>CONFIG["VOL_HIGH_TH"]: return "high"
    return "mid"

def calc_tp_sl(core, market, vola):
    if core<75: tp=0.06
    elif core<80: tp=0.08
    elif core<90: tp=0.10
    else: tp=0.12 + (core-90)/10*0.03

    if market>=70: tp+=0.02
    elif 40<=market<50: tp-=0.02
    elif market<40: tp-=0.04
    tp = max(CONFIG["TP_MIN"], min(CONFIG["TP_MAX"], tp))

    vc = classify_volatility(vola)
    if vc=="low": sl=-0.035
    elif vc=="high": sl=-0.055
    else: sl=-0.045

    if market>=70: sl-=0.005
    elif market<40: sl+=0.005

    sl = max(CONFIG["SL_LOWER"], min(CONFIG["SL_UPPER"], sl))
    return tp, sl

# ============================================================
# OUT signals
# ============================================================
def evaluate_exit(df):
    sig=[]
    last=df.iloc[-1]
    rsi=float(last.get("rsi14",np.nan))
    turn=float(last.get("turnover",np.nan))
    avg20=float(last.get("turnover_avg20",np.nan))

    if rsi>=70: sig.append("RSI過熱")
    if len(df)>=3:
        d=df.tail(3)
        if (d["close"]<d["ma5"]).iloc[-2:].all():
            sig.append("5MA割れ連続")
    if avg20>0 and turn < 0.5*avg20:
        sig.append("出来高急減")
    return sig

# ============================================================
# Sector score
# ============================================================
def calc_sector_strength(sector: str) -> int:
    return 50

# ============================================================
# Position analysis
# ============================================================
def analyze_positions(market_score: int) -> List[str]:
    """positions.csv を読み、分析してメッセージを返す"""
    if not os.path.exists("positions.csv"):
        return ["positions.csv がありません"]

    df = pd.read_csv("positions.csv")
    if "ticker" not in df.columns or "qty" not in df.columns or "avg_price" not in df.columns:
        return ["positions.csv のカラムが不完全です"]

    msgs=[]

    for _,row in df.iterrows():
        ticker=row["ticker"]
        qty=row["qty"]
        avg=row["avg_price"]

        ohlcv = fetch_ohlcv(ticker)
        if ohlcv is None:
            msgs.append(f"{ticker}: データ取得失敗")
            continue

        ohlcv = add_indicators(ohlcv)
        if len(ohlcv)<5:
            msgs.append(f"{ticker}: データ不足")
            continue

        price = float(ohlcv["close"].iloc[-1])
        pnl = (price-avg)/avg*100

        msgs.append(f"{ticker}: 現値 {price:.1f} / 取得 {avg:.1f} / 損益 {pnl:+.2f}%")

    return msgs

# ============================================================
# Line message
# ============================================================
def build_message(ds, market_score, core_list, pos_msg):
    lines=[]
    mlb=("中立","中立〜やや攻め","やや攻め","攻めMAX")

    # 結論
    lines.append(f"📅 {ds} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lev = "中立"
    if market_score>=80: lev="攻めMAX"
    elif market_score>=70: lev="やや攻め"
    elif market_score>=60: lev="中立〜やや攻め"
    lines.append(f"- 地合いスコア: {market_score}点（{lev}）")
    lines.append("")

    # Core
    lines.append("◆ Core候補")
    if not core_list:
        lines.append("本命なし")
    else:
        for i,r in enumerate(core_list[:10],1):
            lines.append(f"{i}. {r['ticker']} {r['name']} Score:{r['score']}")

    # ポジション
    lines.append("\n◆ ポジション分析")
    for t in pos_msg:
        lines.append(f"- {t}")

    return "\n".join(lines)

# ============================================================
# Main
# ============================================================
def screen_all() -> str:
    ds = jst_today().strftime("%Y-%m-%d")
    market_score = calc_market_score()

    universe = load_universe()
    core=[]

    for _,rw in universe.iterrows():
        t=rw["ticker"]; name=rw["name"]; sec=rw["sector"]
        df = fetch_ohlcv(t)
        if df is None: continue
        df = add_indicators(df)
        if len(df)<60: continue

        m = extract_metrics(df)
        price=m["close"]

        if price<CONFIG["MIN_PRICE"]: continue
        if m["turnover_avg20"]<CONFIG["MIN_TURNOVER"]: continue

        sec_s = calc_sector_strength(sec)
        score = calc_core_score(m, market_score, sec_s)
        if score<CONFIG["CORE_SCORE_MIN"]: continue

        vola=m["vola20"]
        tp,sl=calc_tp_sl(score, market_score, vola)

        core.append({
            "ticker":t,
            "name":name,
            "score":score,
            "price":price,
        })

    core.sort(key=lambda x:x["score"], reverse=True)
    pos_msg = analyze_positions(market_score)

    return build_message(ds, market_score, core, pos_msg)

# ============================================================
# LINE Send
# ============================================================
def send_to_lineworker(text: str):
    url=os.getenv("WORKER_URL")
    if not url:
        print("[INFO] WORKER_URL 未設定")
        return
    try:
        r=requests.post(url, json={"text":text}, timeout=15)
        print("[Worker]",r.status_code,r.text)
    except Exception as e:
        print("[WorkerError]",e)

# ============================================================
# Execute
# ============================================================
def main():
    msg = screen_all()
    print(msg)
    send_to_lineworker(msg)

if __name__=="__main__":
    main()