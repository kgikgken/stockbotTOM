"""
main.py - 日本株スイングトレード用 朝イチスクリーニング & 戦略通知ボット（完全版）

・universe_jpx.csv を読み込み
・yfinance で日足取得
・テクニカル指標を計算
・Coreスコア（100点）でスクリーニング
・利確/損切り/レバ推奨値を算出
・LINE通知形式に整形
・Cloudflare WorkerへPOST → LINE通知
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests



# =================================================
# CONFIG — 調整しやすい定数まとめ
# =================================================

CONFIG: Dict[str, float] = {
    # 抽出フィルタ
    "MIN_PRICE": 300.0,
    "MIN_TURNOVER": 1e8,

    # Core候補
    "CORE_SCORE_MIN": 75.0,

    # ボラティリティしきい値
    "VOL_LOW_TH": 0.02,
    "VOL_HIGH_TH": 0.06,

    # 利確幅の下限/上限
    "TP_MIN": 0.06,
    "TP_MAX": 0.15,

    # 損切り幅の上限/下限（負の値）
    "SL_UPPER": -0.03,
    "SL_LOWER": -0.06,
}



# =================================================
# JST Utility
# =================================================

def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))

def jst_today() -> date:
    return jst_now().date()

def jst_today_str() -> str:
    return jst_today().strftime("%Y-%m-%d")

def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)



# =================================================
# Universe 読み込み
# =================================================

def load_universe(path: str = "universe_jpx.csv") -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            raise ValueError("universe_jpx.csv に 'ticker' カラムがありません。")

        df["ticker"] = df["ticker"].astype(str)
        df["name"]   = df.get("name", df["ticker"]).astype(str)
        df["sector"] = df.get("sector", "その他").astype(str)

        return df[["ticker", "name", "sector"]]

    # フォールバック（バックアップ用）
    data = {
        "ticker": ["6920.T", "8035.T", "4502.T", "9984.T", "8316.T", "7203.T"],
    }
    df = pd.DataFrame(data)
    df["name"] = df["ticker"]
    df["sector"] = "その他"
    return df



# =================================================
# OHLCV取得 & インジケータ
# =================================================

def fetch_ohlcv(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"[WARN] fetch_ohlcv failed for {ticker}: {e}")
        return None

    if df is None or df.empty:
        print(f"[WARN] no data for {ticker}")
        return None

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        print(f"[WARN] missing OHLCV columns for {ticker}")
        return None

    return df



def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low  = df["Low"].astype(float)
    open_= df["Open"].astype(float)
    vol  = df["Volume"].astype(float)

    df["close"] = close

    # MA
    df["ma5"]  = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    df["rsi14"] = 100 - 100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean())

    # turnover
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    # 60日高値
    if len(df) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100
    else:
        df["off_high_pct"] = np.nan

    # days_since_high
    tail = close.tail(60)
    if len(tail) == 60:
        idx_max = int(np.argmax(tail.values))
        df["days_since_high60"] = (len(tail) - 1) - idx_max
    else:
        df["days_since_high60"] = np.nan

    # 20日ボラ
    returns = close.pct_change()
    df["vola20"] = returns.rolling(20).std() * np.sqrt(20)

    # slope
    df["trend_slope20"] = df["ma20"].pct_change()

    # 下ヒゲ
    range_ = high - low
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    df["lower_shadow_ratio"] = np.where(range_ > 0, lower_shadow / range_, 0)

    return df



def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    return {
        "close": _safe_float(last.get("close")),
        "ma5": _safe_float(last.get("ma5")),
        "ma20": _safe_float(last.get("ma20")),
        "ma50": _safe_float(last.get("ma50")),
        "rsi14": _safe_float(last.get("rsi14")),
        "turnover_avg20": _safe_float(last.get("turnover_avg20")),
        "off_high_pct": _safe_float(last.get("off_high_pct")),
        "vola20": _safe_float(last.get("vola20")),
        "trend_slope20": _safe_float(last.get("trend_slope20")),
        "lower_shadow_ratio": _safe_float(last.get("lower_shadow_ratio")),
        "days_since_high60": _safe_float(last.get("days_since_high60")),
    }



# =================================================
# 地合い & セクター強度
# =================================================

def calc_market_score() -> int:
    """TOPIX & 日経平均で0〜100点の地合いを算出"""

    def _ret(ticker: str, days: int) -> float:
        try:
            d = yf.download(ticker, period="60d", interval="1d", progress=False)
            close = d["Close"].astype(float)
            if len(close) <= days:
                return 0.0
            return close.iloc[-1] / close.iloc[-(days+1)] - 1
        except Exception:
            return 0.0

    tp1 = _ret("^TOPX", 1)
    tp5 = _ret("^TOPX", 5)
    tp20 = _ret("^TOPX", 20)

    nk1 = _ret("^N225", 1)
    nk5 = _ret("^N225", 5)

    score = 50
    jp1 = (tp1 + nk1)/2
    jp5 = (tp5 + nk5)/2

    score += max(-15, min(15, jp1*100))
    score += max(-10, min(10, jp5*50))
    score += max(-10, min(10, tp20*20))

    return int(max(0, min(100, round(score))))



def calc_sector_strength(sector: str) -> int:
    """簡易セクター強度（全て50点固定）"""
    return 50



# =================================================
# Core Score（100点）
# =================================================

def calc_trend_score(m: Dict[str, float]) -> int:
    score = 0
    slope = m["trend_slope20"]
    close = m["close"]
    ma20 = m["ma20"]
    ma50 = m["ma50"]
    off_high = m["off_high_pct"]

    # slope
    if np.isfinite(slope):
        if slope >= 0.01:
            score += 8
        elif slope > 0:
            score += 4 + (slope/0.01)*4
        else:
            score += max(0, 4 + slope*50)

    # MA条件
    if close > ma20 > ma50:
        score += 8
    elif close > ma20:
        score += 4

    # 高値から距離
    if np.isfinite(off_high):
        if off_high >= -5:
            score += 4
        elif off_high >= -15:
            score += 4 - abs(off_high+5)*0.2

    return max(0, min(20, int(round(score))))



def calc_pullback_score(m: Dict[str, float]) -> int:
    score = 0
    rsi = m["rsi14"]
    off_high = m["off_high_pct"]
    days = m["days_since_high60"]
    shadow = m["lower_shadow_ratio"]

    if 30 <= rsi <= 45:
        score += 7
    elif 20 <= rsi < 30 or 45 < rsi <= 55:
        score += 4
    else:
        score += 1

    if -12 <= off_high <= -5:
        score += 6
    elif -20 <= off_high < -12:
        score += 3
    else:
        score += 1

    if 2 <= days <= 10:
        score += 4
    elif 1 <= days < 2 or 10 < days <= 20:
        score += 2

    if shadow >= 0.5:
        score += 3
    elif shadow >= 0.3:
        score += 1

    return max(0, min(20, int(round(score))))



def calc_liquidity_score(m: Dict[str, float]) -> int:
    turnover = m["turnover_avg20"]
    vola = m["vola20"]

    score = 0

    if turnover >= 10e8:
        score += 16
    elif turnover >= 1e8:
        score += 16*(turnover-1e8)/(9e8)

    if vola < 0.02:
        score += 4
    elif vola < 0.06:
        score += 4*(0.06-vola)/0.04

    return max(0, min(20, int(round(score))))



def calc_core_score(m: Dict[str, float], market_score: int, sector_score: int) -> int:
    score_market = min(20, market_score*0.2)
    score_sector = min(20, sector_score*0.2)

    total = (
        score_market +
        score_sector +
        calc_trend_score(m) +
        calc_pullback_score(m) +
        calc_liquidity_score(m)
    )
    return int(min(100, max(0, round(total))))



# =================================================
# ボラ分類 & 利確/損切り
# =================================================

def classify_volatility(vol: float) -> str:
    if vol < CONFIG["VOL_LOW_TH"]:
        return "low"
    if vol > CONFIG["VOL_HIGH_TH"]:
        return "high"
    return "mid"



def calc_take_profit_and_stop_loss(core_score: int, market_score: int, vol: float) -> Tuple[float, float]:
    # 利確
    if core_score < 80:
        base_tp = 0.08
    elif core_score < 90:
        base_tp = 0.10
    else:
        base_tp = 0.12 + (core_score-90)/10*0.03

    # 地合い補正
    if market_score >= 70:
        base_tp += 0.02
    elif 40 <= market_score < 50:
        base_tp -= 0.02
    elif market_score < 40:
        base_tp -= 0.04

    tp_pct = min(CONFIG["TP_MAX"], max(CONFIG["TP_MIN"], base_tp))

    # 損切り
    vcls = classify_volatility(vol)
    if vcls == "low":
        sl = -0.035
    elif vcls == "high":
        sl = -0.055
    else:
        sl = -0.045

    if market_score >= 70:
        sl -= 0.005
    elif market_score < 40:
        sl += 0.005

    sl_pct = min(CONFIG["SL_UPPER"], max(CONFIG["SL_LOWER"], sl))

    return tp_pct, sl_pct



# =================================================
# OUT シグナル
# =================================================

def evaluate_exit_signals(df: pd.DataFrame) -> List[str]:
    sig = []
    last = df.iloc[-1]

    if last.get("rsi14", 0) >= 70:
        sig.append("RSI過熱")

    if len(df) >= 3:
        s = df.tail(3)
        if (s["close"] < s["ma5"]).iloc[-2:].all():
            sig.append("5MA割れ連続")

    t = last.get("turnover", np.nan)
    t20 = last.get("turnover_avg20", np.nan)
    if np.isfinite(t) and np.isfinite(t20) and t < 0.5*t20:
        sig.append("出来高急減")

    return sig



# =================================================
# 決算・イベント
# =================================================

def is_in_earnings_window(ticker: str, base_date: date, window: int = 3) -> bool:
    try:
        tk = yf.Ticker(ticker)
        ed = tk.get_earnings_dates(limit=4)
        if ed is not None and not ed.empty:
            for idx in ed.index:
                ex = pd.to_datetime(idx).date()
                if abs((ex-base_date).days) <= window:
                    return True
    except Exception:
        pass
    return False


def calc_leverage_advice(market_score: int) -> Tuple[float, str]:
    if market_score >= 80:
        return 2.5, "攻めMAX"
    if market_score >= 70:
        return 2.2, "やや攻め"
    if market_score >= 60:
        return 2.0, "中立〜やや攻め"
    if market_score >= 50:
        return 1.5, "中立"
    if market_score >= 40:
        return 1.2, "守り寄り"
    return 1.0, "守り"



# =================================================
# LINE メッセージ生成
# =================================================

def _fmt_yen(v: float) -> str:
    if not np.isfinite(v):
        return "-"
    return f"{int(round(v)):,}円"


def build_line_message(date_str: str, market_score: int, core_list: List[Dict]) -> str:
    max_lev, lev_label = calc_leverage_advice(market_score)

    lines = []
    lines.append(f"📅 {date_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {market_score}点（{lev_label}）")
    lines.append(f"- レバ目安: 最大 約{max_lev:.1f}倍 / ポジ数目安: 3銘柄前後")
    if market_score >= 70:
        lines.append("- コメント: 押し目狙いは攻め寄り。ただしイベント前のフルベットは避ける。")
    elif market_score >= 50:
        lines.append("- コメント: 通常モード。条件を満たした銘柄のみ厳選IN。")
    elif market_score >= 40:
        lines.append("- コメント: やや守り。無理な新規INはしない。")
    else:
        lines.append("- コメント: 守り優先。様子見〜縮小。")
    lines.append("")

    lines.append("◆ Core候補（本命押し目）")
    if not core_list:
        lines.append("本命条件を満たす銘柄なし。今日は攻めすぎ注意。")
        return "\n".join(lines)

    for i, r in enumerate(core_list[:10], 1):
        code = r["ticker"]
        name = r["name"]
        score = r["score"]

        lines.append(f"{i}. {code} {name}  Score: {score}")

        comment = []
        if score >= 90:
            comment.append("総合◎")
        elif score >= 80:
            comment.append("総合◯")

        if r["trend_score"] >= 15:
            comment.append("トレンド◎")
        if r["pb_score"] >= 12:
            comment.append("押し目良好")
        if r["liq_score"] >= 12:
            comment.append("流動性◎")

        lines.append("   " + " / ".join(comment))

        lines.append(
            f"   現値:{_fmt_yen(r['price'])} "
            f"/ 利確:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['tp_price'])}) "
            f"/ 損切:{r['sl_pct']*100:.1f}%({_fmt_yen(r['sl_price'])})"
        )

        if r["exit_signals"]:
            lines.append("   OUTシグナル: " + " / ".join(r["exit_signals"]))

    return "\n".join(lines)



# =================================================
# 全体スクリーニング
# =================================================

def screen_all() -> str:
    today = jst_today()
    today_str = today.strftime("%Y-%m-%d")
    market_score = calc_market_score()

    try:
        universe = load_universe()
    except Exception as e:
        return f"📅 {today_str}\nuniverse読み込みエラー: {e}"

    core_list = []

    for _, row in universe.iterrows():
        ticker = str(row["ticker"])
        name   = str(row["name"])
        sector = str(row["sector"])

        df = fetch_ohlcv(ticker)
        if df is None:
            continue

        if is_in_earnings_window(ticker, today, window=3):
            print(f"[INFO] skip earnings: {ticker}")
            continue

        df = add_indicators(df)
        if len(df) < 60:
            continue

        m = extract_metrics(df)
        price = m["close"]

        if price < CONFIG["MIN_PRICE"]:
            continue

        if m["turnover_avg20"] < CONFIG["MIN_TURNOVER"]:
            continue

        sector_score = calc_sector_strength(sector)
        core_score = calc_core_score(m, market_score, sector_score)

        if core_score < CONFIG["CORE_SCORE_MIN"]:
            continue

        vol = m["vola20"]
        tp, sl = calc_take_profit_and_stop_loss(core_score, market_score, vol)

        core_list.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "score": core_score,
                "price": price,
                "tp_pct": tp,
                "sl_pct": sl,
                "tp_price": price*(1+tp),
                "sl_price": price*(1+sl),
                "exit_signals": evaluate_exit_signals(df),
                "trend_score": calc_trend_score(m),
                "pb_score": calc_pullback_score(m),
                "liq_score": calc_liquidity_score(m),
            }
        )

    if not core_list:
        max_lev, lev_label = calc_leverage_advice(market_score)
        return (
            f"📅 {today_str} stockbotTOM 日報\n\n"
            f"◆ 今日の結論\n"
            f"- 地合いスコア: {market_score}点（{lev_label}）\n"
            f"- レバ目安: 最大 約{max_lev:.1f}倍\n"
            f"- コメント: 本命押し目なし。ムリなINは控える。"
        )

    core_list.sort(key=lambda x: x["score"], reverse=True)
    return build_line_message(today_str, market_score, core_list)



# =================================================
# Cloudflare Worker → LINE 送信
# =================================================

def send_to_lineworker(text: str):
    WORKER_URL = os.getenv("WORKER_URL")  # ← GitHub Actions で設定

    if not WORKER_URL:
        print("ERROR: WORKER_URL not set")
        return

    try:
        r = requests.post(
            WORKER_URL,
            json={"text": text},
            timeout=15
        )
        print("Worker response:", r.status_code, r.text)
    except Exception as e:
        print("ERROR sending to Worker:", e)



# =================================================
# main()
# =================================================

def main() -> None:
    text = screen_all()
    print(text)
    send_to_lineworker(text)


if __name__ == "__main__":
    main()