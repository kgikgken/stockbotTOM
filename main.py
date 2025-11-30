from __future__ import annotations
import os
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============================================================
# CONFIG（後から調整しやすい定数まとめ）
# ============================================================
CONFIG = {
    "MIN_PRICE": 300.0,       # 最低株価
    "MIN_TURNOVER": 1e8,      # 最低売買代金（直近20日平均）

    "CORE_SCORE_MIN": 72.0,   # Coreスコア閾値（少し緩めてチャンス増やす）

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

def jst_today_str() -> str:
    return jst_today().strftime("%Y-%m-%d")

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)

# ============================================================
# Universe
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

    # fallback（最低限）
    df = pd.DataFrame({
        "ticker": ["8035.T", "6920.T", "4502.T"],
        "name": ["Tokyo Electron", "Lasertec", "Takeda"],
        "sector": ["半導体", "半導体", "医薬"]
    })
    return df

# ============================================================
# OHLCV + Indicators
# ============================================================
def fetch_ohlcv(ticker: str, period="260d") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            ticker, period=period, interval="1d",
            auto_adjust=False, progress=False
        )
    except Exception as e:
        print(f"[WARN] fetch failed {ticker}: {e}")
        return None

    if df is None or df.empty:
        print(f"[WARN] empty data {ticker}")
        return None

    need = {"Open", "High", "Low", "Close", "Volume"}
    if not need.issubset(df.columns):
        print(f"[WARN] missing OHLCV {ticker}")
        return None

    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol   = df["Volume"].astype(float)

    df["close"] = close
    df["ma5"]   = close.rolling(5).mean()
    df["ma20"]  = close.rolling(20).mean()
    df["ma50"]  = close.rolling(50).mean()

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    # ボラ20（20日の日次リターンの標準偏差 × √20）
    df["vola20"] = close.pct_change().rolling(20).std() * np.sqrt(20)

    # 高値から乖離
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100
        tail = close.tail(60)
        idx = int(np.argmax(tail.values))
        df["days_since_high60"] = (len(tail) - 1) - idx
    else:
        df["off_high_pct"] = np.nan
        df["days_since_high60"] = np.nan

    df["trend_slope20"] = df["ma20"].pct_change()

    rng = high - low
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    df["lower_shadow_ratio"] = np.where(rng > 0, lower_shadow / rng, 0)

    return df

def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    keys = [
        "close", "ma5", "ma20", "ma50", "rsi14", "turnover_avg20",
        "off_high_pct", "vola20", "trend_slope20",
        "lower_shadow_ratio", "days_since_high60"
    ]
    return {k: _safe_float(last.get(k, np.nan)) for k in keys}

# ============================================================
# Market Score（安全版）
# ============================================================
def safe_download_close(ticker: str, days: int) -> Optional[pd.Series]:
    try:
        df = yf.download(
            ticker, period="90d", interval="1d",
            auto_adjust=False, progress=False
        )
    except Exception:
        return None
    if df is None or df.empty or "Close" not in df.columns:
        return None
    if len(df) <= days:
        return None
    return df["Close"].astype(float)

def safe_return(ticker: str, days: int, fallback: str = None) -> float:
    s = safe_download_close(ticker, days)
    if s is None:
        if fallback:
            s2 = safe_download_close(fallback, days)
            if s2 is None:
                return 0.0
            return float(s2.iloc[-1] / s2.iloc[-(days + 1)] - 1)
        return 0.0
    try:
        return float(s.iloc[-1] / s.iloc[-(days + 1)] - 1)
    except Exception:
        return 0.0

def calc_market_score() -> int:
    """安全な地合いスコア（0-100）"""
    topix_ret1  = safe_return("^TOPX", 1,  fallback="1306.T")
    topix_ret5  = safe_return("^TOPX", 5,  fallback="1306.T")
    topix_ret20 = safe_return("^TOPX", 20, fallback="1306.T")

    nikkei_ret1 = safe_return("^N225", 1)
    nikkei_ret5 = safe_return("^N225", 5)

    jp1  = (topix_ret1 + nikkei_ret1) / 2
    jp5  = (topix_ret5 + nikkei_ret5) / 2
    jp20 = topix_ret20

    score = 50.0
    score += max(-15, min(15, jp1 * 100))
    score += max(-10, min(10, jp5 * 50))
    score += max(-10, min(10, jp20 * 20))

    score = max(0, min(100, score))
    return int(score)

# ============================================================
# セクター強度（暫定：ここは後で本実装する）
# ============================================================
def calc_sector_strength(sector: str) -> int:
    # ひとまず固定50（中立）。第2弾で本実装する。
    return 50

# ============================================================
# Core スコア（100点）
# ============================================================
def calc_trend_score(m: Dict[str, float]) -> int:
    close = m["close"]
    ma20  = m["ma20"]
    ma50  = m["ma50"]
    slope = m["trend_slope20"]
    off   = m["off_high_pct"]
    sc = 0

    # slope
    if np.isfinite(slope):
        if slope >= 0.01:
            sc += 8
        elif slope > 0:
            sc += 4 + slope / 0.01 * 4
        else:
            sc += max(0, 4 + slope * 50)

    # MA関係
    if np.isfinite(close) and np.isfinite(ma20) and np.isfinite(ma50):
        if close > ma20 and ma20 > ma50:
            sc += 8
        elif close > ma20:
            sc += 4
        elif ma20 > ma50:
            sc += 2

    # 高値系
    if np.isfinite(off):
        if off >= -5:
            sc += 4
        elif off >= -15:
            sc += 4 - abs(off + 5) * 0.2

    return int(max(0, min(20, sc)))

def calc_pullback_score(m: Dict[str, float]) -> int:
    rsi   = m["rsi14"]
    off   = m["off_high_pct"]
    days  = m["days_since_high60"]
    shadow = m["lower_shadow_ratio"]
    sc = 0

    # RSI
    if np.isfinite(rsi):
        if 30 <= rsi <= 45:
            sc += 7
        elif 20 <= rsi < 30 or 45 < rsi <= 55:
            sc += 4
        else:
            sc += 1

    # 下落率
    if np.isfinite(off):
        if -12 <= off <= -5:
            sc += 6
        elif -20 <= off < -12:
            sc += 3
        else:
            sc += 1

    # 日柄
    if np.isfinite(days):
        if 2 <= days <= 10:
            sc += 4
        elif 1 <= days < 2 or 10 < days <= 20:
            sc += 2

    # ヒゲ
    if np.isfinite(shadow):
        if shadow >= 0.5:
            sc += 3
        elif shadow >= 0.3:
            sc += 1

    return int(max(0, min(20, sc)))

def calc_liquidity_score(m: Dict[str, float]) -> int:
    t = m["turnover_avg20"]
    v = m["vola20"]
    sc = 0

    if np.isfinite(t):
        if t >= 10e8:
            sc += 16
        elif t >= 1e8:
            sc += 16 * (t - 1e8) / 9e8

    if np.isfinite(v):
        if v < 0.02:
            sc += 4
        elif v < 0.06:
            sc += 4 * (0.06 - v) / 0.04

    return int(max(0, min(20, sc)))

def calc_core_score(m: Dict[str, float], market_score: int, sector_score: int) -> int:
    s_m = min(20, market_score * 0.2)
    s_s = min(20, sector_score * 0.2)
    s_t = calc_trend_score(m)
    s_p = calc_pullback_score(m)
    s_l = calc_liquidity_score(m)
    return int(min(100, s_m + s_s + s_t + s_p + s_l))

# ============================================================
# Volatility & TP/SL
# ============================================================
def classify_volatility(v: float) -> str:
    if not np.isfinite(v):
        return "mid"
    if v < CONFIG["VOL_LOW_TH"]:
        return "low"
    if v > CONFIG["VOL_HIGH_TH"]:
        return "high"
    return "mid"

def calc_tp_sl(core: int, market_score: int, vol: float) -> Tuple[float, float]:
    # --- TP ---
    if core < 75:
        tp = 0.06
    elif core < 80:
        tp = 0.08
    elif core < 90:
        tp = 0.10
    else:
        tp = 0.12 + (core - 90) / 10 * 0.03

    if market_score >= 70:
        tp += 0.02
    elif 40 <= market_score < 50:
        tp -= 0.02
    elif market_score < 40:
        tp -= 0.04

    tp = max(CONFIG["TP_MIN"], min(CONFIG["TP_MAX"], tp))

    # --- SL ---
    vc = classify_volatility(vol)
    sl = -0.045
    if vc == "low":
        sl = -0.035
    elif vc == "high":
        sl = -0.055

    if market_score >= 70:
        sl -= 0.005
    elif market_score < 40:
        sl += 0.005

    sl = max(CONFIG["SL_LOWER"], min(CONFIG["SL_UPPER"], sl))
    return tp, sl

# ============================================================
# OUT Signals（利確・一部撤退のシグナル）
# ============================================================
def evaluate_exit_signals(df: pd.DataFrame) -> List[str]:
    sig: List[str] = []
    if df.empty:
        return sig

    last = df.iloc[-1]
    rsi  = _safe_float(last.get("rsi14"))
    turn = _safe_float(last.get("turnover"))
    avg20 = _safe_float(last.get("turnover_avg20"))

    if np.isfinite(rsi) and rsi >= 70:
        sig.append("RSI過熱")

    if len(df) >= 3:
        d = df.tail(3)
        c = (d["close"] < d["ma5"])
        if c.iloc[-2:].all():
            sig.append("5MA割れ連続")

    if np.isfinite(turn) and np.isfinite(avg20) and avg20 > 0:
        if turn < 0.5 * avg20:
            sig.append("出来高急減")

    return sig

# ============================================================
# Leverage Advice
# ============================================================
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
    return 1.0, "守り優先"

def _fmt_yen(v: float) -> str:
    if not np.isfinite(v):
        return "-"
    return f"{int(round(v)):,}円"

def _fmt_pct(p: float) -> str:
    if not np.isfinite(p):
        return "-"
    return f"{p*100:.1f}%"

# ============================================================
# ポジション読み込み
# ============================================================
def load_positions(path: str = "positions.csv") -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("[WARN] positions.csv読み込み失敗:", e)
        return None

    if df.empty:
        return None

    required = {"ticker", "qty", "avg_price"}
    if not required.issubset(df.columns):
        print("[WARN] positions.csv カラム不足（ticker,qty,avg_price が必要）")
        return None

    df["ticker"] = df["ticker"].astype(str)
    df["qty"] = df["qty"].astype(float)
    df["avg_price"] = df["avg_price"].astype(float)
    return df

# ============================================================
# ポジション分析（プロ機関投資家スタイル）
# ============================================================
def analyze_positions(universe: pd.DataFrame, market_score: int) -> List[str]:
    pos_df = load_positions()
    if pos_df is None:
        return ["保有ポジション情報がありません（positions.csv 未設定 / 空）。"]

    # ticker → name, sector の辞書
    name_map = {r["ticker"]: r["name"] for _, r in universe.iterrows()}
    sec_map  = {r["ticker"]: r["sector"] for _, r in universe.iterrows()}

    lines: List[str] = []
    total_pnl = 0.0

    for _, row in pos_df.iterrows():
        ticker = row["ticker"]
        qty    = row["qty"]
        avg_px = row["avg_price"]

        df = fetch_ohlcv(ticker, period="180d")
        if df is None or df.empty:
            lines.append(f"- {ticker} : データ取得失敗（分析不可）")
            continue

        df = add_indicators(df)
        m  = extract_metrics(df)

        price = m["close"]
        if not np.isfinite(price):
            lines.append(f"- {ticker} : 現値取得不可（分析不可）")
            continue

        pnl_pct = (price - avg_px) / avg_px
        pnl_yen = pnl_pct * avg_px * qty
        total_pnl += pnl_yen

        vola   = m["vola20"]
        rsi    = m["rsi14"]
        off    = m["off_high_pct"]
        shadow = m["lower_shadow_ratio"]
        slope  = m["trend_slope20"]

        # --- リスクスコア（0-100）：50を基準に加減点 ---
        risk = 50.0

        # 地合い
        if market_score < 50:
            risk -= (50 - market_score) * 0.6
        else:
            risk += (market_score - 50) * 0.2

        # ボラ
        if np.isfinite(vola):
            if vola > 0.07:
                risk -= 10
            elif vola < 0.02:
                risk += 5

        # 高値圏 or 押し目
        if np.isfinite(off):
            if off > -3:          # ほぼ高値圏
                risk -= 10
            elif -10 <= off <= -3:
                risk -= 3
            elif off <= -15:
                risk += 5

        # RSI
        if np.isfinite(rsi):
            if rsi > 70:
                risk -= 15
            elif 60 < rsi <= 70:
                risk -= 8
            elif 30 <= rsi <= 45:
                risk += 5
            elif rsi < 25:
                risk -= 5

        # トレンド傾き
        if np.isfinite(slope):
            if slope > 0.01:
                risk += 5
            elif slope < 0:
                risk -= 5

        # ヒゲ（買い方目線）
        if np.isfinite(shadow):
            if shadow >= 0.5:
                risk += 5
            elif shadow <= 0.2:
                risk -= 3

        # 含み損・含み益
        if pnl_pct <= -0.05:
            risk -= 10
        elif pnl_pct >= 0.10:
            risk -= 5  # 取りすぎリスク

        # スコアを 0-100 にクリップ
        risk = max(0, min(100, risk))

        # ラベル判定
        if risk >= 65:
            risk_label = "🟢 安全〜ホールド寄り"
        elif risk >= 45:
            risk_label = "🟠 中立ゾーン（利確orホールド）"
        else:
            risk_label = "🔴 警戒ゾーン（縮小・撤退検討）"

        # 方針テキスト
        advice = "継続監視"
        if pnl_pct >= 0.12 and risk < 70:
            advice = "部分利確を検討"
        if pnl_pct >= 0.20:
            advice = "利確優先（利益は正義）"
        if pnl_pct <= -0.07 and risk < 50:
            advice = "損切・縮小を検討"
        if pnl_pct <= -0.10 and risk < 60:
            advice = "ロット見直し必須"

        name = name_map.get(ticker, ticker)
        sector = sec_map.get(ticker, "不明")

        lines.append(
            f"- {ticker} {name} / {sector}  x{int(qty)}株"
        )
        lines.append(
            f"   取得:{_fmt_yen(avg_px)} → 現値:{_fmt_yen(price)}"
            f" / 損益:{_fmt_pct(pnl_pct)}({_fmt_yen(pnl_yen)})"
        )
        lines.append(
            f"   リスク評価: {risk_label}  | 地合い:{market_score}点"
        )

        # テクニカル要約（シンプルに1行）
        tech_bits = []
        if np.isfinite(rsi):
            tech_bits.append(f"RSI:{rsi:.0f}")
        if np.isfinite(off):
            tech_bits.append(f"高値乖離:{off:.1f}%")
        if np.isfinite(vola):
            tech_bits.append(f"ボラ20:{vola*100:.1f}%")
        if np.isfinite(shadow):
            tech_bits.append(f"下ヒゲ比:{shadow:.2f}")

        if tech_bits:
            lines.append("   " + " / ".join(tech_bits))

        lines.append(f"   方針: {advice}")
        lines.append("")  # 空行で区切り

    # トータル損益
    lines.append(f"合計評価損益: {_fmt_yen(total_pnl)}")
    return lines

# ============================================================
# LINE Message
# ============================================================
def build_line_message(
    date_str: str,
    market_score: int,
    core_list: List[Dict],
    pos_lines: List[str],
) -> str:
    max_lev, lev_label = calc_leverage_advice(market_score)

    lines: List[str] = []
    lines.append(f"📅 {date_str} stockbotTOM 日報\n")

    # ---- 今日の結論 ----
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {market_score}点（{lev_label}）")
    lines.append(f"- レバ目安: 最大 約{max_lev:.1f}倍 / ポジ数目安: 3銘柄前後")

    if market_score >= 70:
        lines.append("- コメント: 押し目狙いは攻め寄り。ただしイベント前のフルベットは避ける。")
    elif market_score >= 50:
        lines.append("- コメント: 通常モード。Core条件を満たす銘柄のみ厳選IN。")
    elif market_score >= 40:
        lines.append("- コメント: やや守り。サイズ控えめ、無理IN禁止。")
    else:
        lines.append("- コメント: 守り優先ゾーン。基本は様子見。")
    lines.append("")

    # ---- Core候補 ----
    lines.append("◆ Core候補（本命押し目）")
    if not core_list:
        lines.append("本命条件なし。今日は無理しない。")
    else:
        for i, r in enumerate(core_list[:10], 1):
            lines.append(
                f"{i}. {r['ticker']} {r['name']} / {r['sector']}  "
                f"Score: {r['score']}"
            )
            comment = []
            if r["score"] >= 90:
                comment.append("総合◎")
            elif r["score"] >= 80:
                comment.append("総合◯")
            if r["trend_score"] >= 15:
                comment.append("トレンド◎")
            elif r["trend_score"] >= 10:
                comment.append("トレンド◯")
            if r["pb_score"] >= 12:
                comment.append("押し目良好")
            if r["liq_score"] >= 12:
                comment.append("流動性◎")

            lines.append("   " + (" / ".join(comment) if comment else "押し目候補"))

            lines.append(
                f"   現値:{_fmt_yen(r['price'])} / "
                f"利確:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['tp_price'])}) / "
                f"損切:{r['sl_pct']*100:.1f}%({_fmt_yen(r['sl_price'])})"
            )

            if r["exit_signals"]:
                lines.append(f"   OUT: {' / '.join(r['exit_signals'])}")
        lines.append("")

    # ---- ポジション分析 ----
    lines.append("◆ ポジション分析")
    for ln in pos_lines:
        lines.append(ln)

    return "\n".join(lines)

# ============================================================
# Screening
# ============================================================
def screen_all() -> str:
    today = jst_today()
    ds = today.strftime("%Y-%m-%d")

    market_score = calc_market_score()

    try:
        universe = load_universe()
    except Exception as e:
        return f"📅 {ds}\n\nユニバース読み込みエラー: {e}"

    core_list: List[Dict] = []

    for _, rw in universe.iterrows():
        t   = rw["ticker"]
        name = rw["name"]
        sec  = rw["sector"]

        df = fetch_ohlcv(t)
        if df is None:
            continue
        df = add_indicators(df)
        if len(df) < 60:
            continue

        m = extract_metrics(df)
        price = m["close"]

        if not np.isfinite(price) or price < CONFIG["MIN_PRICE"]:
            continue
        if (not np.isfinite(m["turnover_avg20"])
                or m["turnover_avg20"] < CONFIG["MIN_TURNOVER"]):
            continue

        sec_s = calc_sector_strength(sec)
        core = calc_core_score(m, market_score, sec_s)
        if core < CONFIG["CORE_SCORE_MIN"]:
            continue

        vol = m["vola20"]
        tp, sl = calc_tp_sl(core, market_score, vol)
        tp_price = price * (1 + tp)
        sl_price = price * (1 + sl)

        ex = evaluate_exit_signals(df)

        core_list.append({
            "ticker": t,
            "name": name,
            "sector": sec,
            "score": core,
            "price": price,
            "tp_pct": tp,
            "sl_pct": sl,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "trend_score": calc_trend_score(m),
            "pb_score": calc_pullback_score(m),
            "liq_score": calc_liquidity_score(m),
            "exit_signals": ex,
        })

    core_list.sort(key=lambda x: x["score"], reverse=True)

    # ポジション分析
    pos_lines = analyze_positions(universe, market_score)

    msg = build_line_message(ds, market_score, core_list, pos_lines)
    return msg

# ============================================================
# Send to Worker (LINE)
# ============================================================
def send_to_lineworker(text: str):
    url = os.getenv("WORKER_URL")
    if not url:
        print("[INFO] WORKER_URL 未設定 → printのみ")
        return
    try:
        r = requests.post(url, json={"text": text}, timeout=15)
        print("[Worker]", r.status_code, r.text)
    except Exception as e:
        print("[WARN] Worker送信エラー:", e)

# ============================================================
# Entry
# ============================================================
def main():
    text = screen_all()
    print(text)
    send_to_lineworker(text)

if __name__ == "__main__":
    main()