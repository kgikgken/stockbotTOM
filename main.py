from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============================================================
# CONFIG（後から調整しやすい定数まとめ）
# ============================================================
CONFIG: Dict[str, float] = {
    "MIN_PRICE": 300.0,       # 最低株価
    "MIN_TURNOVER": 1e8,      # 最低売買代金（直近20日平均）
    "CORE_SCORE_MAIN": 75.0,  # Aランク閾値
    "CORE_SCORE_ALT": 65.0,   # Bランク閾値（Aランク0件時のみ使用）
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

def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

# ============================================================
# Universe
# ============================================================
def load_universe(path: str = "universe_jpx.csv") -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            raise ValueError("universe_jpx.csv に ticker カラムがありません")
        df["ticker"] = df["ticker"].astype(str)
        if "name" not in df.columns:
            df["name"] = df["ticker"]
        else:
            df["name"] = df["name"].astype(str)
        if "sector" not in df.columns:
            df["sector"] = "その他"
        else:
            df["sector"] = df["sector"].astype(str)
        return df[["ticker", "name", "sector"]]

    # fallback（ファイル無いときだけ）
    df = pd.DataFrame({
        "ticker": ["8035.T", "6920.T", "4502.T"],
        "name": ["Tokyo Electron", "Lasertec", "Takeda"],
        "sector": ["半導体", "半導体", "医薬"]
    })
    return df

# ============================================================
# OHLCV + Indicators
# ============================================================
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
        print(f"[WARN] fetch_ohlcv failed {ticker}: {e}")
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

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 売買代金
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    # ボラ20
    df["vola20"] = close.pct_change().rolling(20).std() * np.sqrt(20)

    # 高値からの乖離 & 日柄
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100
        tail = close.tail(60)
        idx = int(np.argmax(tail.values))
        df["days_since_high60"] = (len(tail) - 1) - idx
    else:
        df["off_high_pct"] = np.nan
        df["days_since_high60"] = np.nan

    # トレンドの傾き（20MA）
    df["trend_slope20"] = df["ma20"].pct_change()

    # 下ヒゲ比率
    rng = high - low
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["lower_shadow_ratio"] = np.where(rng > 0, lower_shadow / rng, 0.0)

    return df

def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    keys = [
        "close", "ma5", "ma20", "ma50",
        "rsi14", "turnover_avg20",
        "off_high_pct", "vola20",
        "trend_slope20", "lower_shadow_ratio",
        "days_since_high60",
    ]
    return {k: _safe_float(last.get(k, np.nan)) for k in keys}

# ============================================================
# Market Score（安全版）
# ============================================================
def safe_download_close(ticker: str, days: int) -> Optional[pd.Series]:
    try:
        df = yf.download(
            ticker,
            period="90d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"[WARN] market download failed {ticker}: {e}")
        return None
    if df is None or df.empty or "Close" not in df.columns:
        return None
    if len(df) <= days:
        return None
    return df["Close"].astype(float)

def safe_return(ticker: str, days: int, fallback: Optional[str] = None) -> float:
    s = safe_download_close(ticker, days)
    if s is None and fallback:
        s = safe_download_close(fallback, days)
    if s is None:
        return 0.0
    try:
        return float(s.iloc[-1] / s.iloc[-(days + 1)] - 1.0)
    except Exception:
        return 0.0

def calc_market_score() -> int:
    """
    地合いスコア（0〜100）
    TOPIX指数(^TOPX)が取れない時は TOPIX ETF (1306.T) にフォールバック。
    """
    topix_ret1  = safe_return("^TOPX", 1, fallback="1306.T")
    topix_ret5  = safe_return("^TOPX", 5, fallback="1306.T")
    topix_ret20 = safe_return("^TOPX", 20, fallback="1306.T")

    nikkei_ret1 = safe_return("^N225", 1)
    nikkei_ret5 = safe_return("^N225", 5)

    jp1  = (topix_ret1 + nikkei_ret1) / 2.0
    jp5  = (topix_ret5 + nikkei_ret5) / 2.0
    jp20 = topix_ret20

    score = 50.0
    score += max(-15.0, min(15.0, jp1 * 100))   # 1日 +1% → +1点
    score += max(-10.0, min(10.0, jp5 * 50))    # 5日 +3% → +7.5点
    score += max(-10.0, min(10.0, jp20 * 20))   # 20日 +5% → +5点

    score = max(0.0, min(100.0, score))
    return int(round(score))

# ============================================================
# セクター強度（簡易: 将来拡張前提）
# ============================================================
def calc_sector_strength(sector: str) -> int:
    # いったん全セクター50点固定
    return 50

# ============================================================
# Core スコア（100点）
# ============================================================
def calc_trend_score(m: Dict[str, float]) -> int:
    close = m.get("close", np.nan)
    ma20  = m.get("ma20", np.nan)
    ma50  = m.get("ma50", np.nan)
    slope = m.get("trend_slope20", np.nan)
    off   = m.get("off_high_pct", np.nan)

    sc = 0.0

    # ① 20MAの傾き（上昇トレンドへの加点）
    if np.isfinite(slope):
        if slope >= 0.01:       # 1日1%上昇ペース → MAX
            sc += 8.0
        elif slope > 0:
            sc += 4.0 + (slope / 0.01) * 4.0
        else:
            sc += max(0.0, 4.0 + slope * 50.0)

    # ② 価格とMAの位置関係
    if np.isfinite(close) and np.isfinite(ma20) and np.isfinite(ma50):
        if close > ma20 > ma50:
            sc += 8.0
        elif close > ma20:
            sc += 4.0
        elif ma20 > ma50:
            sc += 2.0

    # ③ 高値からの位置
    if np.isfinite(off):
        if off >= -5:
            sc += 4.0
        elif off >= -15:
            sc += 4.0 - abs(off + 5.0) * 0.2

    return int(max(0, min(20, sc)))

def calc_pullback_score(m: Dict[str, float]) -> int:
    rsi   = m.get("rsi14", np.nan)
    off   = m.get("off_high_pct", np.nan)
    days  = m.get("days_since_high60", np.nan)
    shadow = m.get("lower_shadow_ratio", np.nan)

    sc = 0.0

    # RSI
    if np.isfinite(rsi):
        if 30 <= rsi <= 45:
            sc += 7.0
        elif 20 <= rsi < 30 or 45 < rsi <= 55:
            sc += 4.0
        else:
            sc += 1.0

    # 高値からの下落率
    if np.isfinite(off):
        if -12 <= off <= -5:
            sc += 6.0
        elif -20 <= off < -12:
            sc += 3.0
        else:
            sc += 1.0

    # 日柄
    if np.isfinite(days):
        if 2 <= days <= 10:
            sc += 4.0
        elif 1 <= days < 2 or 10 < days <= 20:
            sc += 2.0

    # 下ヒゲ
    if np.isfinite(shadow):
        if shadow >= 0.5:
            sc += 3.0
        elif shadow >= 0.3:
            sc += 1.0

    return int(max(0, min(20, sc)))

def calc_liquidity_score(m: Dict[str, float]) -> int:
    t = m.get("turnover_avg20", np.nan)
    v = m.get("vola20", np.nan)
    sc = 0.0

    # 売買代金（最大16点）
    if np.isfinite(t):
        if t >= 10e8:
            sc += 16.0
        elif t >= 1e8:
            sc += 16.0 * (t - 1e8) / 9e8

    # ボラ（最大4点）
    if np.isfinite(v):
        if v < 0.02:
            sc += 4.0
        elif v < 0.06:
            sc += 4.0 * (0.06 - v) / 0.04

    return int(max(0, min(20, sc)))

def calc_core_score(m: Dict[str, float], market_score: int, sector_score: int) -> int:
    s_m = min(20.0, market_score * 0.2)
    s_s = min(20.0, sector_score * 0.2)
    s_t = calc_trend_score(m)
    s_p = calc_pullback_score(m)
    s_l = calc_liquidity_score(m)
    total = s_m + s_s + s_t + s_p + s_l
    return int(min(100, max(0, total)))

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
    # --- 利確幅 ---
    if core < 75:
        tp = 0.06
    elif core < 80:
        tp = 0.08
    elif core < 90:
        tp = 0.10
    else:
        tp = 0.12 + (min(core, 100) - 90) / 10 * 0.03

    if market_score >= 70:
        tp += 0.02
    elif 40 <= market_score < 50:
        tp -= 0.02
    elif market_score < 40:
        tp -= 0.04

    tp = max(CONFIG["TP_MIN"], min(CONFIG["TP_MAX"], tp))

    # --- 損切り幅 ---
    vc = classify_volatility(vol)
    if vc == "low":
        sl = -0.035
    elif vc == "high":
        sl = -0.055
    else:
        sl = -0.045

    if market_score >= 70:
        sl -= 0.005  # 少し広げる
    elif market_score < 40:
        sl += 0.005  # タイトに

    sl = max(CONFIG["SL_LOWER"], min(CONFIG["SL_UPPER"], sl))
    return tp, sl

# ============================================================
# OUT Signals
# ============================================================
def evaluate_exit_signals(df: pd.DataFrame) -> List[str]:
    sig: List[str] = []
    if df.empty:
        return sig

    last = df.iloc[-1]
    rsi = _safe_float(last.get("rsi14"))
    turn = _safe_float(last.get("turnover"))
    avg20 = _safe_float(last.get("turnover_avg20"))

    # RSI過熱
    if np.isfinite(rsi) and rsi >= 70:
        sig.append("RSI過熱")

    # 5MA割れ連続
    if "ma5" in df.columns and "close" in df.columns and len(df) >= 3:
        d = df.tail(3)
        cond = d["close"] < d["ma5"]
        if cond.iloc[-2:].all():
            sig.append("5MA割れ連続")

    # 出来高急減
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

# ============================================================
# ポジション分析
# ============================================================
def load_positions(path: str = "positions.csv") -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] positions.csv 読み込み失敗: {e}")
        return None

    required = {"ticker", "qty", "avg_price"}
    if not required.issubset(df.columns):
        print(f"[WARN] positions.csv 必須カラム不足: {required}")
        return None

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
    return df

def fetch_last_price(ticker: str) -> Optional[float]:
    try:
        df = yf.download(
            ticker,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"[WARN] fetch_last_price failed {ticker}: {e}")
        return None

    if df is None or df.empty or "Close" not in df.columns:
        return None
    try:
        return float(df["Close"].astype(float).iloc[-1])
    except Exception:
        return None

def analyze_positions() -> str:
    df = load_positions()
    if df is None or df.empty:
        return "ポジション情報がありません（positions.csv 未設定）。"

    lines: List[str] = []
    total_value = 0.0

    results = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        qty = int(row["qty"])
        avg_price = _safe_float(row["avg_price"])

        if qty <= 0 or not np.isfinite(avg_price):
            continue

        price = fetch_last_price(ticker)
        if price is None:
            lines.append(f"- {ticker}: データ取得失敗（現値不明）")
            continue

        pos_value = price * qty
        pl_pct = (price / avg_price - 1.0) * 100.0 if avg_price > 0 else np.nan

        total_value += pos_value
        results.append((ticker, qty, avg_price, price, pl_pct, pos_value))

    if not results:
        return "ポジションはありますが、価格データ取得に失敗しました。"

    for ticker, qty, avg_price, price, pl_pct, pos_value in results:
        lines.append(
            f"- {ticker}: 現値 {price:.1f} / 取得 {avg_price:.1f} / 損益 {pl_pct:+.2f}%"
        )

    lines.insert(0, f"推定ポジション総額: {_fmt_yen(total_value)}")
    return "\n".join(lines)

# ============================================================
# LINE Message
# ============================================================
def build_line_message(date_str: str, market_score: int,
                       core_A: List[Dict], core_B: List[Dict],
                       pos_summary: str) -> str:
    max_lev, lev_label = calc_leverage_advice(market_score)

    lines: List[str] = []
    lines.append(f"📅 {date_str} stockbotTOM 日報")
    lines.append("")
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

    # Core A候補
    lines.append("◆ Core候補 Aランク（本命押し目）")
    if not core_A:
        lines.append("本命Aランク条件なし。")
    else:
        for i, r in enumerate(core_A[:10], 1):
            lines.append(f"{i}. {r['ticker']} {r['name']} / {r['sector']}  Score: {r['score']}")
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

    # Core B候補（Aが0件のときだけ表示）
    if not core_A:
        lines.append("◆ Core候補 Bランク（押し目候補・ロット控えめ推奨）")
        if not core_B:
            lines.append("Bランク候補もなし。今日は無理な新規INは控える。")
        else:
            for i, r in enumerate(core_B[:5], 1):
                lines.append(f"{i}. {r['ticker']} {r['name']} / {r['sector']}  Score: {r['score']}")
                lines.append(
                    f"   現値:{_fmt_yen(r['price'])} / "
                    f"利確:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['tp_price'])}) / "
                    f"損切:{r['sl_pct']*100:.1f}%({_fmt_yen(r['sl_price'])})"
                )
    lines.append("")

    # ポジション分析
    lines.append("◆ ポジション分析")
    lines.append(pos_summary)

    return "\n".join(lines)

# ============================================================
# Screening
# ============================================================
def screen_all() -> str:
    today = jst_today()
    ds = today.strftime("%Y-%m-%d")

    market_score = calc_market_score()

    # ポジション分析（常に実施）
    pos_summary = analyze_positions()

    # ユニバース読み込み
    try:
        universe = load_universe()
    except Exception as e:
        return (
            f"📅 {ds} stockbotTOM 日報\n\n"
            f"◆ 今日の結論\n"
            f"- 地合いスコア: {market_score}点\n"
            f"- コメント: ユニバース読み込みエラー: {e}\n\n"
            f"◆ ポジション分析\n{pos_summary}"
        )

    core_A: List[Dict] = []
    core_B: List[Dict] = []

    for _, rw in universe.iterrows():
        t = str(rw["ticker"])
        name = str(rw["name"])
        sec = str(rw["sector"])

        df = fetch_ohlcv(t)
        if df is None:
            continue
        df = add_indicators(df)
        if len(df) < 60:
            continue

        m = extract_metrics(df)
        price = m.get("close", np.nan)

        # フィルタ
        if not np.isfinite(price) or price < CONFIG["MIN_PRICE"]:
            continue
        if not np.isfinite(m.get("turnover_avg20", np.nan)) or m["turnover_avg20"] < CONFIG["MIN_TURNOVER"]:
            continue

        sec_s = calc_sector_strength(sec)
        core = calc_core_score(m, market_score, sec_s)

        vol = m.get("vola20", np.nan)
        tp, sl = calc_tp_sl(core, market_score, vol)
        tp_price = price * (1.0 + tp)
        sl_price = price * (1.0 + sl)
        ex = evaluate_exit_signals(df)

        rec = {
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
        }

        if core >= CONFIG["CORE_SCORE_MAIN"]:
            core_A.append(rec)
        elif core >= CONFIG["CORE_SCORE_ALT"]:
            core_B.append(rec)

    # 並び替え
    core_A.sort(key=lambda x: x["score"], reverse=True)
    core_B.sort(key=lambda x: x["score"], reverse=True)

    # メッセージ生成
    msg = build_line_message(ds, market_score, core_A, core_B, pos_summary)
    return msg

# ============================================================
# Send to Worker (LINE)
# ============================================================
def send_to_lineworker(text: str) -> None:
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
def main() -> None:
    text = screen_all()
    print(text)
    send_to_lineworker(text)

if __name__ == "__main__":
    main()