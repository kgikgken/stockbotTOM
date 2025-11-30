from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============================================================
# CONFIG
# ============================================================
CONFIG: Dict[str, float] = {
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
    # fallback
    df = pd.DataFrame({
        "ticker": ["8035.T", "6920.T", "4502.T"],
        "name": ["Tokyo Electron", "Lasertec", "Takeda"],
        "sector": ["半導体", "半導体", "医薬"],
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
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float)

    df["close"] = close
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 売買代金 & 20日平均
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    # ボラティリティ20
    returns = close.pct_change()
    df["vola20"] = returns.rolling(20).std() * np.sqrt(20)

    # 60日高値からの距離 & 日数
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100.0
        tail = close.tail(60)
        idx = int(np.argmax(tail.values))
        df["days_since_high60"] = (len(tail) - 1) - idx
    else:
        df["off_high_pct"] = np.nan
        df["days_since_high60"] = np.nan

    # 20MAの傾き
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
        "close",
        "ma5",
        "ma20",
        "ma50",
        "rsi14",
        "turnover_avg20",
        "off_high_pct",
        "vola20",
        "trend_slope20",
        "lower_shadow_ratio",
        "days_since_high60",
    ]
    return {k: _safe_float(last.get(k, np.nan)) for k in keys}

# ============================================================
# Market score (safe)
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
    except Exception:
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
    # ^TOPX は落ちるので 1306.T をフォールバック
    topix_ret1 = safe_return("^TOPX", 1, fallback="1306.T")
    topix_ret5 = safe_return("^TOPX", 5, fallback="1306.T")
    topix_ret20 = safe_return("^TOPX", 20, fallback="1306.T")
    nikkei_ret1 = safe_return("^N225", 1)
    nikkei_ret5 = safe_return("^N225", 5)

    jp1 = (topix_ret1 + nikkei_ret1) / 2.0
    jp5 = (topix_ret5 + nikkei_ret5) / 2.0
    jp20 = topix_ret20

    score = 50.0
    score += max(-15.0, min(15.0, jp1 * 100))
    score += max(-10.0, min(10.0, jp5 * 50))
    score += max(-10.0, min(10.0, jp20 * 20))

    score = max(0.0, min(100.0, score))
    return int(round(score))

# ============================================================
# Sector strength (本実装)
# ============================================================
def build_sector_strength_map(universe: pd.DataFrame, price_map: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, int]:
    """
    各セクターの平均リターンから 20〜80 点をつける簡易セクター強度。
    ret_score = 0.2 * 1日 + 0.4 * 5日 + 0.4 * 20日
    """
    sector_ret: Dict[str, List[float]] = {}
    for _, row in universe.iterrows():
        ticker = str(row["ticker"])
        sector = str(row["sector"])
        df = price_map.get(ticker)
        if df is None or df.empty:
            continue
        close = df["Close"].astype(float)
        if len(close) < 21:
            continue
        try:
            r1 = float(close.iloc[-1] / close.iloc[-2] - 1.0)
            r5 = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) > 6 else 0.0
            r20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        except Exception:
            continue
        ret_score = 0.2 * r1 + 0.4 * r5 + 0.4 * r20
        sector_ret.setdefault(sector, []).append(ret_score)

    sector_strength: Dict[str, int] = {}
    if not sector_ret:
        return sector_strength

    sector_avg: Dict[str, float] = {s: float(np.mean(v)) for s, v in sector_ret.items()}
    sectors = list(sector_avg.keys())
    if len(sectors) == 1:
        s = sectors[0]
        sector_strength[s] = 50
        return sector_strength

    # 高リターン順に並べてランクで 20〜80 に割り当て
    sectors_sorted = sorted(sectors, key=lambda s: sector_avg[s], reverse=True)
    n = len(sectors_sorted)
    for rank, s in enumerate(sectors_sorted):
        pos = 1.0 - rank / (n - 1)  # 上位ほど 1 に近い
        strength = 20.0 + pos * 60.0  # 20〜80
        sector_strength[s] = int(round(strength))

    return sector_strength

def calc_sector_strength_simple(sector: str) -> int:
    """ビルド失敗などのフォールバック"""
    return 50

# ============================================================
# Core score
# ============================================================
def calc_trend_score(m: Dict[str, float]) -> int:
    close = m.get("close", np.nan)
    ma20 = m.get("ma20", np.nan)
    ma50 = m.get("ma50", np.nan)
    slope = m.get("trend_slope20", np.nan)
    off = m.get("off_high_pct", np.nan)
    sc = 0.0

    if np.isfinite(slope):
        if slope >= 0.01:
            sc += 8.0
        elif slope > 0:
            sc += 4.0 + slope / 0.01 * 4.0
        else:
            sc += max(0.0, 4.0 + slope * 50.0)

    if np.isfinite(close) and np.isfinite(ma20) and np.isfinite(ma50):
        if close > ma20 and ma20 > ma50:
            sc += 8.0
        elif close > ma20:
            sc += 4.0
        elif ma20 > ma50:
            sc += 2.0

    if np.isfinite(off):
        if off >= -5:
            sc += 4.0
        elif off >= -15:
            sc += 4.0 - abs(off + 5.0) * 0.2

    return int(max(0, min(20, round(sc))))

def calc_pullback_score(m: Dict[str, float]) -> int:
    rsi = m.get("rsi14", np.nan)
    off = m.get("off_high_pct", np.nan)
    days = m.get("days_since_high60", np.nan)
    shadow = m.get("lower_shadow_ratio", np.nan)
    sc = 0.0

    if np.isfinite(rsi):
        if 30 <= rsi <= 45:
            sc += 7.0
        elif 20 <= rsi < 30 or 45 < rsi <= 55:
            sc += 4.0
        else:
            sc += 1.0

    if np.isfinite(off):
        if -12 <= off <= -5:
            sc += 6.0
        elif -20 <= off < -12:
            sc += 3.0
        else:
            sc += 1.0

    if np.isfinite(days):
        if 2 <= days <= 10:
            sc += 4.0
        elif 1 <= days < 2 or 10 < days <= 20:
            sc += 2.0

    if np.isfinite(shadow):
        if shadow >= 0.5:
            sc += 3.0
        elif shadow >= 0.3:
            sc += 1.0

    return int(max(0, min(20, round(sc))))

def calc_liquidity_score(m: Dict[str, float]) -> int:
    t = m.get("turnover_avg20", np.nan)
    v = m.get("vola20", np.nan)
    sc = 0.0

    if np.isfinite(t):
        if t >= 10e8:
            sc += 16.0
        elif t >= 1e8:
            sc += 16.0 * (t - 1e8) / 9e8

    if np.isfinite(v):
        if v < 0.02:
            sc += 4.0
        elif v < 0.06:
            sc += 4.0 * (0.06 - v) / 0.04

    return int(max(0, min(20, round(sc))))

def calc_core_score(m: Dict[str, float], market_score: int, sector_score: int) -> int:
    s_m = max(0.0, min(20.0, market_score * 0.2))
    s_s = max(0.0, min(20.0, sector_score * 0.2))
    s_t = calc_trend_score(m)
    s_p = calc_pullback_score(m)
    s_l = calc_liquidity_score(m)
    total = s_m + s_s + s_t + s_p + s_l
    return int(max(0, min(100, round(total))))

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
    if core < 75:
        tp = 0.06
    elif core < 80:
        tp = 0.08
    elif core < 90:
        tp = 0.10
    else:
        tp = 0.12 + (min(core, 100) - 90) / 10.0 * 0.03

    if market_score >= 70:
        tp += 0.02
    elif 40 <= market_score < 50:
        tp -= 0.02
    elif market_score < 40:
        tp -= 0.04

    tp = max(CONFIG["TP_MIN"], min(CONFIG["TP_MAX"], tp))

    vc = classify_volatility(vol)
    if vc == "low":
        sl = -0.035
    elif vc == "high":
        sl = -0.055
    else:
        sl = -0.045

    if market_score >= 70:
        sl -= 0.005
    elif market_score < 40:
        sl += 0.005

    sl = max(CONFIG["SL_LOWER"], min(CONFIG["SL_UPPER"], sl))
    return tp, sl

# ============================================================
# Exit signals
# ============================================================
def evaluate_exit_signals(df: pd.DataFrame) -> List[str]:
    signals: List[str] = []
    if df.empty:
        return signals

    last = df.iloc[-1]
    rsi = _safe_float(last.get("rsi14"))
    turn = _safe_float(last.get("turnover"))
    avg20 = _safe_float(last.get("turnover_avg20"))

    if np.isfinite(rsi) and rsi >= 70:
        signals.append("RSI過熱")

    if len(df) >= 3 and "close" in df.columns and "ma5" in df.columns:
        sub = df.tail(3)
        cond = sub["close"] < sub["ma5"]
        if cond.iloc[-2:].all():
            signals.append("5MA割れ連続")

    if np.isfinite(turn) and np.isfinite(avg20) and avg20 > 0:
        if turn < 0.5 * avg20:
            signals.append("出来高急減")

    return signals

# ============================================================
# Leverage advice
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
# Equity loader (for total capital)
# ============================================================
def load_equity() -> Optional[float]:
    paths = ["data/equity.json", "equity.json"]
    for p in paths:
        if os.path.exists(p):
            try:
                import json
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                eq = data.get("equity")
                return float(eq) if eq is not None else None
            except Exception as e:
                print(f"[WARN] load_equity failed ({p}): {e}")
                continue
    return None

# ============================================================
# Positions analysis
# ============================================================
def load_positions(path: str = "positions.csv") -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] positions.csv 読み込み失敗: {e}")
        return None
    if "ticker" not in df.columns or "qty" not in df.columns or "avg_price" not in df.columns:
        print("[WARN] positions.csv に必要なカラム (ticker, qty, avg_price) がありません。")
        return None
    df["ticker"] = df["ticker"].astype(str)
    return df[["ticker", "qty", "avg_price"]]

def analyze_positions() -> str:
    df_pos = load_positions()
    if df_pos is None or df_pos.empty:
        return "◆ ポジション分析\n保有ポジション情報がありません（positions.csv 未設定）。"

    equity = load_equity()
    if equity is not None and equity <= 0:
        equity = None

    lines: List[str] = []
    lines.append("◆ ポジション分析")
    if equity is not None:
        lines.append(f"推定運用資産: {_fmt_yen(equity)}")

    total_pos_value = 0.0
    pos_results: List[str] = []

    for _, row in df_pos.iterrows():
        ticker = str(row["ticker"])
        try:
            qty = float(row["qty"])
            avg_price = float(row["avg_price"])
        except Exception:
            continue
        if qty <= 0 or avg_price <= 0:
            continue

        # 現在値取得（fast_info → history の順）
        cur_price = None
        try:
            tk = yf.Ticker(ticker)
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                cur_price = getattr(fi, "last_price", None)
            if cur_price is None:
                hist = tk.history(period="5d", interval="1d")
                if hist is not None and not hist.empty:
                    cur_price = float(hist["Close"].astype(float).iloc[-1])
        except Exception as e:
            print(f"[WARN] position price fetch failed {ticker}: {e}")
            continue

        if cur_price is None:
            pos_results.append(f"- {ticker}: 現値取得失敗")
            continue

        cur_price = float(cur_price)
        pnl_pct = (cur_price / avg_price - 1.0) * 100.0
        pos_value = cur_price * qty
        total_pos_value += pos_value

        weight_str = ""
        if equity is not None and equity > 0:
            w = pos_value / equity * 100.0
            weight_str = f" / 資産比率 {w:.1f}%"

        pos_results.append(
            f"- {ticker}: 現値 {cur_price:.1f} / 取得 {avg_price:.1f} / 損益 {pnl_pct:+.2f}%{weight_str}"
        )

    lines.extend(pos_results)
    if equity is not None and total_pos_value > 0:
        lev = total_pos_value / equity
        lines.append(f"推定総ポジション: {_fmt_yen(total_pos_value)}（レバ約 {lev:.2f}倍）")

    return "\n".join(lines)

# ============================================================
# LINE Message
# ============================================================
def build_line_message(date_str: str, market_score: int, core_list: List[Dict], pos_text: str) -> str:
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

    lines.append("◆ Core候補（本命押し目）")
    if not core_list:
        lines.append("本命条件なし。今日は無理しない。")
    else:
        for i, r in enumerate(core_list[:10], 1):
            lines.append(f"{i}. {r['ticker']} {r['name']} / {r['sector']}  Score: {r['score']}")
            comment_parts: List[str] = []
            if r["score"] >= 90:
                comment_parts.append("総合◎")
            elif r["score"] >= 80:
                comment_parts.append("総合◯")
            if r["trend_score"] >= 15:
                comment_parts.append("トレンド◎")
            elif r["trend_score"] >= 10:
                comment_parts.append("トレンド◯")
            if r["pb_score"] >= 12:
                comment_parts.append("押し目良好")
            if r["liq_score"] >= 12:
                comment_parts.append("流動性◎")
            if not comment_parts:
                comment_parts.append("押し目候補")
            lines.append("   " + " / ".join(comment_parts))

            lines.append(
                "   "
                f"現値:{_fmt_yen(r['price'])} / "
                f"利確:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['tp_price'])}) / "
                f"損切:{r['sl_pct']*100:.1f}%({_fmt_yen(r['sl_price'])})"
            )
            if r["exit_signals"]:
                lines.append(f"   OUT: {' / '.join(r['exit_signals'])}")

    lines.append("")
    lines.append(pos_text)

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
        return f"📅 {ds} stockbotTOM 日報\n\nユニバース読み込みエラー: {e}"

    # 1st pass: 全銘柄の価格だけ取得
    price_map: Dict[str, Optional[pd.DataFrame]] = {}
    for _, row in universe.iterrows():
        ticker = str(row["ticker"])
        df = fetch_ohlcv(ticker)
        price_map[ticker] = df

    # セクター強度マップを構築（失敗したら空 dict → simple fallback）
    sector_strength_map = build_sector_strength_map(universe, price_map)

    core_list: List[Dict] = []

    # 2nd pass: 指標計算とスクリーニング
    for _, row in universe.iterrows():
        ticker = str(row["ticker"])
        name = str(row["name"])
        sector = str(row["sector"])

        df = price_map.get(ticker)
        if df is None:
            continue

        df = add_indicators(df)
        if len(df) < 60:
            continue

        m = extract_metrics(df)
        price = m.get("close", np.nan)
        if not np.isfinite(price) or price < CONFIG["MIN_PRICE"]:
            continue
        if not np.isfinite(m.get("turnover_avg20", np.nan)) or m["turnover_avg20"] < CONFIG["MIN_TURNOVER"]:
            continue

        sector_score = sector_strength_map.get(sector, calc_sector_strength_simple(sector))
        core = calc_core_score(m, market_score, sector_score)
        if core < CONFIG["CORE_SCORE_MIN"]:
            continue

        vol = m.get("vola20", np.nan)
        tp, sl = calc_tp_sl(core, market_score, vol)
        tp_price = price * (1.0 + tp)
        sl_price = price * (1.0 + sl)

        exit_signals = evaluate_exit_signals(df)

        core_list.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "score": core,
                "price": price,
                "tp_pct": tp,
                "sl_pct": sl,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "trend_score": calc_trend_score(m),
                "pb_score": calc_pullback_score(m),
                "liq_score": calc_liquidity_score(m),
                "exit_signals": exit_signals,
            }
        )

    core_list.sort(key=lambda x: x["score"], reverse=True)
    pos_text = analyze_positions()
    msg = build_line_message(ds, market_score, core_list, pos_text)
    return msg

# ============================================================
# Send to Worker (LINE)
# ============================================================
def send_to_lineworker(text: str) -> None:
    url = os.getenv("WORKER_URL")
    if not url:
        print("[INFO] WORKER_URL 未設定 → print のみ")
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