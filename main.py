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
    "MIN_PRICE": 300.0,        # 株価フィルタ
    "MIN_TURNOVER": 1e8,       # 直近20日平均売買代金フィルタ（1億）

    "CORE_SCORE_MIN": 75.0,    # Coreスコア下限

    "VOL_LOW_TH": 0.02,
    "VOL_HIGH_TH": 0.06,

    "TP_MIN": 0.06,            # 利確幅最小  +6%
    "TP_MAX": 0.15,            # 利確幅最大 +15%

    "SL_UPPER": -0.03,         # 損切り最小 -3%
    "SL_LOWER": -0.06,         # 損切り最大 -6%
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
    """
    universe_jpx.csv から
      ticker, name, sector
    を読み込む。無い場合は簡易ユニバース。
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            raise ValueError("universe_jpx.csv に 'ticker' カラムがありません。")
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

    # フォールバック（ファイルが無い場合）
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
    """
    yfinance から日足取得（OHLCV）。
    取れなければ None。
    """
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
    """
    日足に各種インジケータ付与
      - MA5, MA20, MA50
      - RSI14
      - turnover / turnover_avg20
      - vola20
      - off_high_pct, days_since_high60
      - trend_slope20
      - lower_shadow_ratio
    """
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

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # turnover
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    # vola20
    returns = close.pct_change()
    df["vola20"] = returns.rolling(20).std() * np.sqrt(20)

    # 高値からの距離 & 日数
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
    """
    スコア計算用に、最終行から必要な指標だけ取り出す
    """
    last = df.iloc[-1]
    keys = [
        "close", "ma5", "ma20", "ma50", "rsi14",
        "turnover_avg20", "off_high_pct", "vola20",
        "trend_slope20", "lower_shadow_ratio", "days_since_high60",
    ]
    return {k: _safe_float(last.get(k, np.nan)) for k in keys}

# ============================================================
# Market Score（安全版）
# ============================================================
def safe_download_close(ticker: str, days: int) -> Optional[pd.Series]:
    """
    auto_adjust などで落ちないように安全にダウンロード。
    days よりデータが短い場合は None。
    """
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
    """
    (最新 / days日前) - 1 を安全に計算。
    取れない場合は fallback ティッカーを試し、それもダメなら 0.0。
    """
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
    TOPIX (^TOPX) が取れない問題に対応した、安全な地合いスコア (0-100)。
    ^TOPX → 1306.T に自動代替。
    """
    topix_ret1 = safe_return("^TOPX", 1, fallback="1306.T")
    topix_ret5 = safe_return("^TOPX", 5, fallback="1306.T")
    topix_ret20 = safe_return("^TOPX", 20, fallback="1306.T")

    nikkei_ret1 = safe_return("^N225", 1)
    nikkei_ret5 = safe_return("^N225", 5)

    jp1 = (topix_ret1 + nikkei_ret1) / 2.0
    jp5 = (topix_ret5 + nikkei_ret5) / 2.0
    jp20 = topix_ret20

    score = 50.0
    score += max(-15.0, min(15.0, jp1 * 100.0))   # 1日分
    score += max(-10.0, min(10.0, jp5 * 50.0))    # 5日分
    score += max(-10.0, min(10.0, jp20 * 20.0))   # 20日分

    score = max(0.0, min(100.0, score))
    return int(score)

# ============================================================
# セクター強度（本実装）
# ============================================================
def calc_all_sector_strength(universe: pd.DataFrame) -> Dict[str, int]:
    """
    universe_jpx の ticker 群からセクターごとに 5日・20日の平均リターンを出し、
    TOPIX(1306.T) 比で 0〜100 にマッピング。
    """
    sectors = sorted(universe["sector"].dropna().unique())
    result: Dict[str, int] = {}

    topix_ret5 = safe_return("^TOPX", 5, fallback="1306.T")
    topix_ret20 = safe_return("^TOPX", 20, fallback="1306.T")

    for sector in sectors:
        df_sec = universe[universe["sector"] == sector]
        tickers = df_sec["ticker"].tolist()
        if not tickers:
            result[sector] = 50
            continue

        rets5: List[float] = []
        rets20: List[float] = []

        # セクター内が多すぎる時の安全策として先頭10銘柄まで
        for t in tickers[:10]:
            r5 = safe_return(t, 5)
            r20 = safe_return(t, 20)
            if r5 != 0.0:
                rets5.append(r5)
            if r20 != 0.0:
                rets20.append(r20)

        if not rets5 and not rets20:
            result[sector] = 50
            continue

        avg5 = float(np.mean(rets5)) if rets5 else 0.0
        avg20 = float(np.mean(rets20)) if rets20 else 0.0

        rel5 = avg5 - topix_ret5
        rel20 = avg20 - topix_ret20

        # rel5, rel20 を適当に重み付けして ±30pt のレンジにクリップ
        raw = 50.0 + max(-30.0, min(30.0, rel5 * 400.0 + rel20 * 200.0))
        score = int(max(0.0, min(100.0, raw)))
        result[sector] = score

    return result

# ============================================================
# Core スコア
# ============================================================
def calc_trend_score(m: Dict[str, float]) -> int:
    close = m.get("close", np.nan)
    ma20 = m.get("ma20", np.nan)
    ma50 = m.get("ma50", np.nan)
    slope = m.get("trend_slope20", np.nan)
    off = m.get("off_high_pct", np.nan)

    sc = 0.0

    # 20MA の傾き
    if np.isfinite(slope):
        if slope >= 0.01:
            sc += 8.0
        elif slope > 0:
            sc += 4.0 + slope / 0.01 * 4.0
        else:
            sc += max(0.0, 4.0 + slope * 50.0)

    # 価格 > MA20 > MA50
    if np.isfinite(close) and np.isfinite(ma20) and np.isfinite(ma50):
        if close > ma20 and ma20 > ma50:
            sc += 8.0
        elif close > ma20:
            sc += 4.0
        elif ma20 > ma50:
            sc += 2.0

    # 高値からの距離
    if np.isfinite(off):
        if off >= -5.0:
            sc += 4.0
        elif off >= -15.0:
            sc += 4.0 - abs(off + 5.0) * 0.2

    return int(max(0.0, min(20.0, sc)))

def calc_pullback_score(m: Dict[str, float]) -> int:
    rsi = m.get("rsi14", np.nan)
    off = m.get("off_high_pct", np.nan)
    days = m.get("days_since_high60", np.nan)
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

    # 高値からの押し
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

    return int(max(0.0, min(20.0, sc)))

def calc_liquidity_score(m: Dict[str, float]) -> int:
    t = m.get("turnover_avg20", np.nan)
    v = m.get("vola20", np.nan)

    sc = 0.0
    # 売買代金
    if np.isfinite(t):
        if t >= 10e8:
            sc += 16.0
        elif t >= 1e8:
            sc += 16.0 * (t - 1e8) / 9e8

    # ボラ
    if np.isfinite(v):
        if v < 0.02:
            sc += 4.0
        elif v < 0.06:
            sc += 4.0 * (0.06 - v) / 0.04

    return int(max(0.0, min(20.0, sc)))

def calc_core_score(m: Dict[str, float], market_score: int, sector_score: int) -> int:
    s_m = min(20.0, max(0.0, market_score * 0.2))
    s_s = min(20.0, max(0.0, sector_score * 0.2))
    s_t = calc_trend_score(m)
    s_p = calc_pullback_score(m)
    s_l = calc_liquidity_score(m)
    total = s_m + s_s + s_t + s_p + s_l
    return int(max(0.0, min(100.0, total)))

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
    """
    Core + 地合い + ボラ から利確・損切り % を決定。
    """
    # 利確
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

    # 損切り
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
    if len(df) >= 3:
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
# ポジション読み込み & 分析
# ============================================================
def load_positions(path: str = "positions.csv") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    positions.csv を読み込む。
    必須: ticker と 数量 (qty/quantity/shares/size のどれか) と 取得単価 (avg_price/average_price/price/cost_price のどれか)。
    """
    if not os.path.exists(path):
        return None, "保有ポジション情報がありません（positions.csv 未設定）。"

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, f"positions.csv の読み込みに失敗しました: {e}"

    if df.empty:
        return None, "positions.csv に有効な行がありません。"

    if "ticker" not in df.columns:
        return None, "positions.csv に 'ticker' カラムがありません。"

    qty_col = None
    for c in ["qty", "quantity", "shares", "size"]:
        if c in df.columns:
            qty_col = c
            break
    if qty_col is None:
        return None, "positions.csv に数量カラムがありません（qty / quantity / shares / size のいずれかが必要）。"

    price_col = None
    for c in ["avg_price", "average_price", "price", "cost_price"]:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        return None, "positions.csv に取得単価カラムがありません（avg_price / average_price / price / cost_price のいずれかが必要）。"

    df_pos = pd.DataFrame()
    df_pos["ticker"] = df["ticker"].astype(str)
    df_pos["qty"] = df[qty_col].astype(float)
    df_pos["avg_price"] = df[price_col].astype(float)

    if "name" in df.columns:
        df_pos["name"] = df["name"].astype(str)
    else:
        df_pos["name"] = df_pos["ticker"]

    if "sector" in df.columns:
        df_pos["sector"] = df["sector"].astype(str)
    else:
        df_pos["sector"] = "その他"

    df_pos = df_pos[df_pos["qty"] != 0]
    if df_pos.empty:
        return None, "positions.csv に有効な保有数量がありません。"

    return df_pos, None

def analyze_positions(
    df_pos: pd.DataFrame,
    market_score: int,
    sector_strength_map: Dict[str, int],
    ticker_sector_map: Dict[str, str],
    ohlcv_cache: Dict[str, pd.DataFrame],
) -> List[Dict]:
    """
    ポジションごとに:
      - 現在値
      - 損益率・損益額
      - Coreスコア
      - 理想利確・損切り
      - OUTシグナル
      - 持ち越し判定（decision）
    を計算。
    """
    results: List[Dict] = []

    for _, row in df_pos.iterrows():
        ticker = str(row["ticker"])
        qty = float(row["qty"])
        avg_price = float(row["avg_price"])
        name = str(row.get("name", ticker))

        sector = str(row.get("sector", ticker_sector_map.get(ticker, "その他")))
        sector_score = sector_strength_map.get(sector, 50)

        # 既にユニバース走査で取得していれば再利用
        if ticker in ohlcv_cache:
            df = ohlcv_cache[ticker]
        else:
            df = fetch_ohlcv(ticker, period="260d")
            if df is None:
                continue
            df = add_indicators(df)
            ohlcv_cache[ticker] = df

        if df is None or df.empty or len(df) < 60:
            continue

        metrics = extract_metrics(df)
        price = metrics.get("close", np.nan)
        if not np.isfinite(price) or avg_price <= 0.0:
            continue

        core = calc_core_score(metrics, market_score, sector_score)
        vol = metrics.get("vola20", np.nan)
        tp_pct, sl_pct = calc_tp_sl(core, market_score, vol)
        tp_price = price * (1.0 + tp_pct)
        sl_price = price * (1.0 + sl_pct)

        pnl_pct = (price / avg_price - 1.0) * 100.0
        pnl_yen = (price - avg_price) * qty

        exit_signals = evaluate_exit_signals(df)

        # シンプルな持ち越し判定
        if pnl_pct <= -7.0 or "5MA割れ連続" in exit_signals:
            decision = "撤退・縮小検討"
        elif pnl_pct >= 10.0 and "RSI過熱" in exit_signals:
            decision = "利益確定検討"
        elif core >= 80 and not exit_signals:
            decision = "継続◎"
        else:
            decision = "継続◯（サイズ調整検討）"

        results.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "qty": qty,
                "avg_price": avg_price,
                "price": price,
                "pnl_pct": pnl_pct,
                "pnl_yen": pnl_yen,
                "core": core,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "exit_signals": exit_signals,
                "decision": decision,
            }
        )

    return results

# ============================================================
# LINE Message
# ============================================================
def build_line_message(
    date_str: str,
    market_score: int,
    core_list: List[Dict],
    pos_list: Optional[List[Dict]],
    pos_error: Optional[str],
) -> str:
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

    # Core候補
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
            lines.append("   " + (" / ".join(comment_parts) if comment_parts else "押し目候補"))

            lines.append(
                f"   現値:{_fmt_yen(r['price'])} / "
                f"利確:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['tp_price'])}) / "
                f"損切:{r['sl_pct']*100:.1f}%({_fmt_yen(r['sl_price'])})"
            )
            if r["exit_signals"]:
                lines.append(f"   OUT: {' / '.join(r['exit_signals'])}")

    # ポジション分析
    lines.append("")
    lines.append("◆ ポジション分析")
    if pos_error:
        lines.append(pos_error)
    elif not pos_list:
        lines.append("有効なポジションがありません。")
    else:
        for i, p in enumerate(pos_list[:5], 1):
            lines.append(
                f"{i}. {p['ticker']} {p['name']} 数量:{p['qty']}株 "
                f"評価損益:{p['pnl_pct']:.1f}%({_fmt_yen(p['pnl_yen'])})"
            )
            lines.append(
                f"   理想: 利確:+{p['tp_pct']*100:.1f}%({_fmt_yen(p['tp_price'])}) / "
                f"損切:{p['sl_pct']*100:.1f}%({_fmt_yen(p['sl_price'])}) / 判定:{p['decision']}"
            )
            if p["exit_signals"]:
                lines.append(f"   シグナル: {' / '.join(p['exit_signals'])}")

    return "\n".join(lines)

# ============================================================
# Screening
# ============================================================
def screen_all() -> str:
    """
    朝イチスクリーニング + ポジション分析をまとめて実行してテキストを返す
    """
    today = jst_today()
    ds = today.strftime("%Y-%m-%d")

    # 地合い
    market_score = calc_market_score()

    # ユニバース
    try:
        universe = load_universe()
    except Exception as e:
        return f"📅 {ds} stockbotTOM 日報\n\nユニバース読み込みエラー: {e}"

    # セクター強度マップ & ティッカー→セクター
    sector_strength_map = calc_all_sector_strength(universe)
    ticker_sector_map: Dict[str, str] = {
        str(row["ticker"]): str(row["sector"]) for _, row in universe.iterrows()
    }

    core_list: List[Dict] = []
    ohlcv_cache: Dict[str, pd.DataFrame] = {}

    # Core候補スクリーニング
    for _, rw in universe.iterrows():
        ticker = str(rw["ticker"])
        name = str(rw["name"])
        sec = str(rw["sector"])

        df = fetch_ohlcv(ticker)
        if df is None:
            continue
        df = add_indicators(df)
        if len(df) < 60:
            continue

        ohlcv_cache[ticker] = df

        m = extract_metrics(df)
        price = m.get("close", np.nan)

        # フィルタ
        if not np.isfinite(price) or price < CONFIG["MIN_PRICE"]:
            continue
        if not np.isfinite(m.get("turnover_avg20", np.nan)) or m["turnover_avg20"] < CONFIG["MIN_TURNOVER"]:
            continue

        sec_s = sector_strength_map.get(sec, 50)
        core = calc_core_score(m, market_score, sec_s)
        if core < CONFIG["CORE_SCORE_MIN"]:
            continue

        vol = m.get("vola20", np.nan)
        tp, sl = calc_tp_sl(core, market_score, vol)
        tp_price = price * (1.0 + tp)
        sl_price = price * (1.0 + sl)

        ex = evaluate_exit_signals(df)

        core_list.append(
            {
                "ticker": ticker,
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
        )

    core_list.sort(key=lambda x: x["score"], reverse=True)

    # ポジション分析
    df_pos, pos_error = load_positions()
    pos_list: Optional[List[Dict]] = None
    if df_pos is not None:
        try:
            pos_list = analyze_positions(
                df_pos=df_pos,
                market_score=market_score,
                sector_strength_map=sector_strength_map,
                ticker_sector_map=ticker_sector_map,
                ohlcv_cache=ohlcv_cache,
            )
        except Exception as e:
            pos_error = f"ポジション分析中にエラー: {e}"
            pos_list = None

    msg = build_line_message(
        date_str=ds,
        market_score=market_score,
        core_list=core_list,
        pos_list=pos_list,
        pos_error=pos_error,
    )
    return msg

# ============================================================
# Send to Worker (LINE)
# ============================================================
def send_to_lineworker(text: str) -> None:
    """
    Cloudflare Worker 経由で LINE に送信
    """
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