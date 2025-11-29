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

    "CORE_SCORE_MIN": 75.0,   # Coreスコア下限

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
    """
    universe_jpx.csv を読み込む。
    必須: ticker
    任意: name, sector
    """
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

    # fallback（手動ユニバース）
    df = pd.DataFrame(
        {
            "ticker": ["8035.T", "6920.T", "4502.T"],
            "name": ["Tokyo Electron", "Lasertec", "Takeda"],
            "sector": ["半導体", "半導体", "医薬"],
        }
    )
    return df


# ============================================================
# OHLCV + Indicators
# ============================================================
def fetch_ohlcv(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    """
    yfinance から日足を安全に取得。
    失敗したら None。
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
    インジケータ追加：
      - ma5, ma20, ma50
      - rsi14
      - turnover, turnover_avg20
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

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 売買代金
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    # ボラティリティ20
    df["vola20"] = close.pct_change().rolling(20).std() * np.sqrt(20)

    # 60日高値からの乖離率 & 日数
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100.0

        tail = close.tail(60)
        idx = int(np.argmax(tail.values))
        df["days_since_high60"] = (len(tail) - 1) - idx
    else:
        df["off_high_pct"] = np.nan
        df["days_since_high60"] = np.nan

    # 20MA の傾き
    df["trend_slope20"] = df["ma20"].pct_change()

    # 下ヒゲ比率
    rng = high - low
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["lower_shadow_ratio"] = np.where(rng > 0, lower_shadow / rng, 0.0)

    return df


def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """スコア計算で使う最終行指標をまとめる。"""
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
# Market Score（安全版）
# ============================================================
def safe_download_close(ticker: str, days: int) -> Optional[pd.Series]:
    """落ちない安全版ダウンロード。Series or Noneを返す。"""
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
    return = (最新 / X日前) - 1 を安全に計算。
    primaryが取れなかった場合は fallback を試す。
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
    地合いスコア（0〜100）。
    - TOPIX → ^TOPX が取れない場合は 1306.T（TOPIX ETF）に代替
    - 日経平均 → ^N225
    1日・5日・20日のリターンから安全に計算。
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
    # 1日リターン
    score += max(-15.0, min(15.0, jp1 * 100.0))
    # 5日リターン
    score += max(-10.0, min(10.0, jp5 * 50.0))
    # 20日リターン
    score += max(-10.0, min(10.0, jp20 * 20.0))

    score = max(0.0, min(100.0, score))
    return int(score)


# ============================================================
# セクター強度（本実装）
# ============================================================
def _calc_stock_returns_for_sector(df: pd.DataFrame) -> Tuple[float, float]:
    """
    セクター強度用：各銘柄の 5日・20日リターンを計算。
    """
    close = df["close"].astype(float)
    ret5 = np.nan
    ret20 = np.nan
    if len(close) > 6:
        try:
            ret5 = float(close.iloc[-1] / close.iloc[-6] - 1.0)
        except Exception:
            ret5 = np.nan
    if len(close) > 21:
        try:
            ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        except Exception:
            ret20 = np.nan
    return ret5, ret20


def build_sector_strength_map(stock_data: List[Dict]) -> Dict[str, int]:
    """
    ユニバース全体から「セクター強度 (0〜100)」を計算する。

    ロジック：
      - 各銘柄について 5日 & 20日リターンを計算
      - 直近20日平均売買代金で加重平均
      - 0.4 * 5日 + 0.6 * 20日 を「セクターモメンタム」として採用
      - セクター間で min〜max を 20〜80点に線形マッピング
    """
    perf: Dict[str, Dict[str, float]] = {}

    for rec in stock_data:
        sector = rec["sector"]
        ret5 = rec.get("ret5", np.nan)
        ret20 = rec.get("ret20", np.nan)
        metrics = rec.get("metrics", {})
        w = _safe_float(metrics.get("turnover_avg20", np.nan))
        if not np.isfinite(w) or w <= 0:
            w = 1.0

        if not np.isfinite(ret5) and not np.isfinite(ret20):
            continue

        d = perf.setdefault(
            sector,
            {"w": 0.0, "ret5_wsum": 0.0, "ret20_wsum": 0.0},
        )
        d["w"] += w
        if np.isfinite(ret5):
            d["ret5_wsum"] += ret5 * w
        if np.isfinite(ret20):
            d["ret20_wsum"] += ret20 * w

    # セクターごとの生のモメンタムスコア計算
    raw_scores: Dict[str, float] = {}
    for sec, d in perf.items():
        w = d["w"]
        if w <= 0:
            continue
        avg5 = d["ret5_wsum"] / w
        avg20 = d["ret20_wsum"] / w
        # 5日 < 20日 をやや重めに
        combined = 0.4 * avg5 * 100.0 + 0.6 * avg20 * 100.0
        raw_scores[sec] = combined

    if not raw_scores:
        # まともに計算できなかった場合は全部50点
        sectors = {rec["sector"] for rec in stock_data}
        return {s: 50 for s in sectors}

    vals = list(raw_scores.values())
    v_min = min(vals)
    v_max = max(vals)
    strength_map: Dict[str, int] = {}

    if abs(v_max - v_min) < 1e-8:
        # 全部同じ → 全部50
        for sec in raw_scores:
            strength_map[sec] = 50
        return strength_map

    # min〜max を 20〜80 点にマッピング
    for sec, raw in raw_scores.items():
        norm = (raw - v_min) / (v_max - v_min)
        strength = 20.0 + 60.0 * norm
        strength_map[sec] = int(max(0.0, min(100.0, round(strength))))

    # データが無かったセクターは50点
    all_sectors = {rec["sector"] for rec in stock_data}
    for sec in all_sectors:
        if sec not in strength_map:
            strength_map[sec] = 50

    return strength_map


# ============================================================
# Core スコア（100点）
# ============================================================
def calc_trend_score(m: Dict[str, float]) -> int:
    close = m.get("close", np.nan)
    ma20 = m.get("ma20", np.nan)
    ma50 = m.get("ma50", np.nan)
    slope = m.get("trend_slope20", np.nan)
    off = m.get("off_high_pct", np.nan)

    sc = 0.0

    # 20MAの傾き
    if np.isfinite(slope):
        if slope >= 0.01:
            sc += 8.0
        elif slope > 0:
            sc += 4.0 + slope / 0.01 * 4.0
        else:
            sc += max(0.0, 4.0 + slope * 50.0)

    # 価格とMAの関係
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
        if 30.0 <= rsi <= 45.0:
            sc += 7.0
        elif 20.0 <= rsi < 30.0 or 45.0 < rsi <= 55.0:
            sc += 4.0
        else:
            sc += 1.0

    # 高値からの下落率
    if np.isfinite(off):
        if -12.0 <= off <= -5.0:
            sc += 6.0
        elif -20.0 <= off < -12.0:
            sc += 3.0
        else:
            sc += 1.0

    # 日柄
    if np.isfinite(days):
        if 2.0 <= days <= 10.0:
            sc += 4.0
        elif 1.0 <= days < 2.0 or 10.0 < days <= 20.0:
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

    # 売買代金 (最大16点)
    if np.isfinite(t):
        if t >= 10e8:
            sc += 16.0
        elif t >= 1e8:
            sc += 16.0 * (t - 1e8) / 9e8

    # ボラ (最大4点)
    if np.isfinite(v):
        if v < 0.02:
            sc += 4.0
        elif v < 0.06:
            sc += 4.0 * (0.06 - v) / 0.04

    return int(max(0.0, min(20.0, sc)))


def calc_core_score(m: Dict[str, float], market_score: int, sector_score: int) -> int:
    """
    Coreスコア（100点満点）
      地合い 0〜20
      セクター 0〜20
      トレンド 0〜20
      押し目 0〜20
      流動性 0〜20
    """
    s_m = max(0.0, min(20.0, market_score * 0.2))
    s_s = max(0.0, min(20.0, sector_score * 0.2))
    s_t = calc_trend_score(m)
    s_p = calc_pullback_score(m)
    s_l = calc_liquidity_score(m)
    total = s_m + s_s + s_t + s_p + s_l
    return int(max(0.0, min(100.0, round(total))))


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
    利確(TP)・損切り(SL) の％幅を返す。
    例: TP=0.1 → +10%, SL=-0.04 → -4%
    """
    # --- TP ---
    if core < 75:
        tp = 0.06
    elif core < 80:
        tp = 0.08
    elif core < 90:
        tp = 0.10
    else:
        tp = 0.12 + (min(core, 100) - 90) / 10.0 * 0.03

    # 地合い補正
    if market_score >= 70:
        tp += 0.02
    elif 40 <= market_score < 50:
        tp -= 0.02
    elif market_score < 40:
        tp -= 0.04

    tp = max(CONFIG["TP_MIN"], min(CONFIG["TP_MAX"], tp))

    # --- SL ---
    vc = classify_volatility(vol)
    if vc == "low":
        sl = -0.035
    elif vc == "high":
        sl = -0.055
    else:
        sl = -0.045

    if market_score >= 70:
        sl -= 0.005   # ちょい広げる
    elif market_score < 40:
        sl += 0.005   # ちょいタイト

    sl = max(CONFIG["SL_LOWER"], min(CONFIG["SL_UPPER"], sl))

    return tp, sl


# ============================================================
# OUT Signals
# ============================================================
def evaluate_exit_signals(df: pd.DataFrame) -> List[str]:
    sig: List[str] = []
    if df is None or df.empty:
        return sig

    last = df.iloc[-1]
    rsi = _safe_float(last.get("rsi14", np.nan))
    turn = _safe_float(last.get("turnover", np.nan))
    avg20 = _safe_float(last.get("turnover_avg20", np.nan))

    # RSI過熱
    if np.isfinite(rsi) and rsi >= 70.0:
        sig.append("RSI過熱")

    # 5MA割れ連続
    if "close" in df.columns and "ma5" in df.columns and len(df) >= 3:
        sub = df.tail(3)
        cond = sub["close"] < sub["ma5"]
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
# LINE Message
# ============================================================
def build_line_message(date_str: str, market_score: int, core_list: List[Dict]) -> str:
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
        return "\n".join(lines)

    for i, r in enumerate(core_list[:10], 1):
        lines.append(f"{i}. {r['ticker']} {r['name']} / {r['sector']}  Score: {r['score']}")
        # コメント
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
        comment = " / ".join(comment_parts) if comment_parts else "押し目候補"
        lines.append(f"   {comment}")

        # IN/OUT 目安
        lines.append(
            "   "
            f"現値:{_fmt_yen(r['price'])} / "
            f"利確:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['tp_price'])}) / "
            f"損切:{r['sl_pct']*100:.1f}%({_fmt_yen(r['sl_price'])})"
        )

        if r["exit_signals"]:
            lines.append(f"   OUT: {' / '.join(r['exit_signals'])}")

    return "\n".join(lines)


# ============================================================
# Screening
# ============================================================
def screen_all() -> str:
    today = jst_today()
    ds = today.strftime("%Y-%m-%d")

    # 地合いスコア
    market_score = calc_market_score()

    # ユニバース読み込み
    try:
        universe = load_universe()
    except Exception as e:
        return f"📅 {ds} stockbotTOM 日報\n\nユニバース読み込みエラー: {e}"

    # まず全銘柄のデータを集める（セクター強度計算に使う）
    stock_data: List[Dict] = []

    for _, rw in universe.iterrows():
        ticker = str(rw["ticker"])
        name = str(rw["name"])
        sector = str(rw["sector"])

        df = fetch_ohlcv(ticker)
        if df is None:
            continue
        df = add_indicators(df)
        if len(df) < 60:
            continue

        metrics = extract_metrics(df)
        price = metrics.get("close", np.nan)
        if not np.isfinite(price):
            continue

        # セクター強度用リターン
        ret5, ret20 = _calc_stock_returns_for_sector(df)

        stock_data.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "df": df,
                "metrics": metrics,
                "ret5": ret5,
                "ret20": ret20,
            }
        )

    if not stock_data:
        # 何もデータ取れなかった場合
        max_lev, lev_label = calc_leverage_advice(market_score)
        msg_lines = [
            f"📅 {ds} stockbotTOM 日報",
            "",
            "◆ 今日の結論",
            f"- 地合いスコア: {market_score}点（{lev_label}）",
            f"- レバ目安: 最大 約{max_lev:.1f}倍",
            "- コメント: データ取得に失敗 or ユニバース対象外。今日は静観。",
        ]
        return "\n".join(msg_lines)

    # セクター強度マップ（0〜100）
    sector_strength_map = build_sector_strength_map(stock_data)

    # Core候補抽出
    core_list: List[Dict] = []

    for rec in stock_data:
        ticker = rec["ticker"]
        name = rec["name"]
        sector = rec["sector"]
        df = rec["df"]
        metrics = rec["metrics"]

        price = metrics.get("close", np.nan)
        turnover_avg20 = metrics.get("turnover_avg20", np.nan)

        # 株価 & 流動性フィルタ
        if not np.isfinite(price) or price < CONFIG["MIN_PRICE"]:
            continue
        if not np.isfinite(turnover_avg20) or turnover_avg20 < CONFIG["MIN_TURNOVER"]:
            continue

        sector_score = sector_strength_map.get(sector, 50)
        core = calc_core_score(metrics, market_score, sector_score)
        if core < CONFIG["CORE_SCORE_MIN"]:
            continue

        vol = metrics.get("vola20", np.nan)
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
                "trend_score": calc_trend_score(metrics),
                "pb_score": calc_pullback_score(metrics),
                "liq_score": calc_liquidity_score(metrics),
                "exit_signals": exit_signals,
            }
        )

    if not core_list:
        max_lev, lev_label = calc_leverage_advice(market_score)
        msg_lines = [
            f"📅 {ds} stockbotTOM 日報",
            "",
            "◆ 今日の結論",
            f"- 地合いスコア: {market_score}点（{lev_label}）",
            f"- レバ目安: 最大 約{max_lev:.1f}倍",
            "- コメント: Core候補なし。今日は静観。",
        ]
        return "\n".join(msg_lines)

    # Coreスコア順に並べる
    core_list.sort(key=lambda x: x["score"], reverse=True)

    # メッセージ構築
    msg = build_line_message(ds, market_score, core_list)
    return msg


# ============================================================
# Send to Worker (LINE)
# ============================================================
def send_to_lineworker(text: str) -> None:
    """
    Cloudflare Worker 経由で LINE に送信。
    GitHub Actions の Secrets に WORKER_URL を設定しておく前提。
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