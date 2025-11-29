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
# CONFIG（後から調整しやすい定数まとめ）
# ============================================================
CONFIG: Dict[str, float] = {
    # 抽出フィルタ
    "MIN_PRICE": 300.0,        # 最低株価
    "MIN_TURNOVER": 1e8,       # 最低売買代金（直近20日平均）

    # Core候補条件（あなたの希望で 72 に）
    "CORE_SCORE_MIN": 72.0,

    # ボラティリティ区分
    "VOL_LOW_TH": 0.02,
    "VOL_HIGH_TH": 0.06,

    # 利確幅の下限/上限（%）
    "TP_MIN": 0.06,            # +6%
    "TP_MAX": 0.15,            # +15%

    # 損切り幅の下限/上限（マイナス）
    "SL_UPPER": -0.03,         # -3%（タイト）
    "SL_LOWER": -0.06,         # -6%（最も広い）
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
# Universe 読み込み
# ============================================================
def load_universe(path: str = "universe_jpx.csv") -> pd.DataFrame:
    """
    universe_jpx.csv を読み込む。
    必須: ticker
    任意: name, sector
    その他のカラム（industry_big など）は無視してOK。
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            raise ValueError("universe_jpx.csv に 'ticker' カラムがありません。")

        df["ticker"] = df["ticker"].astype(str)

        if "name" in df.columns:
            df["name"] = df["name"].astype(str)
        else:
            df["name"] = df["ticker"]

        if "sector" in df.columns:
            df["sector"] = df["sector"].astype(str)
        else:
            df["sector"] = "その他"

        return df[["ticker", "name", "sector"]]

    # フォールバック（万一CSVが無い場合）
    df = pd.DataFrame({
        "ticker": ["8035.T", "6920.T", "4502.T"],
        "name": ["Tokyo Electron", "Lasertec", "Takeda"],
        "sector": ["半導体", "半導体", "医薬"],
    })
    return df[["ticker", "name", "sector"]]


# ============================================================
# OHLCV + インジケータ
# ============================================================
def fetch_ohlcv(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    """
    yfinance から日足OHLCVを取得（安全版）。
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
    各種インジケータを付与：
      - MA5 / MA20 / MA50
      - RSI14
      - 20日平均売買代金
      - 60日高値からの乖離率 / 日数
      - 20日ボラティリティ
      - 20MA傾き
      - 下ヒゲ比率
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

    # 売買代金 & 20日平均
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    # ボラ20
    df["vola20"] = close.pct_change().rolling(20).std() * np.sqrt(20)

    # 60日高値から乖離 & 日数
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100.0
        tail = close.tail(60)
        idx = int(np.argmax(tail.values))
        df["days_since_high60"] = (len(tail) - 1) - idx
    else:
        df["off_high_pct"] = np.nan
        df["days_since_high60"] = np.nan

    # 20MA傾き
    df["trend_slope20"] = df["ma20"].pct_change()

    # 下ヒゲ比率
    rng = high - low
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["lower_shadow_ratio"] = np.where(rng > 0, lower_shadow / rng, 0.0)

    return df


def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    return {
        "close": _safe_float(last.get("close", np.nan)),
        "ma5": _safe_float(last.get("ma5", np.nan)),
        "ma20": _safe_float(last.get("ma20", np.nan)),
        "ma50": _safe_float(last.get("ma50", np.nan)),
        "rsi14": _safe_float(last.get("rsi14", np.nan)),
        "turnover_avg20": _safe_float(last.get("turnover_avg20", np.nan)),
        "off_high_pct": _safe_float(last.get("off_high_pct", np.nan)),
        "vola20": _safe_float(last.get("vola20", np.nan)),
        "trend_slope20": _safe_float(last.get("trend_slope20", np.nan)),
        "lower_shadow_ratio": _safe_float(last.get("lower_shadow_ratio", np.nan)),
        "days_since_high60": _safe_float(last.get("days_since_high60", np.nan)),
    }


# ============================================================
# 地合いスコア（安全版）
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
    """
    地合いスコア（0〜100）
    - ^TOPX が取れないときは 1306.T（TOPIX ETF）に自動代替。
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
    score += max(-15.0, min(15.0, jp1 * 100))
    score += max(-10.0, min(10.0, jp5 * 50))
    score += max(-10.0, min(10.0, jp20 * 20))

    score = max(0.0, min(100.0, score))
    return int(score)


# ============================================================
# セクター強度（本実装）
# ============================================================
def build_sector_strength_map(symbol_data: List[Dict]) -> Dict[str, int]:
    """
    各銘柄の 20日リターンから セクターごとに平均値を出し、0〜100点にマッピング。
    """
    sector_ret: Dict[str, List[float]] = {}

    for d in symbol_data:
        sec = d["sector"]
        r = d.get("ret20", np.nan)
        if np.isfinite(r):
            sector_ret.setdefault(sec, []).append(r)

    # TOPIX 20日リターン（基準）
    base_ret20 = safe_return("^TOPX", 20, fallback="1306.T")

    sector_strength: Dict[str, int] = {}
    for sec, arr in sector_ret.items():
        if not arr:
            sector_strength[sec] = 50
            continue
        avg_r = float(np.mean(arr))
        rel = avg_r - base_ret20  # 市場比
        # -10%〜+10% を 0〜100 にマッピング（中心50）
        score = 50.0 + rel * 500.0  # 0.02 の差で +10pt
        score = max(0.0, min(100.0, score))
        sector_strength[sec] = int(round(score))

    return sector_strength


# ============================================================
# Core スコア（100点構造）
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

    # 価格と移動平均の関係
    if np.isfinite(close) and np.isfinite(ma20) and np.isfinite(ma50):
        if close > ma20 > ma50:
            sc += 8.0
        elif close > ma20:
            sc += 4.0
        elif ma20 > ma50:
            sc += 2.0

    # 高値からの位置（浅めの押しなら加点）
    if np.isfinite(off):
        if off >= -5:
            sc += 4.0
        elif off >= -15:
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

    return int(max(0.0, min(20.0, sc)))


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

    # ボラティリティ（最大4点）
    if np.isfinite(v):
        if v < 0.02:
            sc += 4.0
        elif v < 0.06:
            sc += 4.0 * (0.06 - v) / 0.04

    return int(max(0.0, min(20.0, sc)))


def calc_core_score(m: Dict[str, float], market_score: int, sector_score: int) -> int:
    s_m = max(0.0, min(20.0, market_score * 0.2))
    s_s = max(0.0, min(20.0, sector_score * 0.2))
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
    利確幅(tp), 損切り幅(sl) を返す（%）
    """
    # 利確幅
    if core < 75:
        tp = 0.06
    elif core < 80:
        tp = 0.08
    elif core < 90:
        tp = 0.10
    else:
        tp = 0.12 + (core - 90) / 10.0 * 0.03

    if market_score >= 70:
        tp += 0.02
    elif 40 <= market_score < 50:
        tp -= 0.02
    elif market_score < 40:
        tp -= 0.04

    tp = max(CONFIG["TP_MIN"], min(CONFIG["TP_MAX"], tp))

    # 損切り幅
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
# OUTシグナル
# ============================================================
def evaluate_exit_signals(df: pd.DataFrame) -> List[str]:
    sig: List[str] = []
    if df.empty:
        return sig

    last = df.iloc[-1]
    rsi = _safe_float(last.get("rsi14", np.nan))
    turn = _safe_float(last.get("turnover", np.nan))
    avg20 = _safe_float(last.get("turnover_avg20", np.nan))

    if np.isfinite(rsi) and rsi >= 70:
        sig.append("RSI過熱")

    if len(df) >= 3:
        d = df.tail(3)
        c = d["close"] < d["ma5"]
        if c.iloc[-2:].all():
            sig.append("5MA割れ連続")

    if np.isfinite(turn) and np.isfinite(avg20) and avg20 > 0:
        if turn < 0.5 * avg20:
            sig.append("出来高急減")

    return sig


# ============================================================
# レバレッジ目安 & イベント
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


def detect_market_events(d: date) -> List[str]:
    """
    重要イベント（FOMC / 日銀 / SQ など）を手動で管理する枠。
    必要になったらこの辞書に日付を足していけばOK。
    """
    event_map: Dict[str, str] = {
        # "2025-12-15": "FOMC",
        # "2025-12-20": "日銀会合",
    }
    key = d.strftime("%Y-%m-%d")
    if key in event_map:
        return [event_map[key]]
    return []


def _fmt_yen(v: float) -> str:
    if not np.isfinite(v):
        return "-"
    return f"{int(round(v)):,}円"


# ============================================================
# ポジション管理（positions.csv）
# ============================================================
def load_positions(path: str = "positions.csv") -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return None

    df["ticker"] = df["ticker"].astype(str)

    # size
    if "size" not in df.columns:
        df["size"] = 0
    df["size"] = df["size"].astype(float)

    # price or avg_price
    price_col = None
    if "price" in df.columns:
        price_col = "price"
    elif "avg_price" in df.columns:
        price_col = "avg_price"
    if price_col is None:
        return None

    df["entry_price"] = df[price_col].astype(float)

    if "entry_date" in df.columns:
        df["entry_date"] = df["entry_date"].astype(str)
    else:
        df["entry_date"] = ""

    if "name" not in df.columns:
        df["name"] = df["ticker"].astype(str)
    else:
        df["name"] = df["name"].astype(str)

    return df[["ticker", "name", "size", "entry_price", "entry_date"]]


def classify_position_decision(
    pnl_pct: float,
    core_score: int,
    price: float,
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
    exit_signals: List[str],
    market_score: int,
) -> str:
    """
    ポジションのざっくり判定ロジック。
    """
    # 理論的 TP / SL
    tp_level = tp_pct * 100.0
    sl_level = sl_pct * 100.0  # マイナス値

    # 損切りゾーンに近い
    if pnl_pct <= sl_level * 0.7:
        return "撤退優先（損切りゾーン接近）"

    # 利確水準を超えている
    if pnl_pct >= tp_level:
        return "利確優先（目標到達）"

    # 利確手前 & 過熱・OUTシグナル
    if pnl_pct >= tp_level * 0.7 and exit_signals:
        return "部分利確〜縮小（シグナル点灯）"

    # Coreスコア高くて含み益ゾーン
    if core_score >= CONFIG["CORE_SCORE_MIN"]:
        if pnl_pct >= 0:
            return "継続でOK（本命押し目継続）"
        else:
            return "押し目継続（許容範囲の含み損）"

    # Coreスコアが弱く、含み損
    if core_score < CONFIG["CORE_SCORE_MIN"] and pnl_pct < 0:
        return "優位性低下。縮小・撤退検討。"

    return "様子見〜縮小寄り"


def analyze_positions(
    pos_df: pd.DataFrame,
    market_score: int,
    sector_strength_map: Dict[str, int],
    symbol_map: Dict[str, Dict],
) -> List[Dict]:
    results: List[Dict] = []

    for _, row in pos_df.iterrows():
        ticker = str(row["ticker"])
        size = float(row["size"])
        entry_price = float(row["entry_price"])
        entry_date = str(row["entry_date"])
        name_from_pos = str(row["name"])

        data = symbol_map.get(ticker)
        if data is None:
            # ユニバース外の保有銘柄 → 個別に取得
            df = fetch_ohlcv(ticker)
            if df is None:
                continue
            df = add_indicators(df)
            if len(df) < 60:
                continue
            metrics = extract_metrics(df)
            sector = "不明"
            name = name_from_pos or ticker
        else:
            df = data["df"]
            metrics = data["metrics"]
            sector = data["sector"]
            name = data["name"]

        price_now = metrics.get("close", np.nan)
        if not np.isfinite(price_now) or entry_price <= 0:
            continue

        pnl_pct = (price_now / entry_price - 1.0) * 100.0
        vol = metrics.get("vola20", np.nan)
        sec_score = sector_strength_map.get(sector, 50)

        core_score = calc_core_score(metrics, market_score, sec_score)
        tp_pct, sl_pct = calc_tp_sl(core_score, market_score, vol)
        tp_price = price_now * (1.0 + tp_pct)
        sl_price = price_now * (1.0 + sl_pct)
        exit_signals = evaluate_exit_signals(df)

        decision = classify_position_decision(
            pnl_pct=pnl_pct,
            core_score=core_score,
            price=price_now,
            entry_price=entry_price,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            exit_signals=exit_signals,
            market_score=market_score,
        )

        results.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "size": size,
                "entry_price": entry_price,
                "entry_date": entry_date,
                "price_now": price_now,
                "pnl_pct": pnl_pct,
                "core_score": core_score,
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
# LINE メッセージ組み立て
# ============================================================
def build_line_message(
    date_str: str,
    market_score: int,
    core_list: List[Dict],
    sector_strength_map: Dict[str, int],
    positions: Optional[List[Dict]],
    events: List[str],
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

    # イベント
    if events:
        lines.append("")
        lines.append("◆ 今日の主なイベント")
        for e in events:
            lines.append(f"- {e}")

    # セクターTOP3
    if sector_strength_map:
        ranked = sorted(sector_strength_map.items(), key=lambda x: x[1], reverse=True)
        lines.append("")
        lines.append("◆ 今日の強いセクターTOP3")
        for i, (sec, sc) in enumerate(ranked[:3], 1):
            lines.append(f"{i}. {sec}（強度 {sc}）")

    # Core候補
    lines.append("")
    lines.append("◆ Core候補（本命押し目）")
    if not core_list:
        lines.append("本命条件なし。今日は無理しない。")
    else:
        for i, r in enumerate(core_list[:10], 1):
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

    # ポジション分析
    lines.append("")
    lines.append("◆ ポジション分析")
    if not positions:
        lines.append("保有ポジション情報がありません（positions.csv 未設定）。")
    else:
        for p in positions:
            lines.append(
                f"{p['ticker']} {p['name']} / {p['sector']}  数量: {int(p['size'])}株"
            )
            lines.append(
                f"   取得:{_fmt_yen(p['entry_price'])} / 現在:{_fmt_yen(p['price_now'])} / 損益:{p['pnl_pct']:.1f}%"
            )
            lines.append(
                f"   理論TP:{_fmt_yen(p['tp_price'])} / 理論SL:{_fmt_yen(p['sl_price'])}"
            )
            if p["exit_signals"]:
                lines.append(f"   シグナル: {' / '.join(p['exit_signals'])}")
            lines.append(f"   判定: {p['decision']}")

    return "\n".join(lines)


# ============================================================
# Screening 全体
# ============================================================
def screen_all() -> str:
    today = jst_today()
    ds = today.strftime("%Y-%m-%d")

    # 地合い
    market_score = calc_market_score()

    # ユニバース読み込み
    try:
        universe = load_universe()
    except Exception as e:
        return f"📅 {ds} stockbotTOM 日報\n\nユニバース読み込みエラー: {e}"

    symbol_data: List[Dict] = []

    # 1st pass: 全銘柄の df / metrics / ret20 を集める
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

        metrics = extract_metrics(df)

        # 20日リターン
        ret20 = np.nan
        try:
            close = df["close"].astype(float)
            if len(close) > 20:
                ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        except Exception:
            ret20 = np.nan

        symbol_data.append(
            {
                "ticker": t,
                "name": name,
                "sector": sec,
                "df": df,
                "metrics": metrics,
                "ret20": ret20,
            }
        )

    # シンボルマップ
    symbol_map: Dict[str, Dict] = {d["ticker"]: d for d in symbol_data}

    # セクター強度マップ
    sector_strength_map = build_sector_strength_map(symbol_data)

    # Core候補抽出
    core_list: List[Dict] = []
    for d in symbol_data:
        t = d["ticker"]
        name = d["name"]
        sec = d["sector"]
        df = d["df"]
        m = d["metrics"]

        price = m.get("close", np.nan)
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
        )

    core_list.sort(key=lambda x: x["score"], reverse=True)

    # ポジション分析
    pos_df = load_positions()
    positions: Optional[List[Dict]]
    if pos_df is None:
        positions = None
    else:
        positions = analyze_positions(pos_df, market_score, sector_strength_map, symbol_map)

    # イベント（今は空枠）
    events = detect_market_events(today)

    msg = build_line_message(
        date_str=ds,
        market_score=market_score,
        core_list=core_list,
        sector_strength_map=sector_strength_map,
        positions=positions,
        events=events,
    )
    return msg


# ============================================================
# Send to Worker (LINE)
# ============================================================
def send_to_lineworker(text: str):
    """
    GitHub Actions 環境変数 WORKER_URL に
    Cloudflare Worker の URL が入っている想定。
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
# Entry Point
# ============================================================
def main():
    text = screen_all()
    print(text)
    send_to_lineworker(text)


if __name__ == "__main__":
    main()