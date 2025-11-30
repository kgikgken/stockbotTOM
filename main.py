from __future__ import annotations

import os
import math
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============================================================
# CONFIG（後から調整しやすい定数）
# ============================================================
CONFIG: Dict[str, float] = {
    # 抽出フィルタ
    "MIN_PRICE": 300.0,          # 最低株価
    "MIN_TURNOVER": 1e8,         # 最低売買代金（直近20日平均）

    # Coreスコア閾値
    "CORE_A_MIN": 75.0,          # Aランク（本命）
    "CORE_B_MIN": 68.0,          # Bランク（準本命・枚数控えめ）

    # ボラ分類しきい値
    "VOL_LOW_TH": 0.02,          # 20日ボラ 2% 未満 → low
    "VOL_HIGH_TH": 0.06,         # 20日ボラ 6% 超   → high

    # 利確幅の下限/上限（%）
    "TP_MIN": 0.06,              # +6%
    "TP_MAX": 0.15,              # +15%

    # 損切り幅の上下限（マイナス）
    "SL_UPPER": -0.03,           # -3%（もっともタイト）
    "SL_LOWER": -0.06,           # -6%（もっとも広い）
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

    # フォールバック（ファイル無いときの保険）
    df = pd.DataFrame({
        "ticker": ["8035.T", "6920.T", "4502.T"],
        "name": ["Tokyo Electron", "Lasertec", "Takeda"],
        "sector": ["半導体", "半導体", "医薬"],
    })
    return df


# ============================================================
# OHLCV + インジケータ
# ============================================================
def fetch_ohlcv(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    """yfinance で日足取得（エラーは握りつぶして None）"""
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
        print(f"[WARN] empty ohlcv {ticker}")
        return None

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        print(f"[WARN] missing OHLCV columns {ticker}")
        return None

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    各種テクニカル指標を追加：
      - MA5, MA20, MA50
      - RSI(14)
      - 売買代金 / 20日平均
      - 60日高値からの距離＆日数
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

    # ボラ20
    df["vola20"] = close.pct_change().rolling(20).std() * np.sqrt(20)

    # 20MAの傾き
    df["trend_slope20"] = df["ma20"].pct_change()

    # 下ヒゲ比率
    rng = high - low
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
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


def calc_ret_5d(df: pd.DataFrame) -> float:
    """直近5営業日リターン。取れないときは NaN。"""
    close = df["Close"].astype(float)
    if len(close) <= 6:
        return float("nan")
    try:
        return float(close.iloc[-1] / close.iloc[-6] - 1.0)
    except Exception:
        return float("nan")


# ============================================================
# 地合いスコア（安全版：^TOPX → 1306.T フォールバック）
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
    if s is None and fallback is not None:
        s = safe_download_close(fallback, days)
    if s is None:
        return 0.0
    try:
        return float(s.iloc[-1] / s.iloc[-(days + 1)] - 1.0)
    except Exception:
        return 0.0


def calc_market_score() -> int:
    """
    地合いスコア 0-100。
    ^TOPX が取れないケースに備えて 1306.T をフォールバックに使用。
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
    score += max(-15.0, min(15.0, jp1 * 100))   # 1日 +1% → +1pt
    score += max(-10.0, min(10.0, jp5 * 50))    # 5日 +3% → +7.5pt
    score += max(-10.0, min(10.0, jp20 * 20))   # 20日 +5% → +5pt

    score = max(0.0, min(100.0, score))
    return int(score)


# ============================================================
# セクター強度（本実装）
# ============================================================
def build_sector_strength_map(sector_ret: Dict[str, List[float]]) -> Dict[str, int]:
    """
    セクターごとの「平均5日リターン」を使って 0〜100点にスケーリング。
    - 全セクター平均（中央値）を 50点
    - その ±5% を 20〜80点のレンジとして線形補完
    - それ以上/以下はクリップ
    """
    # 有効データ抽出
    sector_avg: Dict[str, float] = {}
    for s, rets in sector_ret.items():
        vals = [r for r in rets if np.isfinite(r)]
        if not vals:
            continue
        sector_avg[s] = float(np.nanmean(vals))

    if not sector_avg:
        return {}

    values = list(sector_avg.values())
    median = float(np.nanmedian(values))

    result: Dict[str, int] = {}
    for s, avg in sector_avg.items():
        diff = avg - median  # 相対的な超過リターン

        if not np.isfinite(diff):
            score = 50.0
        else:
            if diff <= -0.05:      # -5%以上アンダーパフォーム
                score = 20.0
            elif diff >= 0.05:     # +5%以上アウトパフォーム
                score = 80.0
            else:
                # -5%〜+5% → 20〜80 の中で中央値50に寄せる
                score = 50.0 + (diff / 0.05) * 30.0

        score = max(0.0, min(100.0, score))
        result[s] = int(round(score))

    return result


# ============================================================
# Core スコア（100点）
# ============================================================
def calc_trend_score(m: Dict[str, float]) -> int:
    close = m["close"]
    ma20 = m["ma20"]
    ma50 = m["ma50"]
    slope = m["trend_slope20"]
    off = m["off_high_pct"]

    sc = 0.0

    # 20MA の傾き
    if np.isfinite(slope):
        if slope >= 0.01:
            sc += 8.0
        elif slope > 0:
            sc += 4.0 + (slope / 0.01) * 4.0
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
    rsi = m["rsi14"]
    off = m["off_high_pct"]
    days = m["days_since_high60"]
    shadow = m["lower_shadow_ratio"]

    sc = 0.0

    # RSI
    if np.isfinite(rsi):
        if 30 <= rsi <= 45:
            sc += 7.0
        elif 20 <= rsi < 30 or 45 < rsi <= 55:
            sc += 4.0
        else:
            sc += 1.0

    # 下落率
    if np.isfinite(off):
        if -12.0 <= off <= -5.0:
            sc += 6.0
        elif -20.0 <= off < -12.0:
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
    t = m["turnover_avg20"]
    v = m["vola20"]

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

    return int(max(0.0, min(20.0, sc)))


def calc_core_score(m: Dict[str, float], market_score: int, sector_score: int) -> int:
    """
    Coreスコア（100点）
      - 地合い：0〜20
      - セクター：0〜20
      - トレンド：0〜20
      - 押し目：0〜20
      - 流動性：0〜20
    """
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
    利確(tp), 損切り(sl) を %（0.1=10%）で返す。
    """
    # 利確
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
# OUTシグナル
# ============================================================
def evaluate_exit_signals(df: pd.DataFrame) -> List[str]:
    sig: List[str] = []
    if df.empty:
        return sig

    last = df.iloc[-1]
    rsi = _safe_float(last.get("rsi14"))
    turn = _safe_float(last.get("turnover"))
    avg20 = _safe_float(last.get("turnover_avg20"))

    if np.isfinite(rsi) and rsi >= 70:
        sig.append("RSI過熱")

    if len(df) >= 3:
        d = df.tail(3)
        cond = (d["close"] < d["ma5"])
        if cond.iloc[-2:].all():
            sig.append("5MA割れ連続")

    if np.isfinite(turn) and np.isfinite(avg20) and avg20 > 0:
        if turn < 0.5 * avg20:
            sig.append("出来高急減")

    return sig


# ============================================================
# レバレッジ目安
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
# equity.json & positions.csv 読み込み
# ============================================================
def load_equity(path: str = "equity.json") -> Optional[float]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        eq = data.get("equity", None)
        if eq is None:
            return None
        return float(eq)
    except Exception as e:
        print("[WARN] equity.json 読み込み失敗:", e)
        return None


def load_positions(path: str = "positions.csv") -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("[WARN] positions.csv 読み込み失敗:", e)
        return None

    if "ticker" not in df.columns or "qty" not in df.columns or "avg_price" not in df.columns:
        print("[WARN] positions.csv のカラムが不正（ticker, qty, avg_price 必須）")
        return None

    df["ticker"] = df["ticker"].astype(str)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
    df = df.dropna(subset=["ticker", "qty", "avg_price"])
    return df


def analyze_positions(price_map: Dict[str, float]) -> List[str]:
    """
    positions.csv + 現値(price_map) + equity.json から
    ・推定運用資産
    ・各銘柄の損益、資産比率
    ・総ポジション、レバ
    を計算し、行リストで返す。
    """
    lines: List[str] = []

    pos_df = load_positions()
    if pos_df is None or pos_df.empty:
        lines.append("保有ポジション情報がありません（positions.csv 未設定）。")
        return lines

    equity = load_equity()
    total_pos_value = 0.0

    per_lines: List[str] = []

    for _, row in pos_df.iterrows():
        ticker = str(row["ticker"])
        qty = float(row["qty"])
        avg_price = float(row["avg_price"])

        price = price_map.get(ticker, float("nan"))
        if not np.isfinite(price):
            per_lines.append(f"- {ticker}: データ取得失敗（現値不明）")
            continue

        pos_value = price * qty
        total_pos_value += pos_value

        pnl_pct = (price / avg_price - 1.0) * 100.0
        if equity and equity > 0:
            ratio = pos_value / equity * 100.0
            per_lines.append(
                f"- {ticker}: 現値 {price:.1f} / 取得 {avg_price:.1f} / 損益 {pnl_pct:+.2f}% / 資産比率 {ratio:.1f}%"
            )
        else:
            per_lines.append(
                f"- {ticker}: 現値 {price:.1f} / 取得 {avg_price:.1f} / 損益 {pnl_pct:+.2f}%"
            )

    if equity and equity > 0:
        lev = total_pos_value / equity if equity > 0 else float("nan")
        lines.append(f"推定運用資産: {_fmt_yen(equity)}")
        if total_pos_value > 0:
            lines.append(
                f"推定総ポジション: {_fmt_yen(total_pos_value)}（レバ約 {lev:.2f}倍）"
            )
    else:
        if total_pos_value > 0:
            lines.append(
                f"推定総ポジション: {_fmt_yen(total_pos_value)}（equity.json 不明）"
            )

    if not per_lines:
        lines.append("※ 全銘柄で現値取得に失敗。")
    else:
        lines.extend(per_lines)

    return lines


# ============================================================
# LINE メッセージ構築
# ============================================================
def build_line_message(
    date_str: str,
    market_score: int,
    a_list: List[Dict],
    b_list: List[Dict],
    pos_lines: List[str],
) -> str:
    max_lev, lev_label = calc_leverage_advice(market_score)

    lines: List[str] = []

    # ヘッダー
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
        lines.append("- コメント: やや守り。サイズ控えめ、ルール外IN禁止。")
    else:
        lines.append("- コメント: 守り優先ゾーン。基本は様子見〜縮小。")
    lines.append("")

    # Aランク
    lines.append("◆ Aランク Core候補（本命押し目）")
    if not a_list:
        lines.append("本命条件なし。")
    else:
        for i, r in enumerate(a_list[:10], 1):
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

            if r["sector_score"] >= 70:
                comment.append("セクター追い風")
            elif r["sector_score"] <= 40:
                comment.append("セクター逆風")

            lines.append("   " + (" / ".join(comment) if comment else "押し目候補"))

            lines.append(
                f"   現値:{_fmt_yen(r['price'])} / "
                f"利確:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['tp_price'])}) / "
                f"損切:{r['sl_pct']*100:.1f}%({_fmt_yen(r['sl_price'])})"
            )

            if r["exit_signals"]:
                lines.append(f"   OUT: {' / '.join(r['exit_signals'])}")

    # Bランク
    lines.append("")
    lines.append("◆ Bランク候補（期待値はAより低い / 枚数控えめ推奨）")
    if not b_list:
        lines.append("Bランク候補なし。")
    else:
        for i, r in enumerate(b_list[:5], 1):
            lines.append(f"{i}. {r['ticker']} {r['name']} / {r['sector']}  Score: {r['score']}")
            comment = []
            if r["trend_score"] >= 12:
                comment.append("トレンド◯")
            if r["pb_score"] >= 10:
                comment.append("押し目△〜◯")
            if r["sector_score"] >= 70:
                comment.append("セクター追い風")
            lines.append("   " + (" / ".join(comment) if comment else "押し目候補（精度低め）"))
            lines.append(
                f"   現値:{_fmt_yen(r['price'])} / "
                f"利確:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['tp_price'])}) / "
                f"損切:{r['sl_pct']*100:.1f}%({_fmt_yen(r['sl_price'])})"
            )

    # どちらもない場合のコメント
    if not a_list and not b_list:
        lines.append("")
        lines.append("※ A/Bとも候補なし。スクリーニング上はノーポジ〜縮小推奨。")

    # ポジション分析
    lines.append("")
    lines.append("◆ ポジション分析")
    if not pos_lines:
        lines.append("ポジション情報なし。")
    else:
        lines.extend(pos_lines)

    return "\n".join(lines)


# ============================================================
# Screening 本体
# ============================================================
def screen_all() -> str:
    today = jst_today()
    ds = today.strftime("%Y-%m-%d")

    # 地合い
    market_score = calc_market_score()

    # ユニバース
    try:
        universe = load_universe()
    except Exception as e:
        return f"📅 {ds} stockbotTOM 日報\n\nユニバース読み込みエラー: {e}"

    # -------- 1st pass: データ取得 & インジケータ & 5日リターン --------
    symbol_raw: List[Dict] = []
    sector_ret: Dict[str, List[float]] = {}
    price_map: Dict[str, float] = {}

    for _, row in universe.iterrows():
        ticker = str(row["ticker"])
        name = str(row["name"])
        sector = str(row["sector"])

        df = fetch_ohlcv(ticker)
        if df is None or len(df) < 60:
            continue

        df = add_indicators(df)
        metrics = extract_metrics(df)
        ret5 = calc_ret_5d(df)

        symbol_raw.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "df": df,
                "metrics": metrics,
                "ret5": ret5,
            }
        )

        if np.isfinite(ret5):
            sector_ret.setdefault(sector, []).append(ret5)

        price = metrics.get("close", np.nan)
        if np.isfinite(price):
            price_map[ticker] = float(price)

    # データ全滅ケース
    if not symbol_raw:
        # ポジションだけでも分析する
        pos_lines = analyze_positions(price_map)
        msg_lines = [
            f"📅 {ds} stockbotTOM 日報",
            "",
            "◆ 今日の結論",
            f"- 地合いスコア: {market_score}点",
            "- コメント: 個別データ取得に失敗。今日は無理に攻めない。",
            "",
            "◆ ポジション分析",
        ]
        msg_lines.extend(pos_lines)
        return "\n".join(msg_lines)

    # -------- セクター強度マップ作成 --------
    sector_strength_map = build_sector_strength_map(sector_ret)

    # -------- 2nd pass: Coreスコア & TP/SL 計算 --------
    scored_list: List[Dict] = []

    for item in symbol_raw:
        ticker = item["ticker"]
        name = item["name"]
        sector = item["sector"]
        df = item["df"]
        m = item["metrics"]

        price = m.get("close", np.nan)
        if not np.isfinite(price) or price < CONFIG["MIN_PRICE"]:
            continue

        turnover = m.get("turnover_avg20", np.nan)
        if not np.isfinite(turnover) or turnover < CONFIG["MIN_TURNOVER"]:
            continue

        sector_score = sector_strength_map.get(sector, 50)
        core = calc_core_score(m, market_score, sector_score)

        vol = m.get("vola20", np.nan)
        tp, sl = calc_tp_sl(core, market_score, vol)
        tp_price = price * (1.0 + tp)
        sl_price = price * (1.0 + sl)

        exit_signals = evaluate_exit_signals(df)

        scored_list.append(
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
                "sector_score": sector_score,
                "exit_signals": exit_signals,
            }
        )

    # ポジション分析（price_map を使う）
    pos_lines = analyze_positions(price_map)

    # 候補ゼロ → 地合い＋ポジションだけ返す
    if not scored_list:
        max_lev, lev_label = calc_leverage_advice(market_score)
        msg_lines = [
            f"📅 {ds} stockbotTOM 日報",
            "",
            "◆ 今日の結論",
            f"- 地合いスコア: {market_score}点（{lev_label}）",
            f"- レバ目安: 最大 約{max_lev:.1f}倍",
            "- コメント: Core候補なし。今日は無理に攻めない。",
            "",
            "◆ ポジション分析",
        ]
        msg_lines.extend(pos_lines)
        return "\n".join(msg_lines)

    # A/B ランク分け
    scored_list.sort(key=lambda x: x["score"], reverse=True)
    a_list = [r for r in scored_list if r["score"] >= CONFIG["CORE_A_MIN"]]
    b_list = [r for r in scored_list if CONFIG["CORE_B_MIN"] <= r["score"] < CONFIG["CORE_A_MIN"]]

    # メッセージ構築
    msg = build_line_message(ds, market_score, a_list, b_list, pos_lines)
    return msg


# ============================================================
# Worker へ送信（→ LINE）
# ============================================================
def send_to_lineworker(text: str) -> None:
    url = os.getenv("WORKER_URL")
    if not url:
        print("[INFO] WORKER_URL 未設定 → コンソール出力のみ")
        return

    try:
        r = requests.post(url, json={"text": text}, timeout=15)
        print("[Worker]", r.status_code, r.text)
    except Exception as e:
        print("[WARN] Worker送信エラー:", e)


# ============================================================
# Entry Point
# ============================================================
def main() -> None:
    text = screen_all()
    print(text)
    send_to_lineworker(text)


if __name__ == "__main__":
    main()