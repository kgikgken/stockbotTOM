 from __future__ import annotations

"""
stockbotTOM/main.py

日本株スイングトレード用 朝イチスクリーニング & 戦略通知ボット（完全版）

機能:
- universe_jpx.csv を読み込み（全銘柄ユニバース想定）
- yfinance で日足 OHLCV を取得
- テクニカル指標を計算
- Coreスコア (0-100) を算出
    * 地合い (0-20)
    * セクター強度 (0-20, 今は簡易50固定だが後で本実装しやすい構造)
    * トレンド (0-20)
    * 押し目の質 (0-20)
    * 流動性・安定度 (0-20)
- Coreスコアから Aランク / Bランク に分類
    * A: 本命押し目（ロット大きめ前提）
    * B: 期待値はあるがロット控えめ
- 地合いスコアからレバ目安とコメント生成
- セクターTOP3（5日騰落率平均）を表示
- positions.csv からポジション分析
    * 現値、取得単価、損益率、ポジション金額
    * data/equity.json（LINE報告で更新される資産）から推定運用資産を読み込み
    * 推定レバレッジ（総ポジション / 資産）
- Cloudflare Worker 経由で LINE にテキスト送信

使い方:
- GitHub Actions から `python stockbotTOM/main.py` を実行
- 環境変数 WORKER_URL に Cloudflare Worker の URL を設定
"""

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
# CONFIG（後から調整しやすい定数まとめ）
# ============================================================
CONFIG: Dict[str, float] = {
    # 抽出フィルタ
    "MIN_PRICE": 300.0,        # 最低株価
    "MIN_TURNOVER": 1e8,       # 最低売買代金（直近20日平均）

    # Coreスコア閾値
    "CORE_A_MIN": 80.0,        # Aランク（本命押し目）
    "CORE_B_MIN": 70.0,        # Bランク（押し目候補）

    # ボラティリティ分類
    "VOL_LOW_TH": 0.02,
    "VOL_HIGH_TH": 0.06,

    # 利確幅の下限/上限
    "TP_MIN": 0.06,            # +6%
    "TP_MAX": 0.15,            # +15%

    # 損切り幅の下限/上限（マイナス）
    "SL_UPPER": -0.03,         # -3%（一番タイト）
    "SL_LOWER": -0.06,         # -6%（一番広い）
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


def _fmt_yen(v: float) -> str:
    if not np.isfinite(v):
        return "-"
    return f"{int(round(v)):,}円"


# ============================================================
# Universe
# ============================================================
def load_universe(path: str = "universe_jpx.csv") -> pd.DataFrame:
    """
    universe_jpx.csv を読み込む。
    必須: ticker
    任意: name, sector
    それ以外のカラム（industry_big, market など）は無視してOK。
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            raise ValueError("universe_jpx.csv に ticker カラムがありません。")

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

    # フォールバック（CSVが無いとき用）
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
    yfinance から日足OHLCVを取得（失敗したら None）。
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
        print(f"[WARN] fetch_ohlcv failed {ticker}: {e}")
        return None

    if df is None or df.empty:
        print(f"[WARN] empty ohlcv {ticker}")
        return None

    need = {"Open", "High", "Low", "Close", "Volume"}
    if not need.issubset(df.columns):
        print(f"[WARN] missing OHLCV columns {ticker}")
        return None

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    チャート用インジケータを全部ここで作る。
    - MA5 / MA20 / MA50
    - RSI14
    - 売買代金 & 20日平均
    - ボラティリティ20
    - 60日高値からの距離 & 経過日数
    - 20MAの傾き
    - 下ヒゲ比率
    - 5日, 20日リターン（セクター強度用）
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

    # 売買代金
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    # ボラティリティ20（日次リターンのstd×√20）
    df["vola20"] = close.pct_change().rolling(20).std() * np.sqrt(20)

    # 60日高値からの距離 & 日数
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100
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
    df["lower_shadow_ratio"] = np.where(rng > 0, lower_shadow / rng, 0.0)

    # セクター強度用の 5日 / 20日リターン
    df["ret5"] = close / close.shift(5) - 1
    df["ret20"] = close / close.shift(20) - 1

    return df


def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    out = {}
    for k in [
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
        "ret5",
        "ret20",
    ]:
        out[k] = _safe_float(last.get(k, np.nan))
    return out


# ============================================================
# 地合いスコア（安全版）
# ============================================================
def safe_download_close(ticker: str, days: int) -> Optional[pd.Series]:
    """
    指数用の安全ダウンロード。失敗 or データ不足なら None。
    """
    try:
        df = yf.download(
            ticker,
            period="90d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"[WARN] safe_download_close failed {ticker}: {e}")
        return None

    if df is None or df.empty or "Close" not in df.columns:
        return None
    if len(df) <= days:
        return None
    return df["Close"].astype(float)


def safe_return(ticker: str, days: int, fallback: Optional[str] = None) -> float:
    """
    return = (最新 / X日前) - 1 の安全計算。
    primary → fallback（例: ^TOPX がダメなら 1306.T）→ 0.0
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
    地合いスコア (0-100)。
    - ^TOPX が取れない問題対策として 1306.T を fallback に。
    """
    topix_ret1 = safe_return("^TOPX", 1, fallback="1306.T")
    topix_ret5 = safe_return("^TOPX", 5, fallback="1306.T")
    topix_ret20 = safe_return("^TOPX", 20, fallback="1306.T")

    nikkei_ret1 = safe_return("^N225", 1)
    nikkei_ret5 = safe_return("^N225", 5)

    jp1 = (topix_ret1 + nikkei_ret1) / 2
    jp5 = (topix_ret5 + nikkei_ret5) / 2
    jp20 = topix_ret20

    score = 50.0
    # 1日分
    score += max(-15.0, min(15.0, jp1 * 100))
    # 5日分
    score += max(-10.0, min(10.0, jp5 * 50))
    # 20日分
    score += max(-10.0, min(10.0, jp20 * 20))

    score = max(0.0, min(100.0, score))
    return int(round(score))


# ============================================================
# セクター強度（簡易スコア + TOP3用）
# ============================================================
def calc_sector_strength(sector: str) -> int:
    """
    今は「全セクター50点固定」。
    後でセクターインデックス or universe_jpx から本実装に差し替え予定。
    """
    return 50


def build_sector_top3(sector_ret_map: Dict[str, List[float]]) -> List[Tuple[str, float]]:
    """
    セクター毎の 5日リターン平均から TOP3 を返す。
    """
    avg_map: Dict[str, float] = {}
    for sec, rets in sector_ret_map.items():
        vals = [r for r in rets if np.isfinite(r)]
        if not vals:
            continue
        avg_map[sec] = float(np.mean(vals))

    # 大きい順に並べて上位3件
    items = sorted(avg_map.items(), key=lambda x: x[1], reverse=True)
    return items[:3]


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

    # 20MAの傾き
    if np.isfinite(slope):
        if slope >= 0.01:
            sc += 8.0
        elif slope > 0:
            sc += 4.0 + (slope / 0.01) * 4.0
        else:
            sc += max(0.0, 4.0 + slope * 50.0)

    # MAの並び
    if np.isfinite(close) and np.isfinite(ma20) and np.isfinite(ma50):
        if close > ma20 and ma20 > ma50:
            sc += 8.0
        elif close > ma20:
            sc += 4.0
        elif ma20 > ma50:
            sc += 2.0

    # 高値からの距離
    if np.isfinite(off):
        if off >= -5:
            sc += 4.0
        elif off >= -15:
            sc += 4.0 - abs(off + 5.0) * 0.2

    return int(max(0, min(20, round(sc))))


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

    return int(max(0, min(20, round(sc))))


def calc_liquidity_score(m: Dict[str, float]) -> int:
    t = m["turnover_avg20"]
    v = m["vola20"]
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
    """
    利確幅 (tp: +0.1 = +10%), 損切り幅 (sl: -0.04 = -4%) を返す。
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

    if np.isfinite(rsi) and rsi >= 70:
        sig.append("RSI過熱")

    if len(df) >= 3:
        d = df.tail(3)
        cond = d["close"] < d["ma5"]
        if cond.iloc[-2:].all():
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


# ============================================================
# ポジション分析（positions.csv + data/equity.json）
# ============================================================
@dataclass
class PositionRow:
    ticker: str
    qty: float
    avg_price: float
    last_price: Optional[float]  # Noneなら取得失敗
    pnl_pct: Optional[float]     # Noneなら算出不可
    value: Optional[float]       # ポジション金額


def load_equity(path: str = "data/equity.json") -> Optional[float]:
    """
    LINE から送っている equity.json を読む。
    形式: {"equity": 3375662}
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        eq = data.get("equity")
        if eq is None:
            return None
        return float(eq)
    except Exception as e:
        print("[WARN] equity.json load error:", e)
        return None


def fetch_last_close(ticker: str) -> Optional[float]:
    """
    個別銘柄の直近終値を1本だけ取得（失敗したら None）。
    """
    try:
        df = yf.download(
            ticker,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"[WARN] fetch_last_close failed {ticker}: {e}")
        return None

    if df is None or df.empty or "Close" not in df.columns:
        return None

    return float(df["Close"].iloc[-1])


def load_positions(path: str = "positions.csv") -> List[PositionRow]:
    """
    positions.csv:
    ticker,qty,avg_price
    4971.T,400,5120
    """
    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("[WARN] positions.csv read error:", e)
        return []

    if "ticker" not in df.columns or "qty" not in df.columns or "avg_price" not in df.columns:
        print("[WARN] positions.csv missing columns (need ticker, qty, avg_price)")
        return []

    rows: List[PositionRow] = []
    for _, r in df.iterrows():
        ticker = str(r["ticker"])
        qty = _safe_float(r["qty"], default=np.nan)
        avg_price = _safe_float(r["avg_price"], default=np.nan)
        if not np.isfinite(qty) or not np.isfinite(avg_price):
            continue

        last_price = fetch_last_close(ticker)
        if last_price is None:
            rows.append(
                PositionRow(
                    ticker=ticker,
                    qty=qty,
                    avg_price=avg_price,
                    last_price=None,
                    pnl_pct=None,
                    value=None,
                )
            )
        else:
            value = last_price * qty
            pnl_pct = (last_price / avg_price - 1.0) * 100.0
            rows.append(
                PositionRow(
                    ticker=ticker,
                    qty=qty,
                    avg_price=avg_price,
                    last_price=last_price,
                    pnl_pct=pnl_pct,
                    value=value,
                )
            )

    return rows


def build_position_section() -> Tuple[List[str], Optional[float], float]:
    """
    ポジション分析のテキスト行 + equity + total_pos を返す。
    """
    lines: List[str] = []

    pos_rows = load_positions()
    if not pos_rows:
        lines.append("◆ ポジション分析")
        lines.append("ポジション情報がありません（positions.csv 未設定）。")
        return lines, None, 0.0

    total_pos = sum(r.value for r in pos_rows if r.value is not None) or 0.0
    equity = load_equity()

    lines.append("◆ ポジション分析")
    if equity is not None and equity > 0:
        lev = total_pos / equity
        lines.append(f"推定運用資産: {_fmt_yen(equity)}")
        lines.append(f"推定ポジション総額: {_fmt_yen(total_pos)}（レバ約 {lev:.2f}倍）")
    else:
        lines.append(f"推定ポジション総額: {_fmt_yen(total_pos)}（資産情報なし）")

    for r in pos_rows:
        if r.last_price is None:
            lines.append(f"- {r.ticker}: データ取得失敗（現値不明）")
        else:
            pnl_str = f"{r.pnl_pct:+.2f}%" if r.pnl_pct is not None else "N/A"
            lines.append(
                f"- {r.ticker}: 現値 {r.last_price:.1f} / 取得 {r.avg_price:.1f} / 損益 {pnl_str}"
            )

    return lines, equity, total_pos


# ============================================================
# LINEメッセージ生成
# ============================================================
def build_line_message(
    date_str: str,
    market_score: int,
    core_A: List[Dict],
    core_B: List[Dict],
    sector_top3: List[Tuple[str, float]],
    pos_lines: List[str],
) -> str:
    lines: List[str] = []

    max_lev, lev_label = calc_leverage_advice(market_score)

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
        lines.append("- コメント: やや守り。サイズ控えめ、無理IN禁止。")
    else:
        lines.append("- コメント: 守り優先ゾーン。基本は様子見。")
    lines.append("")

    # セクターTOP3
    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    if not sector_top3:
        lines.append("算出できるセクターデータがありません。")
    else:
        for rank, (sec, val) in enumerate(sector_top3, 1):
            lines.append(f"{rank}位: {sec} （{val*100:+.2f}%）")
    lines.append("")

    # Core A
    lines.append("◆ Core候補 Aランク（本命押し目）")
    if not core_A:
        lines.append("本命Aランク条件なし。")
    else:
        for i, r in enumerate(core_A[:10], 1):
            lines.append(
                f"{i}. {r['ticker']} {r['name']} / {r['sector']}  Score: {r['score']} (A)"
            )
            c_parts: List[str] = []
            if r["score"] >= 90:
                c_parts.append("総合◎")
            elif r["score"] >= 80:
                c_parts.append("総合◯")
            if r["trend_score"] >= 15:
                c_parts.append("トレンド◎")
            elif r["trend_score"] >= 10:
                c_parts.append("トレンド◯")
            if r["pb_score"] >= 12:
                c_parts.append("押し目良好")
            if r["liq_score"] >= 12:
                c_parts.append("流動性◎")
            comment = " / ".join(c_parts) if c_parts else "押し目候補"
            lines.append(f"   {comment}")
            lines.append(
                f"   現値:{_fmt_yen(r['price'])} / "
                f"利確:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['tp_price'])}) / "
                f"損切:{r['sl_pct']*100:.1f}%({_fmt_yen(r['sl_price'])})"
            )
            if r["exit_signals"]:
                lines.append(f"   OUTシグナル: {' / '.join(r['exit_signals'])}")
    lines.append("")

    # Core B
    lines.append("◆ Core候補 Bランク（押し目候補・ロット控えめ推奨）")
    if not core_B:
        lines.append("Bランク候補もなし。今日は無理な新規INは控える。")
    else:
        for i, r in enumerate(core_B[:10], 1):
            lines.append(
                f"{i}. {r['ticker']} {r['name']} / {r['sector']}  Score: {r['score']} (B)"
            )
            c_parts: List[str] = []
            if r["trend_score"] >= 12:
                c_parts.append("トレンド◯")
            if r["pb_score"] >= 10:
                c_parts.append("押し目◯")
            if r["liq_score"] >= 10:
                c_parts.append("流動性◯")
            comment = " / ".join(c_parts) if c_parts else "押し目候補（慎重IN）"
            lines.append(f"   {comment}")
            lines.append(
                f"   現値:{_fmt_yen(r['price'])} / "
                f"利確:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['tp_price'])}) / "
                f"損切:{r['sl_pct']*100:.1f}%({_fmt_yen(r['sl_price'])})"
            )
            if r["exit_signals"]:
                lines.append(f"   OUTシグナル: {' / '.join(r['exit_signals'])}")
    lines.append("")

    # ポジション分析（そのまま結合）
    lines.extend(pos_lines)

    return "\n".join(lines)


# ============================================================
# Screening 全体フロー
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

    core_A: List[Dict] = []
    core_B: List[Dict] = []
    sector_ret5_map: Dict[str, List[float]] = {}

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

        price = m["close"]
        if not np.isfinite(price) or price < CONFIG["MIN_PRICE"]:
            continue

        if not np.isfinite(m["turnover_avg20"]) or m["turnover_avg20"] < CONFIG["MIN_TURNOVER"]:
            continue

        # セクター5日リターン用
        r5 = m.get("ret5", np.nan)
        sector_ret5_map.setdefault(sec, []).append(r5)

        sector_score = calc_sector_strength(sec)
        core = calc_core_score(m, market_score, sector_score)

        if core < CONFIG["CORE_B_MIN"]:
            # Bランク未満は候補外
            continue

        vol = m["vola20"]
        tp, sl = calc_tp_sl(core, market_score, vol)
        tp_price = price * (1.0 + tp)
        sl_price = price * (1.0 + sl)
        ex = evaluate_exit_signals(df)

        row = {
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

        if core >= CONFIG["CORE_A_MIN"]:
            core_A.append(row)
        else:
            core_B.append(row)

    # ソート
    core_A.sort(key=lambda x: x["score"], reverse=True)
    core_B.sort(key=lambda x: x["score"], reverse=True)

    # セクターTOP3
    sector_top3 = build_sector_top3(sector_ret5_map)

    # ポジション分析
    pos_lines, equity, total_pos = build_position_section()

    # Core候補が一切ない場合のシンプル版
    if not core_A and not core_B:
        lines: List[str] = []
        lines.append(f"📅 {ds} stockbotTOM 日報")
        lines.append("")
        lines.append("◆ 今日の結論")
        lines.append(f"- 地合いスコア: {market_score}点")
        if equity is not None:
            lines.append(f"- 推定運用資産: {_fmt_yen(equity)}")
        lines.append("- コメント: 個別の本命候補なし。今日は無理に攻めない。")
        lines.append("")
        # セクターTOP3
        lines.append("◆ 今日のTOPセクター（5日騰落率）")
        if not sector_top3:
            lines.append("算出できるセクターデータがありません。")
        else:
            for rank, (sec, val) in enumerate(sector_top3, 1):
                lines.append(f"{rank}位: {sec} （{val*100:+.2f}%）")
        lines.append("")
        lines.extend(pos_lines)
        return "\n".join(lines)

    # 通常版メッセージ
    msg = build_line_message(ds, market_score, core_A, core_B, sector_top3, pos_lines)
    return msg


# ============================================================
# Send to Worker (LINE)
# ============================================================
def send_to_lineworker(text: str) -> None:
    """
    Cloudflare Worker に POST → Worker が LINE にプッシュ。
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