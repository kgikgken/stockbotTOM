"""
main.py - 日本株スイングトレード用 朝イチスクリーニング & 戦略通知ボット

機能概要:
  - universe_jpx.csv から (ticker, name, sector) を読み込み
  - yfinance で日足OHLCVを取得
  - テクニカル指標を計算
  - Coreスコア(100点) を算出
      地合い 0-20
      セクター 0-20 (現状は固定50点で簡易)
      トレンド 0-20
      押し目の質 0-20
      流動性 & 安定度 0-20
  - Coreスコア / 地合い / ボラティリティ から
      利確 % / 損切り % / レバ目安 を決定
  - Core候補 (score >= 75) を抽出して 1本のテキストに整形し標準出力へ
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


# =================================================
# 0. CONFIG — 調整しやすい定数まとめ
# =================================================

CONFIG: Dict[str, float] = {
    # 抽出フィルタ
    "MIN_PRICE": 300.0,        # 株価下限
    "MIN_TURNOVER": 1e8,       # 直近20日平均売買代金の下限(1億)

    # Core候補条件
    "CORE_SCORE_MIN": 75.0,    # Coreスコアの下限

    # ボラティリティ分類(デフォルトしきい値)
    "VOL_LOW_TH": 0.02,        # 20日ボラ 2% 未満 → low
    "VOL_HIGH_TH": 0.06,       # 20日ボラ 6% 超 → high

    # 利確幅の下限/上限（%）
    "TP_MIN": 0.06,            # +6%
    "TP_MAX": 0.15,            # +15%

    # 損切り幅の下限/上限（%）（マイナス値）
    "SL_UPPER": -0.03,         # -3%（最もタイト）
    "SL_LOWER": -0.06,         # -6%（最も広い）
}


# =================================================
# 1. ユーティリティ
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
# 1-1. ユニバース読み込み
# =================================================

def load_universe(path: str = "universe_jpx.csv") -> pd.DataFrame:
    """
    universe_jpx.csv を読み込む。
    必須カラム: ticker
    任意カラム: name, sector
    無い場合は簡易ユニバースを返す。
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

    # フォールバック
    data = {
        "ticker": ["6920.T", "8035.T", "4502.T", "9984.T", "8316.T", "7203.T"],
    }
    df = pd.DataFrame(data)
    df["name"] = df["ticker"]
    df["sector"] = "その他"
    return df[["ticker", "name", "sector"]]


# =================================================
# 1-2. 日足取得 & インジケータ
# =================================================

def fetch_ohlcv(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    """
    yfinance から日足OHLCVを取得。
    ネットワークエラーなどは None を返してスキップ。
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
    """
    日足 OHLCV DataFrame に各種インジケータを追加：
      - MA5, MA20, MA50
      - RSI14
      - 出来高20日平均比
      - 売買代金 & 20日平均
      - 60日高値からの距離（%）
      - ボラティリティ20（日次リターンstd×√20）
      - 20MA の傾き
      - 下ヒゲ比率
      - 60日高値からの日数
    """
    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    volume = df["Volume"].astype(float)

    df["close"] = close

    # 移動平均
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

    # 出来高20日平均比
    vol20 = volume.rolling(20).mean()
    df["vol_ratio20"] = volume / vol20

    # 売買代金
    df["turnover"] = close * volume
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    # 60日高値からの距離 & 日数
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100.0

        tail = close.tail(60)
        idx_max = int(np.argmax(tail.values))
        days_since_high60 = (len(tail) - 1) - idx_max
    else:
        df["off_high_pct"] = np.nan
        days_since_high60 = np.nan

    df["days_since_high60"] = days_since_high60

    # ボラティリティ20
    returns = close.pct_change()
    df["vola20"] = returns.rolling(20).std() * np.sqrt(20)

    # 20MAの傾き（1日あたり変化率）
    df["trend_slope20"] = df["ma20"].pct_change()

    # 下ヒゲ比率
    range_ = high - low
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["lower_shadow_ratio"] = np.where(range_ > 0, lower_shadow / range_, 0.0)

    return df


def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """スコア計算で使う指標を最終行から抜き出す。"""
    last = df.iloc[-1]
    return {
        "close": _safe_float(last.get("close", np.nan)),
        "ma5": _safe_float(last.get("ma5", np.nan)),
        "ma20": _safe_float(last.get("ma20", np.nan)),
        "ma50": _safe_float(last.get("ma50", np.nan)),
        "rsi14": _safe_float(last.get("rsi14", np.nan)),
        "vol_ratio20": _safe_float(last.get("vol_ratio20", np.nan)),
        "turnover_avg20": _safe_float(last.get("turnover_avg20", np.nan)),
        "off_high_pct": _safe_float(last.get("off_high_pct", np.nan)),
        "vola20": _safe_float(last.get("vola20", np.nan)),
        "trend_slope20": _safe_float(last.get("trend_slope20", np.nan)),
        "lower_shadow_ratio": _safe_float(last.get("lower_shadow_ratio", np.nan)),
        "days_since_high60": _safe_float(last.get("days_since_high60", np.nan)),
    }


# =================================================
# 2. 地合い & セクター強度
# =================================================

def calc_market_score() -> int:
    """
    地合いスコア (0〜100) を計算。
    シンプルに TOPIX (^TOPX) と 日経平均 (^N225) の
    1日変化・5日変化・20日変化を組み合わせる。
    """
    def _ret(ticker: str, days: int) -> float:
        try:
            df = yf.download(ticker, period="60d", interval="1d", progress=False)
            if df is None or len(df) <= days:
                return 0.0
            close = df["Close"].astype(float)
            return float(close.iloc[-1] / close.iloc[-(days + 1)] - 1.0)
        except Exception as e:
            print(f"[WARN] market download failed: {ticker}: {e}")
            return 0.0

    tp_1 = _ret("^TOPX", 1)
    tp_5 = _ret("^TOPX", 5)
    tp_20 = _ret("^TOPX", 20)
    nk_1 = _ret("^N225", 1)
    nk_5 = _ret("^N225", 5)

    score = 50.0

    jp_1 = (tp_1 + nk_1) / 2.0
    jp_5 = (tp_5 + nk_5) / 2.0
    jp_20 = tp_20

    score += max(-15.0, min(15.0, jp_1 * 100))   # 1日 +1% → +1
    score += max(-10.0, min(10.0, jp_5 * 50))    # 5日 +3% → +7.5
    score += max(-10.0, min(10.0, jp_20 * 20))   # 20日 +5% → +5

    score = max(0.0, min(100.0, score))
    return int(round(score))


def calc_sector_strength(sector: str) -> int:
    """
    セクター強度 (0〜100)。
    本実装では「簡易版」として全セクター50点固定。
    将来: セクターインデックスやユニバース平均から計算して差し替え可能。
    """
    return 50


# =================================================
# 3. Coreスコア（100点構造）
# =================================================

def calc_trend_score(metrics: Dict[str, float]) -> int:
    """
    トレンド強度スコア（0〜20）
      - 20MAの傾き（trend_slope20）
      - 価格 > MA20 > MA50
      - 高値からの距離（off_high_pct）
    """
    close = metrics.get("close", np.nan)
    ma20 = metrics.get("ma20", np.nan)
    ma50 = metrics.get("ma50", np.nan)
    slope = metrics.get("trend_slope20", np.nan)
    off_high = metrics.get("off_high_pct", np.nan)

    score = 0.0

    # 20MAの傾き
    if np.isfinite(slope):
        if slope >= 0.01:          # 1%/日 = 非常に強い
            score += 8.0
        elif slope > 0:
            score += 4.0 + (slope / 0.01) * 4.0
        else:
            score += max(0.0, 4.0 + slope * 50.0)  # 軽い減点

    # 価格 > MA20 > MA50
    if np.isfinite(close) and np.isfinite(ma20) and np.isfinite(ma50):
        cond1 = close > ma20
        cond2 = ma20 > ma50
        if cond1 and cond2:
            score += 8.0
        elif cond1:
            score += 4.0
        elif cond2:
            score += 2.0

    # 高値からの距離
    if np.isfinite(off_high):
        if off_high >= -5:
            score += 4.0
        elif off_high >= -15:
            score += 4.0 - abs(off_high + 5.0) * 0.2
        else:
            score += 0.0

    return max(0, min(20, int(round(score))))


def calc_pullback_score(metrics: Dict[str, float]) -> int:
    """
    押し目の質スコア（0〜20）
      - RSI(14)：30〜45が理想
      - 高値からの下落率：-5〜-12% が理想
      - 日柄：2〜10日が理想
      - 下ヒゲ比率：0.3以上が良い
    """
    rsi = metrics.get("rsi14", np.nan)
    off_high = metrics.get("off_high_pct", np.nan)
    days_from_high = metrics.get("days_since_high60", np.nan)
    shadow = metrics.get("lower_shadow_ratio", np.nan)

    score = 0.0

    # RSI
    if np.isfinite(rsi):
        if 30 <= rsi <= 45:
            score += 7.0
        elif 20 <= rsi < 30 or 45 < rsi <= 55:
            score += 4.0
        else:
            score += 1.0

    # 下落率
    if np.isfinite(off_high):
        if -12 <= off_high <= -5:
            score += 6.0
        elif -20 <= off_high < -12:
            score += 3.0
        else:
            score += 1.0

    # 日柄
    if np.isfinite(days_from_high):
        if 2 <= days_from_high <= 10:
            score += 4.0
        elif 1 <= days_from_high < 2 or 10 < days_from_high <= 20:
            score += 2.0

    # 下ヒゲ
    if np.isfinite(shadow):
        if shadow >= 0.5:
            score += 3.0
        elif shadow >= 0.3:
            score += 1.0

    return max(0, min(20, int(round(score))))


def calc_liquidity_score(metrics: Dict[str, float]) -> int:
    """
    流動性 & 安定度スコア（0〜20）
      - turnover_avg20 が高いほど加点
      - vola20 が高すぎると減点
    """
    turnover = metrics.get("turnover_avg20", np.nan)
    vola = metrics.get("vola20", np.nan)

    score = 0.0

    # 売買代金 (最大16点)
    if np.isfinite(turnover):
        if turnover >= 10e8:
            score += 16.0
        elif turnover >= 1e8:
            score += 16.0 * (turnover - 1e8) / (9e8)

    # ボラ (最大4点)
    if np.isfinite(vola):
        if vola < 0.02:
            score += 4.0
        elif vola < 0.06:
            score += 4.0 * (0.06 - vola) / 0.04

    return max(0, min(20, int(round(score))))


def calc_core_score(metrics: Dict[str, float], market_score: int, sector_score: int) -> int:
    """
    Coreスコア（100点満点）
      地合い 0〜20
      セクター 0〜20
      トレンド 0〜20
      押し目 0〜20
      流動性 0〜20
    """
    score_market = max(0.0, min(20.0, market_score * 0.2))
    score_sector = max(0.0, min(20.0, sector_score * 0.2))

    score_trend = calc_trend_score(metrics)
    score_pb = calc_pullback_score(metrics)
    score_liq = calc_liquidity_score(metrics)

    total = score_market + score_sector + score_trend + score_pb + score_liq
    return max(0, min(100, int(round(total))))


# =================================================
# 4. ボラ分類 & 利確/損切り
# =================================================

def classify_volatility(vol: float) -> str:
    """
    ボラティリティ区分を返す: "low" / "mid" / "high"
    実際には全銘柄から分布を計算するのが理想だが、
    ここでは CONFIG のしきい値で簡易分類する。
    """
    if not np.isfinite(vol):
        return "mid"
    if vol < CONFIG["VOL_LOW_TH"]:
        return "low"
    if vol > CONFIG["VOL_HIGH_TH"]:
        return "high"
    return "mid"


def calc_take_profit_and_stop_loss(core_score: int, market_score: int, vol: float) -> Tuple[float, float]:
    """
    利確幅(tp_pct), 損切り幅(sl_pct) を返す（%表記、例: +0.1=+10%, -0.04=-4%）

    1) Coreスコア基準の利確幅
       75〜80 → 8%
       80〜90 → 10%
       90〜100 → 12〜15% で線形
    2) 地合いスコアによる補正
       market_score >=70 → +2%
       50〜70 → 0
       40〜50 → -2%
       <40    → -4% (整理モード)
    3) 最終的に 6〜15%にクリップ
    損切り:
      - vol_low  → -3.5%
      - vol_mid  → -4.5%
      - vol_high → -5.5%
      地合いが悪ければタイトに、良ければ+0.5%緩める。
    """
    # --- 利確幅 ---
    if core_score < 75:
        base_tp = 0.06
    elif core_score < 80:
        base_tp = 0.08
    elif core_score < 90:
        base_tp = 0.10
    else:
        # 90〜100 → 0.12〜0.15 に線形
        base_tp = 0.12 + (min(core_score, 100) - 90) / 10 * 0.03

    # 地合い補正
    if market_score >= 70:
        base_tp += 0.02
    elif 40 <= market_score < 50:
        base_tp -= 0.02
    elif market_score < 40:
        base_tp -= 0.04

    tp_pct = max(CONFIG["TP_MIN"], min(CONFIG["TP_MAX"], base_tp))

    # --- 損切り幅 ---
    vol_class = classify_volatility(vol)
    if vol_class == "low":
        sl = -0.035
    elif vol_class == "high":
        sl = -0.055
    else:
        sl = -0.045

    if market_score >= 70:
        sl -= 0.005   # 少し広げる (例: -4.5 → -5.0)
    elif market_score < 40:
        sl += 0.005   # タイトにする (例: -4.5 → -4.0)

    sl_pct = max(CONFIG["SL_LOWER"], min(CONFIG["SL_UPPER"], sl))

    return tp_pct, sl_pct


# =================================================
# 5. OUTシグナル（簡易版）
# =================================================

def evaluate_exit_signals(df: pd.DataFrame) -> List[str]:
    """
    OUTシグナル（利確・警戒のシグナル）を判定する。
    - RSI過熱: RSI14 >= 70
    - 5日移動平均割れ連続: Close < MA5 が2日以上
    - 出来高急減: 当日売買代金 < 20日平均の50%
    """
    signals: List[str] = []
    if df.empty:
        return signals

    last = df.iloc[-1]
    rsi = _safe_float(last.get("rsi14", np.nan))
    turnover = _safe_float(last.get("turnover", np.nan))
    turnover_avg20 = _safe_float(last.get("turnover_avg20", np.nan))

    # RSI過熱
    if np.isfinite(rsi) and rsi >= 70:
        signals.append("RSI過熱")

    # 5MA割れ連続
    if "ma5" in df.columns and "close" in df.columns and len(df) >= 3:
        sub = df.tail(3)
        cond = sub["close"] < sub["ma5"]
        if cond.iloc[-2:].all():
            signals.append("5MA割れ連続")

    # 出来高急減
    if np.isfinite(turnover) and np.isfinite(turnover_avg20) and turnover_avg20 > 0:
        if turnover < 0.5 * turnover_avg20:
            signals.append("出来高急減")

    return signals


# =================================================
# 6. 決算ウインドウ & イベント（簡易）
# =================================================

def is_in_earnings_window(ticker: str, base_date: date, window: int = 3) -> bool:
    """
    決算日 ±window 日に入っていれば True。
    yfinance の earnings_dates / calendar を使って近傍の日付を推定する。
    取得できなければ False（=除外しない）。
    """
    try:
        tk = yf.Ticker(ticker)
        # 直近のearnings_datesを利用（存在しない銘柄も多い点に注意）
        try:
            ed = tk.get_earnings_dates(limit=4)
            if ed is not None and not ed.empty:
                for idx in ed.index:
                    edate = idx.date() if hasattr(idx, "date") else pd.to_datetime(idx).date()
                    if abs((edate - base_date).days) <= window:
                        return True
        except Exception:
            pass

        # calendar も試す
        cal = getattr(tk, "calendar", None)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for v in cal.iloc[:, 0]:
                try:
                    edate = pd.to_datetime(v).date()
                    if abs((edate - base_date).days) <= window:
                        return True
                except Exception:
                    continue
    except Exception as e:
        print(f"[WARN] earnings check failed for {ticker}: {e}")
        return False

    return False


def detect_market_events(d: date) -> List[str]:
    """
    重要イベント（FOMC / CPI / 雇用統計 / 大型決算など）を判定する簡易関数。
    本コードでは、実用性を優先し「手動で補足する前提」の枠だけ用意しておく。
    必要に応じて日付リストをメンテすればよい。
    """
    # 例: ここに "YYYY-MM-DD": "FOMC" のような辞書を追加していく
    event_map: Dict[str, str] = {
        # "2025-12-15": "FOMC",
        # "2025-12-20": "NVIDIA決算",
    }
    key = d.strftime("%Y-%m-%d")
    if key in event_map:
        return [event_map[key]]
    return []


def calc_leverage_advice(market_score: int) -> Tuple[float, str]:
    """
    地合いスコアから推奨最大レバレッジとラベルを返す。
    """
    if market_score >= 80:
        return 2.5, "攻めMAX"
    elif market_score >= 70:
        return 2.2, "やや攻め"
    elif market_score >= 60:
        return 2.0, "中立〜やや攻め"
    elif market_score >= 50:
        return 1.5, "中立"
    elif market_score >= 40:
        return 1.2, "守り寄り"
    else:
        return 1.0, "守り優先"


# =================================================
# 7. LINEメッセージ生成
# =================================================

def _fmt_yen(v: float) -> str:
    if not np.isfinite(v):
        return "-"
    return f"{int(round(v)):,}円"


def build_line_message(date_str: str, market_score: int, core_list: List[Dict]) -> str:
    """
    LINE通知本文を構築する。
    Core候補はスコア上位10銘柄まで表示。
    """
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
        lines.append("- コメント: やや守り。サイズ控えめに、無理な新規INはしない。")
    else:
        lines.append("- コメント: 守り優先ゾーン。基本は様子見〜縮小。")
    lines.append("")

    # Core候補
    lines.append("◆ Core候補（本命押し目）")
    if not core_list:
        lines.append("本命条件を満たす銘柄なし。今日は無理に攻めない選択もあり。")
        return "\n".join(lines)

    for i, r in enumerate(core_list[:10], 1):
        code = r["ticker"]
        name = r["name"]
        sector = r["sector"]
        score = r["score"]
        price = r["price"]
        tp_pct = r["tp_pct"]
        sl_pct = r["sl_pct"]
        tp_price = r["tp_price"]
        sl_price = r["sl_price"]
        exit_signals = r.get("exit_signals", [])

        lines.append(f"{i}. {code} {name} / {sector}  Score: {score}")
        # 1行コメント：トレンド/押し目/流動性をざっくり
        comment_parts: List[str] = []
        if score >= 90:
            comment_parts.append("総合◎")
        elif score >= 80:
            comment_parts.append("総合◯")
        if r.get("trend_score", 0) >= 15:
            comment_parts.append("トレンド◎")
        elif r.get("trend_score", 0) >= 10:
            comment_parts.append("トレンド◯")
        if r.get("pb_score", 0) >= 12:
            comment_parts.append("押し目良好")
        if r.get("liq_score", 0) >= 12:
            comment_parts.append("流動性◎")
        comment = " / ".join(comment_parts) if comment_parts else "押し目候補"
        lines.append(f"   {comment}")

        # IN/OUT目安
        lines.append(
            f"   現値: {_fmt_yen(price)} / 利確目安: +{tp_pct*100:.1f}%({_fmt_yen(tp_price)})"
            f" / 損切り: {sl_pct*100:.1f}%({_fmt_yen(sl_price)})"
        )

        # OUTシグナル（あれば）
        if exit_signals:
            lines.append(f"   OUTシグナル: {' / '.join(exit_signals)}")

    return "\n".join(lines)


# =================================================
# 8. 全体スクリーニング処理
# =================================================

def screen_all() -> str:
    today = jst_today()
    today_str = today.strftime("%Y-%m-%d")

    # 地合い
    market_score = calc_market_score()

    # ユニバース
    try:
        universe = load_universe()
    except Exception as e:
        return f"📅 {today_str} stockbotTOM 日報\n\nユニバース読み込みエラー: {e}"

    core_list: List[Dict] = []

    for _, row in universe.iterrows():
        ticker = str(row["ticker"])
        name = str(row["name"])
        sector = str(row["sector"])

        # 日足取得
        df = fetch_ohlcv(ticker)
        if df is None:
            continue

        # 決算ウインドウなら除外
        if is_in_earnings_window(ticker, today, window=3):
            print(f"[INFO] skip {ticker} due to earnings window")
            continue

        # インジケータ付与
        df = add_indicators(df)
        if len(df) < 60:
            continue

        metrics = extract_metrics(df)

        # 株価フィルタ
        price = metrics.get("close", np.nan)
        if not np.isfinite(price) or price < CONFIG["MIN_PRICE"]:
            continue

        # 流動性フィルタ
        turnover_avg20 = metrics.get("turnover_avg20", np.nan)
        if not np.isfinite(turnover_avg20) or turnover_avg20 < CONFIG["MIN_TURNOVER"]:
            continue

        # セクター強度
        sector_score = calc_sector_strength(sector)

        # Coreスコア
        core_score = calc_core_score(metrics, market_score, sector_score)
        if core_score < CONFIG["CORE_SCORE_MIN"]:
            continue

        # 利確/損切り
        vol = metrics.get("vola20", np.nan)
        tp_pct, sl_pct = calc_take_profit_and_stop_loss(core_score, market_score, vol)
        tp_price = price * (1.0 + tp_pct)
        sl_price = price * (1.0 + sl_pct)

        # OUTシグナル
        exit_signals = evaluate_exit_signals(df)

        core_list.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "score": core_score,
                "price": price,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "exit_signals": exit_signals,
                "trend_score": calc_trend_score(metrics),
                "pb_score": calc_pullback_score(metrics),
                "liq_score": calc_liquidity_score(metrics),
            }
        )

    if not core_list:
        # 候補ゼロでも、地合いのコメントだけ返す
        max_lev, lev_label = calc_leverage_advice(market_score)
        msg = []
        msg.append(f"📅 {today_str} stockbotTOM 日報")
        msg.append("")
        msg.append("◆ 今日の結論")
        msg.append(f"- 地合いスコア: {market_score}点（{lev_label}）")
        msg.append(f"- レバ目安: 最大 約{max_lev:.1f}倍 / ポジ数目安: 0〜1銘柄")
        msg.append("- コメント: 本命の押し目候補なし。今日は無理に攻めない。")
        return "\n".join(msg)

    # スコア順に並べ替え
    core_list.sort(key=lambda x: x["score"], reverse=True)

    # メッセージ構築
    msg = build_line_message(today_str, market_score, core_list)
    return msg


# =================================================
# 9. エントリポイント
# =================================================

def main() -> None:
    text = screen_all()
    print(text)


if __name__ == "__main__":
    main()