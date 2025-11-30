from __future__ import annotations
import os
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
    # スクリーニング共通
    "MIN_PRICE": 300.0,       # 最低株価
    "MIN_TURNOVER": 1e8,      # 最低売買代金（直近20日平均）

    # Core スコアの閾値
    "CORE_SCORE_MIN": 75.0,

    # ボラティリティ分類
    "VOL_LOW_TH": 0.02,
    "VOL_HIGH_TH": 0.06,

    # TP / SL
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


def _fmt_yen(v: float) -> str:
    if not np.isfinite(v):
        return "-"
    return f"{int(round(v)):,}円"


# ============================================================
# Universe
# ============================================================
def load_universe(path: str = "universe_jpx.csv") -> pd.DataFrame:
    """
    ユニバースを読み込み。
    必須カラム: ticker, name, sector
    それ以外のカラムは無視。
    """
    if not os.path.exists(path):
        # 最低限のフェイルセーフ（まず使わない想定）
        df = pd.DataFrame(
            {
                "ticker": ["8035.T", "6920.T", "4502.T"],
                "name": ["Tokyo Electron", "Lasertec", "Takeda"],
                "sector": ["半導体", "半導体", "医薬"],
            }
        )
        return df

    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("universe_jpx.csv に ticker カラムがありません")

    # name / sector がなければダミーを作る
    if "name" not in df.columns:
        df["name"] = df["ticker"].astype(str)
    if "sector" not in df.columns:
        df["sector"] = "その他"

    df["ticker"] = df["ticker"].astype(str)
    df["name"] = df["name"].astype(str)
    df["sector"] = df["sector"].astype(str)

    return df[["ticker", "name", "sector"]]


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

    # 出来高関係
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    # ボラ20
    df["vola20"] = close.pct_change().rolling(20).std() * np.sqrt(20)

    # 60日高値からの乖離 & 経過日数
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100
        tail = close.tail(60)
        idx = int(np.argmax(tail.values))
        df["days_since_high60"] = (len(tail) - 1) - idx
    else:
        df["off_high_pct"] = np.nan
        df["days_since_high60"] = np.nan

    # トレンド傾き（20MA）
    df["trend_slope20"] = df["ma20"].pct_change()

    # 下ヒゲ比率
    rng = high - low
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    df["lower_shadow_ratio"] = np.where(rng > 0, lower_shadow / rng, 0)

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
# Market Score（安全版）
# ============================================================
def safe_download_close(ticker: str, days: int) -> Optional[pd.Series]:
    """落ちない安全版ダウンロード。Series or Noneを返す"""
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


def safe_return(ticker: str, days: int, fallback: str | None = None) -> float:
    """return = (最新 / X日前) - 1 の安全計算"""
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

    # ^TOPX は取れない → 1306.T (TOPIX ETF) で代替
    topix_ret1 = safe_return("^TOPX", 1, fallback="1306.T")
    topix_ret5 = safe_return("^TOPX", 5, fallback="1306.T")
    topix_ret20 = safe_return("^TOPX", 20, fallback="1306.T")

    nikkei_ret1 = safe_return("^N225", 1)
    nikkei_ret5 = safe_return("^N225", 5)

    jp1 = (topix_ret1 + nikkei_ret1) / 2
    jp5 = (topix_ret5 + nikkei_ret5) / 2
    jp20 = topix_ret20

    score = 50.0
    # ごく短期
    score += max(-15, min(15, jp1 * 100))
    # 1週間
    score += max(-10, min(10, jp5 * 50))
    # 1ヶ月
    score += max(-10, min(10, jp20 * 20))

    score = max(0, min(100, score))
    return int(score)


# ============================================================
# セクター強度（暫定：フラット）
# ============================================================
def calc_sector_strength(sector: str) -> int:
    """
    今はフラット（50点）。将来ここにセクター指数ロジックを入れる。
    """
    return 50


# ============================================================
# Core スコア（100点）
# ============================================================
def calc_trend_score(m: Dict[str, float]) -> int:
    close = m["close"]
    ma20 = m["ma20"]
    ma50 = m["ma50"]
    slope = m["trend_slope20"]
    off = m["off_high_pct"]

    sc = 0

    # slope（傾き）
    if np.isfinite(slope):
        if slope >= 0.01:
            sc += 8
        elif slope > 0:
            sc += 4 + slope / 0.01 * 4
        else:
            sc += max(0, 4 + slope * 50)

    # MA 関係
    if np.isfinite(close) and np.isfinite(ma20) and np.isfinite(ma50):
        if close > ma20 and ma20 > ma50:
            sc += 8
        elif close > ma20:
            sc += 4
        elif ma20 > ma50:
            sc += 2

    # 高値乖離
    if np.isfinite(off):
        if off >= -5:
            sc += 4
        elif off >= -15:
            sc += 4 - abs(off + 5) * 0.2

    return int(max(0, min(20, sc)))


def calc_pullback_score(m: Dict[str, float]) -> int:
    rsi = m["rsi14"]
    off = m["off_high_pct"]
    days = m["days_since_high60"]
    shadow = m["lower_shadow_ratio"]

    sc = 0

    # RSI 押し目具合
    if np.isfinite(rsi):
        if 30 <= rsi <= 45:
            sc += 7
        elif 20 <= rsi < 30 or 45 < rsi <= 55:
            sc += 4
        else:
            sc += 1

    # 高値からの下落度合い
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

    # 下ヒゲ
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

    # 出来高
    if np.isfinite(t):
        if t >= 10e8:
            sc += 16
        elif t >= 1e8:
            sc += 16 * (t - 1e8) / 9e8

    # ボラ
    if np.isfinite(v):
        if v < 0.02:
            sc += 4
        elif v < 0.06:
            sc += 4 * (0.06 - v) / 0.04

    return int(max(0, min(20, sc)))


def calc_core_score(m: Dict[str, float], market_score: int, sector_score: int) -> int:
    s_m = min(20, market_score * 0.2)  # 地合い
    s_s = min(20, sector_score * 0.2)  # セクター
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
# OUT Signals（利確・撤退ヒント）
# ============================================================
def evaluate_exit_signals(df: pd.DataFrame) -> List[str]:
    sig: List[str] = []
    if df.empty:
        return sig

    last = df.iloc[-1]
    rsi = _safe_float(last.get("rsi14"))
    turn = _safe_float(last.get("turnover"))
    avg20 = _safe_float(last.get("turnover_avg20"))

    # RSI 過熱
    if np.isfinite(rsi) and rsi >= 70:
        sig.append("RSI過熱")

    # 5MA 割れ連続
    if len(df) >= 3:
        d = df.tail(3)
        c = d["close"] < d["ma5"]
        if c.iloc[-2:].all():
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


# ============================================================
# ポジション読み込み & 分析
# ============================================================
def load_positions(path: str = "positions.csv") -> Tuple[List[Dict], Optional[str]]:
    """
    positions.csv を読み込み。
    フォーマット:
        ticker,qty,avg_price
    """
    if not os.path.exists(path):
        return [], "positions.csv が見つかりません（ポジション情報なし）。"

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return [], f"positions.csv 読み込みエラー: {e}"

    for col in ["ticker", "qty", "avg_price"]:
        if col not in df.columns:
            return [], f"positions.csv に {col} カラムがありません。"

    df["ticker"] = df["ticker"].astype(str)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")

    df = df[df["qty"] > 0]
    if df.empty:
        return [], "有効なポジションがありません（qty>0 が存在しません）。"

    return df.to_dict(orient="records"), None


def evaluate_hold_score(
    m: Dict[str, float], market_score: int, pnl_pct: float
) -> Tuple[int, str, str]:
    """
    持ち越し判定スコア（0-100）とラベル・コメントを返す。
    地合い + トレンド + RSI + ヒゲ + ボラ + 損益率 を合成。
    """
    score = 0.0

    # 地合い 0-30点
    score += market_score * 0.3

    # トレンド（20MA傾き）
    slope = m["trend_slope20"]
    if np.isfinite(slope):
        score += max(-15, min(20, slope * 400))  # 日次0.5% ↑ で +20 点くらい

    # RSI
    rsi = m["rsi14"]
    if np.isfinite(rsi):
        if 40 <= rsi <= 60:
            score += 15
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            score += 8
        elif rsi < 30:
            score += 5  # かなり売られてるが、リバ期待枠
        else:
            score += 3

    # 下ヒゲ（買い支え）
    shadow = m["lower_shadow_ratio"]
    if np.isfinite(shadow):
        if shadow >= 0.5:
            score += 10
        elif shadow >= 0.3:
            score += 5

    # ボラティリティ
    vola = m["vola20"]
    if np.isfinite(vola):
        if vola < 0.03:
            score += 5
        elif vola > 0.08:
            score -= 5

    # 高値からの乖離
    off = m["off_high_pct"]
    if np.isfinite(off):
        if -20 <= off <= 0:
            score += 5
        elif off < -25:
            score -= 10

    # 損益率（含み損・含み益の状態）
    if np.isfinite(pnl_pct):
        if pnl_pct < -8:
            score -= 10  # 含み損大きい
        elif pnl_pct < -3:
            score -= 5
        elif pnl_pct > 15:
            score -= 5  # 含み益パンパン → 利確寄りに傾ける

    score = float(max(0, min(100, score)))

    # ラベル決定
    if score >= 80:
        if pnl_pct >= 10:
            label = "分割利確推奨"
            comment = "地合い・トレンドともに強いが、含み益十分。サイズ調整しつつ利確を検討。"
        else:
            label = "強気持ち越し"
            comment = "地合い・個別の強さともに良好。通常サイズなら持ち越し優勢。"
    elif score >= 65:
        label = "持ち越しOK"
        comment = "持ち越し自体は問題なし。サイズが大きすぎる場合だけ調整を検討。"
    elif score >= 50:
        label = "様子見・軽め推奨"
        comment = "トレンド・地合いが微妙。サイズを落として様子見、次の足の形を確認。"
    elif score >= 35:
        label = "縮小・撤退候補"
        comment = "総合点が低め。戻り売りや、リバでの縮小・撤退を優先。"
    else:
        label = "撤退優先"
        comment = "持ち越すリスクがリターンを上回る水準。ルールに従い早めの損切りを検討。"

    return int(score), label, comment


def analyze_positions(
    market_score: int, universe: pd.DataFrame
) -> Tuple[List[Dict], Optional[str]]:
    """
    positions.csv に基づいてポジション分析を行う。
    戻り値:
        pos_list: 1銘柄ごとの詳細 dict のリスト
        pos_error: エラーメッセージ（あれば）
    """
    raw_positions, err = load_positions("positions.csv")
    if err:
        return [], err

    # ユニバース情報を dict 化（名前・セクター）
    uni_map = universe.set_index("ticker")[["name", "sector"]].to_dict("index")

    pos_list: List[Dict] = []
    failed: List[str] = []

    for row in raw_positions:
        ticker = str(row["ticker"])
        qty = int(row["qty"])
        avg_price = float(row["avg_price"])

        info = uni_map.get(ticker, {})
        name = info.get("name", ticker)
        sector = info.get("sector", "不明セクター")

        df = fetch_ohlcv(ticker)
        if df is None or len(df) < 60:
            failed.append(ticker)
            continue

        df = add_indicators(df)
        m = extract_metrics(df)

        price = m["close"]
        if not np.isfinite(price):
            failed.append(ticker)
            continue

        pnl_pct = (price - avg_price) / avg_price * 100.0
        pnl_yen = (price - avg_price) * qty

        sec_s = calc_sector_strength(sector)
        core_score = calc_core_score(m, market_score, sec_s)
        tp_pct, sl_pct = calc_tp_sl(core_score, market_score, m["vola20"])
        tp_price = price * (1 + tp_pct)
        sl_price = price * (1 + sl_pct)

        hold_score, hold_label, hold_comment = evaluate_hold_score(
            m, market_score, pnl_pct
        )
        exit_signals = evaluate_exit_signals(df)

        pos_list.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "qty": qty,
                "avg_price": avg_price,
                "cur_price": price,
                "pnl_pct": pnl_pct,
                "pnl_yen": pnl_yen,
                "core_score": core_score,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "hold_score": hold_score,
                "hold_label": hold_label,
                "hold_comment": hold_comment,
                "exit_signals": exit_signals,
            }
        )

    pos_error = None
    if failed:
        pos_error = f"※ 一部銘柄でデータ取得失敗: {', '.join(failed)}"

    # スコア高い順にソート（上から優先度が高い）
    pos_list.sort(key=lambda x: (x["hold_score"], x["core_score"]), reverse=True)

    return pos_list, pos_error


# ============================================================
# LINE Message
# ============================================================
def build_line_message(
    date_str: str,
    market_score: int,
    core_list: List[Dict],
    pos_list: Optional[List[Dict]] = None,
    pos_error: Optional[str] = None,
) -> str:
    max_lev, lev_label = calc_leverage_advice(market_score)

    lines: List[str] = []
    lines.append(f"📅 {date_str} stockbotTOM 日報\n")
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
            lines.append(
                f"{i}. {r['ticker']} {r['name']} / {r['sector']}  "
                f"Score: {r['score']}"
            )
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

            lines.append(
                "   " + (" / ".join(comment_parts) if comment_parts else "押し目候補")
            )
            lines.append(
                f"   現値:{_fmt_yen(r['price'])} / "
                f"利確:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['tp_price'])}) / "
                f"損切:{r['sl_pct']*100:.1f}%({_fmt_yen(r['sl_price'])})"
            )

            if r["exit_signals"]:
                lines.append(f"   OUT: {' / '.join(r['exit_signals'])}")
        lines.append("")

    # ポジション分析
    lines.append("◆ ポジション分析")
    if pos_error and not pos_list:
        # エラーだけ出すパターン
        lines.append(pos_error)
    elif not pos_list:
        lines.append("保有ポジション情報がありません（positions.csv 未設定 or 空）。")
    else:
        for i, p in enumerate(pos_list, 1):
            lines.append(
                f"{i}. {p['ticker']} {p['name']} / {p['sector']} "
                f"Core:{p['core_score']}点"
            )
            lines.append(
                f"   保有: {p['qty']}株 @ {_fmt_yen(p['avg_price'])}  "
                f"現値: {_fmt_yen(p['cur_price'])}"
            )
            lines.append(
                f"   評価損益: {_fmt_yen(p['pnl_yen'])}"
                f"（{p['pnl_pct']:+.1f}%）"
            )
            lines.append(
                f"   持ち越し判定: {p['hold_score']}点（{p['hold_label']}）"
            )
            lines.append(f"   コメント: {p['hold_comment']}")
            if p["exit_signals"]:
                lines.append(
                    f"   OUTサイン: {' / '.join(p['exit_signals'])}"
                )
        if pos_error:
            lines.append("")
            lines.append(pos_error)

    return "\n".join(lines)


# ============================================================
# Screening（Core候補抽出 + ポジション分析）
# ============================================================
def screen_all() -> str:
    today = jst_today()
    ds = today.strftime("%Y-%m-%d")

    market_score = calc_market_score()
    print("[DEBUG] market_score:", market_score)

    try:
        universe = load_universe()
    except Exception as e:
        return f"📅 {ds}\n\nユニバース読み込みエラー: {e}"

    core_list: List[Dict] = []

    for _, rw in universe.iterrows():
        t = rw["ticker"]
        name = rw["name"]
        sec = rw["sector"]

        df = fetch_ohlcv(t)
        if df is None:
            continue
        df = add_indicators(df)
        if len(df) < 60:
            continue

        m = extract_metrics(df)
        price = m["close"]

        # フィルター
        if not np.isfinite(price) or price < CONFIG["MIN_PRICE"]:
            continue
        if (
            not np.isfinite(m["turnover_avg20"])
            or m["turnover_avg20"] < CONFIG["MIN_TURNOVER"]
        ):
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
    pos_list, pos_error = analyze_positions(market_score, universe)

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
    Cloudflare Worker 経由で LINE に送信する。
    環境変数 WORKER_URL を使用。
    """
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