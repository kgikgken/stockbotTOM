from __future__ import annotations

import os
import json
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests


# ============================================================
# CONFIG（あとで調整したくなるやつをここに集約）
# ============================================================
CONFIG: Dict[str, float] = {
    # 抽出フィルタ
    "MIN_PRICE": 300.0,        # 最低株価
    "MIN_TURNOVER": 1e8,       # 最低売買代金（直近20日平均）

    # Coreスコア閾値（Aランク）
    "CORE_SCORE_MIN": 72.0,    # ここを動かせばAランクの厳しさを変えられる

    # ボラティリティしきい値
    "VOL_LOW_TH": 0.02,
    "VOL_HIGH_TH": 0.06,

    # 利確幅の下限/上限（%）
    "TP_MIN": 0.06,
    "TP_MAX": 0.15,

    # 損切り幅の下限/上限（マイナス値）
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
    任意: name, sector （なければ全部埋める）
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            raise ValueError("universe_jpx.csv に 'ticker' カラムがありません。")

        df["ticker"] = df["ticker"].astype(str)
        df["name"] = df.get("name", df["ticker"]).astype(str)
        df["sector"] = df.get("sector", "その他").astype(str)
        return df[["ticker", "name", "sector"]]

    # フォールバック（何もないとき用の簡易ユニバース）
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
    """yfinance から日足OHLCVを取得（失敗したら None）。"""
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
    日足 OHLCV DataFrame に各種インジケータを追加：
      - MA5, MA20, MA50
      - RSI14
      - 売買代金 & 20日平均
      - ボラティリティ20（日次リターンstd×√20）
      - 60日高値からの距離 & 経過日数
      - 20MAの傾き
      - 下ヒゲ比率
      - 5日/20日リターン（セクター強度用）
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

    # 60日高値からの距離 & 経過日数
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
    df["lower_shadow_ratio"] = np.where(rng > 0, lower_shadow / rng, 0.0)

    # 5日 & 20日リターン（セクター強度用）
    df["ret5"] = close.pct_change(5)
    df["ret20"] = close.pct_change(20)

    return df


def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """スコア計算で使う指標を最終行から抜き出す。"""
    last = df.iloc[-1]
    keys = [
        "close", "ma5", "ma20", "ma50", "rsi14", "turnover_avg20",
        "off_high_pct", "vola20", "trend_slope20",
        "lower_shadow_ratio", "days_since_high60",
        "ret5", "ret20",
    ]
    return {k: _safe_float(last.get(k, np.nan)) for k in keys}


# ============================================================
# Market Score（安全版）
# ============================================================
def safe_download_close(ticker: str, days: int) -> Optional[pd.Series]:
    """安全版ダウンロード。取れなければ None。"""
    try:
        df = yf.download(
            ticker,
            period="90d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"[WARN] market download failed: {ticker}: {e}")
        return None

    if df is None or df.empty or "Close" not in df.columns:
        return None
    if len(df) <= days:
        return None
    return df["Close"].astype(float)


def safe_return(ticker: str, days: int, fallback: Optional[str] = None) -> float:
    """return = (最新 / X日前) - 1 を安全に計算。"""
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
    """安全な地合いスコア（0-100）。TOPIX ETF 1306.T をフォールバックに使用。"""
    topix_ret1 = safe_return("^TOPX", 1, fallback="1306.T")
    topix_ret5 = safe_return("^TOPX", 5, fallback="1306.T")
    topix_ret20 = safe_return("^TOPX", 20, fallback="1306.T")

    nikkei_ret1 = safe_return("^N225", 1)
    nikkei_ret5 = safe_return("^N225", 5)

    jp1 = (topix_ret1 + nikkei_ret1) / 2.0
    jp5 = (topix_ret5 + nikkei_ret5) / 2.0
    jp20 = topix_ret20

    score = 50.0
    score += max(-15.0, min(15.0, jp1 * 100))   # 1日 +1% → +1
    score += max(-10.0, min(10.0, jp5 * 50))    # 5日 +3% → +7.5
    score += max(-10.0, min(10.0, jp20 * 20))   # 20日 +5% → +5

    score = max(0.0, min(100.0, score))
    return int(score)


# ============================================================
# セクター強度（実データ版）
# ============================================================
def build_sector_strength_map(sector_stats: Dict[str, Dict[str, List[float]]]) -> Dict[str, int]:
    """
    セクターごとの 5日・20日リターン平均を取り、
    TOPIX（1306.T）に対する相対強弱で 0〜100点に変換。
    """
    topix5 = safe_return("^TOPX", 5, fallback="1306.T")
    topix20 = safe_return("^TOPX", 20, fallback="1306.T")

    result: Dict[str, int] = {}

    for sector, stats in sector_stats.items():
        ret5_list = [x for x in stats.get("ret5", []) if np.isfinite(x)]
        ret20_list = [x for x in stats.get("ret20", []) if np.isfinite(x)]

        if not ret5_list and not ret20_list:
            result[sector] = 50
            continue

        avg5 = float(np.mean(ret5_list)) if ret5_list else 0.0
        avg20 = float(np.mean(ret20_list)) if ret20_list else 0.0

        rel5 = avg5 - topix5
        rel20 = avg20 - topix20

        score = 50.0
        # 20日相対 +5% → +20点、-5% → -20点（クリップ）
        score += max(-20.0, min(20.0, rel20 * 400.0))
        # 5日相対 +2.5% → +20点、-2.5% → -20点（クリップ）
        score += max(-20.0, min(20.0, rel5 * 800.0))

        # 上昇トレンドセクターには少しボーナス
        if avg20 > 0 and avg5 > 0:
            score += 5.0
        elif avg20 < 0 and avg5 < 0:
            score -= 5.0

        score = max(0.0, min(100.0, score))
        result[sector] = int(round(score))

    return result


# ============================================================
# Core スコア（100点）
# ============================================================
def calc_trend_score(m: Dict[str, float]) -> int:
    """
    トレンド強度（0〜20）
      - 20MAの傾き
      - 価格 > MA20 > MA50
      - 60日高値からの距離
    """
    close = m.get("close", np.nan)
    ma20 = m.get("ma20", np.nan)
    ma50 = m.get("ma50", np.nan)
    slope = m.get("trend_slope20", np.nan)
    off = m.get("off_high_pct", np.nan)

    sc = 0.0

    # 20MAの傾き
    if np.isfinite(slope):
        if slope >= 0.01:      # 1%/日レベル → 超強い
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
        if off >= -5:
            sc += 4.0
        elif -15 <= off < -5:
            sc += 4.0 - abs(off + 5.0) * 0.2

    return int(max(0, min(20, round(sc))))


def calc_pullback_score(m: Dict[str, float]) -> int:
    """
    押し目の質（0〜20）
      - RSI：30〜45 が理想
      - 高値からの下落率：-5〜-12% が理想
      - 日柄：2〜10日が理想
      - 下ヒゲ比率
    """
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

    return int(max(0, min(20, round(sc))))


def calc_liquidity_score(m: Dict[str, float]) -> int:
    """
    流動性 & 安定度（0〜20）
      - turnover_avg20 が高いほど加点
      - vola20 が高すぎると減点
    """
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

    return int(max(0, min(20, round(sc))))


def calc_core_score(m: Dict[str, float], market_score: int, sector_score: int) -> int:
    """
    Coreスコア（100点）
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
    利確幅(tp), 損切り幅(sl) を返す（例: +0.1=+10%, -0.04=-4%）
    """
    # --- TP ---
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

    # --- SL ---
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
    rsi = _safe_float(last.get("rsi14", np.nan))
    turn = _safe_float(last.get("turnover", np.nan))
    avg20 = _safe_float(last.get("turnover_avg20", np.nan))

    # RSI過熱
    if np.isfinite(rsi) and rsi >= 70:
        sig.append("RSI過熱")

    # 5MA割れ連続
    if len(df) >= 3 and "ma5" in df.columns and "close" in df.columns:
        d = df.tail(3)
        c = (d["close"] < d["ma5"])
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


def _fmt_yen(v: float) -> str:
    if not np.isfinite(v):
        return "-"
    return f"{int(round(v)):,}円"


# ============================================================
# LINE Message (Core セクション)
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

    lines.append("◆ Core候補（本命押し目）")
    if not core_list:
        lines.append("本命条件なし。今日は無理しない。")
        return "\n".join(lines)

    # PickUpモードの説明（本命0だが一番マシな1銘柄だけ出している日）
    if any(r.get("is_pickup", False) for r in core_list):
        lines.append("※ 本命スコアには届かないが、ユニバース内で相対的にマシな押し目候補。")
        lines.append("   無理INせず、板・出来高・ニュースを確認した上で慎重に判断。")

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

        comment = " / ".join(comment_parts) if comment_parts else "押し目候補"
        lines.append(f"   {comment}")

        lines.append(
            f"   現値:{_fmt_yen(r['price'])} / "
            f"利確:+{r['tp_pct']*100:.1f}%({_fmt_yen(r['tp_price'])}) / "
            f"損切:{r['sl_pct']*100:.1f}%({_fmt_yen(r['sl_price'])})"
        )

        if r["exit_signals"]:
            lines.append(f"   OUT: {' / '.join(r['exit_signals'])}")

    return "\n".join(lines)


# ============================================================
# ポジション分析セクション
# ============================================================
def read_equity() -> Optional[float]:
    """
    equity.json を読み込み（なければ None）
    env で data/equity.json に書いている前提を想定。
    """
    candidates = ["data/equity.json", "equity.json"]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                eq = float(data.get("equity", 0.0))
                if eq > 0:
                    return eq
            except Exception as e:
                print(f"[WARN] read_equity failed ({path}): {e}")
                continue
    return None


def build_positions_section(market_score: int) -> List[str]:
    """
    positions.csv を読み込んでポジション分析コメントを返す。
    必須: ticker, qty, avg_price
    """
    lines: List[str] = []
    lines.append("◆ ポジション分析")

    if not os.path.exists("positions.csv"):
        lines.append("positions.csv がありません。保有ポジションがある場合は設定してください。")
        return lines

    try:
        df = pd.read_csv("positions.csv")
    except Exception as e:
        lines.append(f"positions.csv 読み込みエラー: {e}")
        return lines

    required_cols = {"ticker", "qty", "avg_price"}
    if not required_cols.issubset(df.columns):
        lines.append("positions.csv には 'ticker','qty','avg_price' カラムが必要です。")
        return lines

    df["ticker"] = df["ticker"].astype(str)
    df["qty"] = df["qty"].astype(float)
    df["avg_price"] = df["avg_price"].astype(float)

    df = df[df["qty"] > 0]
    if df.empty:
        lines.append("保有ポジションはありません。")
        return lines

    equity = read_equity()
    if equity is not None:
        lines.append(f"推定運用資産: {_fmt_yen(equity)}")

    total_value = 0.0
    pos_detail_lines: List[str] = []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        qty = row["qty"]
        avg_price = row["avg_price"]

        px = np.nan
        ohlcv = fetch_ohlcv(ticker, period="60d")
        if ohlcv is not None and not ohlcv.empty:
            px = _safe_float(ohlcv["Close"].iloc[-1])
        else:
            pos_detail_lines.append(f"- {ticker}: データ取得失敗（現値不明）")
            continue

        if not np.isfinite(px):
            pos_detail_lines.append(f"- {ticker}: データ取得失敗（現値NaN）")
            continue

        value = px * qty
        total_value += value
        pnl_pct = (px - avg_price) / avg_price * 100.0

        weight_str = ""
        if equity is not None and equity > 0:
            w = value / equity * 100.0
            weight_str = f" / 資産比率 {w:.1f}%"

        pos_detail_lines.append(
            f"- {ticker}: 現値 {px:.1f} / 取得 {avg_price:.1f} / 損益 {pnl_pct:+.2f}%{weight_str}"
        )

    # 詳細行
    lines.extend(pos_detail_lines)

    # 総ポジション・レバ
    if equity is not None and equity > 0 and total_value > 0:
        lev = total_value / equity
        lines.append(f"推定総ポジション: {_fmt_yen(total_value)}（レバ約 {lev:.2f}倍）")

        rec_lev, _ = calc_leverage_advice(market_score)
        if lev > rec_lev * 1.1:
            lines.append("※ 推奨レバを超過中。サイズ縮小候補。")
        elif lev < rec_lev * 0.5:
            lines.append("※ 余力あり。地合いとCore候補次第で追加余地あり。")

    return lines


# ============================================================
# Screening 本体
# ============================================================
def screen_all() -> str:
    today = jst_today()
    ds = today.strftime("%Y-%m-%d")

    market_score = calc_market_score()

    # --- ユニバース読み込み ---
    try:
        universe = load_universe()
    except Exception as e:
        core_msg = (
            f"📅 {ds} stockbotTOM 日報\n\n"
            f"◆ 今日の結論\n"
            f"- 地合いスコア: {market_score}点\n"
            f"- コメント: ユニバース読み込みエラー: {e}\n"
        )
        pos = "\n".join(build_positions_section(market_score))
        return core_msg + "\n\n" + pos

    # --- 1st pass: 各銘柄の指標とセクターデータ収集 ---
    records: List[Dict] = []
    sector_stats: Dict[str, Dict[str, List[float]]] = {}

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

        # セクター強度用に 5日・20日リターンを集計
        ret5 = metrics.get("ret5", np.nan)
        ret20 = metrics.get("ret20", np.nan)

        if sector not in sector_stats:
            sector_stats[sector] = {"ret5": [], "ret20": []}
        if np.isfinite(ret5):
            sector_stats[sector]["ret5"].append(ret5)
        if np.isfinite(ret20):
            sector_stats[sector]["ret20"].append(ret20)

        records.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "df": df,
                "metrics": metrics,
            }
        )

    # records がゼロ → データ取得失敗系
    if not records:
        max_lev, lev_label = calc_leverage_advice(market_score)
        core_msg = (
            f"📅 {ds} stockbotTOM 日報\n\n"
            f"◆ 今日の結論\n"
            f"- 地合いスコア: {market_score}点（{lev_label}）\n"
            f"- レバ目安: 最大 約{max_lev:.1f}倍\n"
            f"- コメント: データ取得に失敗 or ユニバース対象外。今日は静観。\n"
        )
        pos = "\n".join(build_positions_section(market_score))
        return core_msg + "\n\n" + pos

    # --- セクター強度マップ作成 ---
    sector_strength_map = build_sector_strength_map(sector_stats)

    # --- 2nd pass: Coreスコア計算 + フィルタ ---
    core_list: List[Dict] = []
    scored_list: List[Dict] = []

    for rec in records:
        ticker = rec["ticker"]
        name = rec["name"]
        sector = rec["sector"]
        df = rec["df"]
        m = rec["metrics"]

        price = m.get("close", np.nan)
        turnover_avg20 = m.get("turnover_avg20", np.nan)

        # 価格・流動性フィルタ
        if not np.isfinite(price) or price < CONFIG["MIN_PRICE"]:
            continue
        if not np.isfinite(turnover_avg20) or turnover_avg20 < CONFIG["MIN_TURNOVER"]:
            continue

        sector_score = sector_strength_map.get(sector, 50)
        core = calc_core_score(m, market_score, sector_score)

        vol = m.get("vola20", np.nan)
        tp, sl = calc_tp_sl(core, market_score, vol)
        tp_price = price * (1.0 + tp)
        sl_price = price * (1.0 + sl)
        exit_signals = evaluate_exit_signals(df)

        row = {
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
            "is_pickup": False,
        }

        scored_list.append(row)
        if core >= CONFIG["CORE_SCORE_MIN"]:
            core_list.append(row)

    # Core候補ゼロの日 → ユニバース内で一番スコア高い1銘柄を「PickUp候補」として出す
    if not core_list and scored_list:
        best = max(scored_list, key=lambda x: x["score"])
        best["is_pickup"] = True
        core_list = [best]

    # Core候補もPickUp候補もない（ほぼ起こらない想定）
    if not core_list:
        max_lev, lev_label = calc_leverage_advice(market_score)
        core_msg = (
            f"📅 {ds} stockbotTOM 日報\n\n"
            f"◆ 今日の結論\n"
            f"- 地合いスコア: {market_score}点（{lev_label}）\n"
            f"- レバ目安: 最大 約{max_lev:.1f}倍\n"
            f"- コメント: Core候補なし。今日は無理に攻めない。\n"
        )
        pos = "\n".join(build_positions_section(market_score))
        return core_msg + "\n\n" + pos

    # スコア順ソート
    core_list.sort(key=lambda x: x["score"], reverse=True)

    # Coreセクション
    core_msg = build_line_message(ds, market_score, core_list)

    # ポジションセクション
    pos_msg_lines = build_positions_section(market_score)
    full = core_msg + "\n\n" + "\n".join(pos_msg_lines)
    return full


# ============================================================
# Send to Worker (LINE)
# ============================================================
def send_to_lineworker(text: str) -> None:
    """
    Cloudflare Worker に結果を送る → Worker が LINE へ投稿。
    GitHub Actions では secrets.WORKER_URL に設定してある前提。
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
# Entry point
# ============================================================
def main() -> None:
    text = screen_all()
    print(text)
    send_to_lineworker(text)


if __name__ == "__main__":
    main()