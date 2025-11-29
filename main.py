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
# CONFIG（あとでチューニングしやすい定数）
# ============================================================
CONFIG = {
    # 抽出フィルタ
    "MIN_PRICE": 300.0,        # 最低株価
    "MIN_TURNOVER": 1e8,       # 最低売買代金（直近20日平均）

    # Coreスコアしきい値（72〜74に落としてヒット数確保）
    "CORE_SCORE_MIN": 72.0,

    # ボラティリティ分類のしきい値
    "VOL_LOW_TH": 0.02,
    "VOL_HIGH_TH": 0.06,

    # 利確幅の下限/上限（%）
    "TP_MIN": 0.06,            # +6%
    "TP_MAX": 0.15,            # +15%

    # 損切り幅の下限/上限（マイナス値、%）
    "SL_UPPER": -0.03,         # -3%（最もタイト）
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


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)


# ============================================================
# Universe 読み込み
# ============================================================
def load_universe(path: str = "universe_jpx.csv") -> pd.DataFrame:
    """
    universe_jpx.csv から
      ticker, name, sector
    を読み込む。なければサンプルで代用。
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

    # フォールバック（ファイルが無い場合だけ）
    df = pd.DataFrame({
        "ticker": ["8035.T", "6920.T", "4502.T"],
        "name": ["Tokyo Electron", "Lasertec", "Takeda"],
        "sector": ["半導体", "半導体", "医薬"]
    })
    return df


# ============================================================
# OHLCV + インジケータ
# ============================================================
def fetch_ohlcv(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    """
    yfinance から日足を取得（安全版）。
    失敗時は None を返してスキップ。
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
    日足にテクニカル指標を付与：
      - MA5 / MA20 / MA50
      - RSI(14)
      - turnover / turnover_avg20
      - vola20
      - 60日高値からの乖離率 & 経過日数
      - 20MAの傾き
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

    # ボラ（20日）
    df["vola20"] = close.pct_change().rolling(20).std() * np.sqrt(20)

    # 60日高値からの乖離 & 日数
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
        "close", "ma5", "ma20", "ma50", "rsi14", "turnover_avg20",
        "off_high_pct", "vola20", "trend_slope20",
        "lower_shadow_ratio", "days_since_high60",
    ]
    return {k: _safe_float(last.get(k, np.nan)) for k in keys}


# ============================================================
# 地合いスコア（安全版）
# ============================================================
def safe_download_close(ticker: str, days: int) -> Optional[pd.Series]:
    """日次Closeだけ安全に取るヘルパー"""
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


def safe_return(ticker: str, days: int, fallback: str = None) -> float:
    """
    (最新 / X日前) - 1 を計算。
    取れなければ fallback（ETFなど）を使って0.0で妥協。
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
    地合いスコア（0〜100）
    ^TOPX が取れない問題を回避するため、1306.T をフォールバックに使用。
    """
    # TOPIX 近似
    topix_ret1 = safe_return("^TOPX", 1, fallback="1306.T")
    topix_ret5 = safe_return("^TOPX", 5, fallback="1306.T")
    topix_ret20 = safe_return("^TOPX", 20, fallback="1306.T")

    # 日経平均
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
def build_sector_perf(universe: pd.DataFrame,
                      df_cache: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, float]]:
    """
    各セクターごとに
      平均5日騰落率, 平均20日騰落率
    を計算する（ユニバースで重みなし平均）。
    """
    sector_data: Dict[str, List[Tuple[float, float]]] = {}

    for _, row in universe.iterrows():
        ticker = str(row["ticker"])
        sector = str(row["sector"])
        df = df_cache.get(ticker)
        if df is None or len(df) < 21:
            continue

        close = df["close"].astype(float)
        try:
            last = close.iloc[-1]
            d5 = close.iloc[-6]
            d20 = close.iloc[-21]
            r5 = float(last / d5 - 1.0)
            r20 = float(last / d20 - 1.0)
        except Exception:
            continue

        sector_data.setdefault(sector, []).append((r5, r20))

    avg_map: Dict[str, Tuple[float, float]] = {}
    for sec, vals in sector_data.items():
        if not vals:
            continue
        arr = np.array(vals)
        avg5 = float(np.nanmean(arr[:, 0]))
        avg20 = float(np.nanmean(arr[:, 1]))
        avg_map[sec] = (avg5, avg20)

    return avg_map


def build_sector_strength_map(universe: pd.DataFrame,
                              df_cache: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """
    セクター別に「0〜100」の強度スコアを作成。
    - 5日パフォーマンス
    - TOPIX比の20日パフォーマンス
    から加点減点してスコア化。
    """
    perf = build_sector_perf(universe, df_cache)
    topix20 = safe_return("^TOPX", 20, fallback="1306.T")

    sec_map: Dict[str, int] = {}
    for sec, (r5, r20) in perf.items():
        score = 50.0

        # 5日：+5%で +20pt / -5%で -20pt（±20にクリップ）
        score += max(-20.0, min(20.0, r5 * 400.0))

        # 20日：TOPIXに対する相対 +5%で +15pt（±15にクリップ）
        rel20 = r20 - topix20
        score += max(-15.0, min(15.0, rel20 * 300.0))

        score = max(0.0, min(100.0, score))
        sec_map[sec] = int(round(score))

    # データが無かったセクターは50点で埋める
    for _, row in universe.iterrows():
        sec = str(row["sector"])
        if sec not in sec_map:
            sec_map[sec] = 50

    return sec_map


# ============================================================
# Coreスコア（100点）
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
            sc += 4.0 + slope / 0.01 * 4.0
        else:
            sc += max(0.0, 4.0 + slope * 50.0)

    # MA関係
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

    return int(max(0, min(20, sc)))


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

    return int(max(0, min(20, sc)))


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

    return int(max(0, min(20, sc)))


def calc_core_score(m: Dict[str, float],
                    market_score: int,
                    sector_score: int) -> int:
    """
    Coreスコア = 地合い + セクター + トレンド + 押し目 + 流動性
    各 0〜20点 → 合計 0〜100点
    """
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
    """
    利確幅(tp), 損切り幅(sl) を % で返す（+0.1 → +10%）
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
    if df is None or df.empty:
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
        c = (d["close"] < d["ma5"])
        if c.iloc[-2:].all():
            sig.append("5MA割れ連続")

    # 出来高急減
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
# ポジション管理（positions.csv を前提 / 無ければスキップ）
# ============================================================
def load_positions(path: str = "positions.csv") -> List[Dict]:
    """
    positions.csv（任意）を読み込む。
    必須カラム: ticker, entry_price, size
    任意: note
    """
    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] positions.csv 読み込み失敗: {e}")
        return []

    required = {"ticker", "entry_price", "size"}
    if not required.issubset(df.columns):
        print("[WARN] positions.csv のカラム不足 (ticker, entry_price, size 必須)")
        return []

    pos_list: List[Dict] = []
    for _, row in df.iterrows():
        try:
            pos_list.append(
                {
                    "ticker": str(row["ticker"]),
                    "entry_price": float(row["entry_price"]),
                    "size": float(row["size"]),
                    "note": str(row.get("note", "")),
                }
            )
        except Exception:
            continue

    return pos_list


def evaluate_position_comment(pnl_pct: float,
                              core_score: Optional[int],
                              market_score: int) -> str:
    """
    含み損益・Coreスコア・地合いからざっくりコメント。
    （あくまで「参考コメント」）
    """
    if not np.isfinite(pnl_pct):
        return "評価不可"

    if pnl_pct <= -5.0:
        return "想定以上の悪化。基本は撤退検討ゾーン。"
    if -5.0 < pnl_pct <= -2.5:
        return "損切りライン接近。地合いと個別トレンド次第で一部撤退検討。"
    if -2.5 < pnl_pct < 0.0:
        return "許容範囲の含み損。ルール内の損切りラインを厳守。"

    if 0.0 <= pnl_pct < 5.0:
        if core_score and core_score >= 80 and market_score >= 50:
            return "含み益小。トレンド強ければホールド優勢。利食い早すぎ注意。"
        return "軽い含み益。地合い悪化時は早め利食いも選択肢。"

    if 5.0 <= pnl_pct < 15.0:
        return "十分な含み益ゾーン。半利確 or トレーリングストップを検討。"

    return "大きな含み益ゾーン。利確優先でOK。欲張りすぎ注意。"


def build_positions_summary(
    positions: List[Dict],
    df_cache: Dict[str, pd.DataFrame],
    metrics_cache: Dict[str, Dict[str, float]],
    core_map: Dict[str, int],
    market_score: int,
) -> str:
    if not positions:
        return "◆ ポジション分析\n保有ポジション情報がありません（positions.csv 未設定）。"

    lines: List[str] = []
    lines.append("◆ ポジション分析（参考）")

    for pos in positions:
        ticker = pos["ticker"]
        entry = pos["entry_price"]
        size = pos["size"]
        note = pos.get("note", "")

        df = df_cache.get(ticker)
        metrics = metrics_cache.get(ticker)
        price = None

        if metrics is not None:
            price = metrics.get("close", np.nan)
        else:
            if df is None:
                df = fetch_ohlcv(ticker, period="60d")
            if df is not None:
                df = add_indicators(df)
                df_cache[ticker] = df
                metrics = extract_metrics(df)
                metrics_cache[ticker] = metrics
                price = metrics.get("close", np.nan)

        if price is None or not np.isfinite(price) or entry <= 0:
            pnl_pct = np.nan
        else:
            pnl_pct = (price / entry - 1.0) * 100.0

        core_score = core_map.get(ticker)
        comment = evaluate_position_comment(pnl_pct, core_score, market_score)

        lines.append(
            f"- {ticker}  取得: {_fmt_yen(entry)} / 現在値: {_fmt_yen(price)} / 損益: {pnl_pct:+.2f}%"
        )
        if note and note != "nan":
            lines.append(f"   メモ: {note}")
        lines.append(f"   判断: {comment}")

    return "\n".join(lines)


# ============================================================
# LINE メッセージ構築
# ============================================================
def build_line_message(
    date_str: str,
    market_score: int,
    core_list: List[Dict],
    sector_strength_map: Dict[str, int],
    positions_text: str,
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

    # セクター強度 TOP3
    if sector_strength_map:
        lines.append("◆ 強いセクターTOP3")
        ranked = sorted(sector_strength_map.items(), key=lambda x: x[1], reverse=True)
        for i, (sec, sc) in enumerate(ranked[:3], 1):
            lines.append(f"{i}位: {sec}（強度 {sc}）")
        lines.append("")

    # ポジション分析
    lines.append(positions_text)
    lines.append("")

    # Core候補
    lines.append("◆ Core候補（本命押し目）")
    if not core_list:
        lines.append("本命条件なし。今日は無理しない。")
        return "\n".join(lines)

    for i, r in enumerate(core_list[:10], 1):
        lines.append(f"{i}. {r['ticker']} {r['name']} / {r['sector']}  Score: {r['score']}")

        comment: List[str] = []
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

    return "\n".join(lines)


# ============================================================
# メインスクリーニング処理
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
        return f"📅{ds}\n\nユニバース読み込みエラー:{e}"

    # 1周目：全銘柄のOHLCV＋指標をキャッシュ
    df_cache: Dict[str, pd.DataFrame] = {}
    metrics_cache: Dict[str, Dict[str, float]] = {}

    for _, rw in universe.iterrows():
        t = str(rw["ticker"])
        df = fetch_ohlcv(t)
        if df is None:
            continue

        df_ind = add_indicators(df)
        if len(df_ind) < 60:
            continue

        df_cache[t] = df_ind
        metrics_cache[t] = extract_metrics(df_ind)

    # セクター強度マップ
    sector_strength_map = build_sector_strength_map(universe, df_cache)

    # ポジション読み込み
    positions = load_positions()

    core_map: Dict[str, int] = {}
    core_list: List[Dict] = []

    # 2周目：スコアリング＆Core候補抽出
    for _, rw in universe.iterrows():
        t = str(rw["ticker"])
        name = str(rw["name"])
        sec = str(rw["sector"])

        df = df_cache.get(t)
        metrics = metrics_cache.get(t)
        if df is None or metrics is None:
            continue

        price = metrics.get("close", np.nan)
        if not np.isfinite(price) or price < CONFIG["MIN_PRICE"]:
            continue

        if (not np.isfinite(metrics.get("turnover_avg20", np.nan))
                or metrics["turnover_avg20"] < CONFIG["MIN_TURNOVER"]):
            continue

        sec_s = sector_strength_map.get(sec, 50)
        core = calc_core_score(metrics, market_score, sec_s)

        if core < CONFIG["CORE_SCORE_MIN"]:
            continue

        vol = metrics.get("vola20", np.nan)
        tp, sl = calc_tp_sl(core, market_score, vol)
        tp_price = price * (1 + tp)
        sl_price = price * (1 + sl)

        ex = evaluate_exit_signals(df)

        core_map[t] = core
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
                "trend_score": calc_trend_score(metrics),
                "pb_score": calc_pullback_score(metrics),
                "liq_score": calc_liquidity_score(metrics),
                "exit_signals": ex,
            }
        )

    # ポジション分析テキスト
    positions_text = build_positions_summary(
        positions, df_cache, metrics_cache, core_map, market_score
    )

    # Core候補ゼロでも、地合い＋ポジ分析は出す
    if not core_list:
        max_lev, lev_label = calc_leverage_advice(market_score)
        msg: List[str] = []
        msg.append(f"📅 {ds} stockbotTOM 日報")
        msg.append("")
        msg.append("◆ 今日の結論")
        msg.append(f"- 地合いスコア: {market_score}点（{lev_label}）")
        msg.append(f"- レバ目安: 最大 約{max_lev:.1f}倍")
        msg.append("- コメント: Core候補なし。今日は静観。")
        msg.append("")
        msg.append(positions_text)
        return "\n".join(msg)

    # スコア順ソート
    core_list.sort(key=lambda x: x["score"], reverse=True)

    # メッセージ構築
    msg = build_line_message(
        ds, market_score, core_list, sector_strength_map, positions_text
    )
    return msg


# ============================================================
# Cloudflare Worker → LINE 送信
# ============================================================
def send_to_lineworker(text: str):
    """
    GitHub Actions から Cloudflare Worker にPOST → LINE通知
    環境変数 WORKER_URL を事前設定しておく。
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
# Entry point
# ============================================================
def main():
    text = screen_all()
    print(text)
    send_to_lineworker(text)


if __name__ == "__main__":
    main()