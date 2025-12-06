from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.util import jst_today_str


# ============================================================
# 基本設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"      # あれば読む（無ければ無視）
WORKER_URL = os.getenv("WORKER_URL")

# スクリーニング関連
SCREENING_TOP_N = 15            # まずは Top15 まで抽出
MAX_FINAL_STOCKS = 5            # 最終的に LINE に出すのは最大5銘柄
MAX_CORE_POSITIONS = 3          # 同時に持つ本命 Core 最大数

# 決算フィルタ: ±N日
EARNINGS_EXCLUDE_DAYS = 3


# ============================================================
# 日付 / イベント関連
# ============================================================
def jst_today_date() -> datetime.date:
    # JST の「今日」の date
    return datetime.now(timezone(timedelta(hours=9))).date()


def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    # events.csv があれば読み込んでリスト返す
    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read events: {e}")
        return []

    events: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        date_str = str(row.get("date", "")).strip()
        label = str(row.get("label", "")).strip()
        kind = str(row.get("kind", "")).strip()
        if not date_str or not label:
            continue
        events.append({"date": date_str, "label": label, "kind": kind})
    return events


def build_event_warnings(today: datetime.date) -> List[str]:
    # イベントの2日前〜翌日まで警告
    events = load_events()
    warns: List[str] = []
    for ev in events:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except Exception:
            continue

        delta = (d - today).days
        if -1 <= delta <= 2:
            if delta > 0:
                when = f"{delta}日後"
            elif delta == 0:
                when = "本日"
            else:
                when = "直近"
            warns.append(f"⚠ {ev['label']}（{when}）: ポジションサイズ注意")
    return warns


# ============================================================
# Universe / データ取得
# ============================================================
def load_universe(path: str = UNIVERSE_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"[WARN] universe file not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read universe: {e}")
        return None

    if "ticker" not in df.columns:
        print("[WARN] universe has no 'ticker' column")
        return None

    df["ticker"] = df["ticker"].astype(str)

    # earnings_date を一度だけパース
    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


def in_earnings_window(row: pd.Series, today: datetime.date) -> bool:
    # 決算日 ±EARNINGS_EXCLUDE_DAYS に入っていれば True
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    try:
        delta = abs((d - today).days)
    except Exception:
        return False
    return delta <= EARNINGS_EXCLUDE_DAYS


def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    # 株価履歴取得（簡易リトライ付き）
    for attempt in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f"[WARN] fetch history failed {ticker} (try {attempt+1}): {e}")
            time.sleep(1.0)

    return None


# ============================================================
# テクニカル指標
# ============================================================
def calc_ma(close: pd.Series, window: int) -> float:
    if len(close) < window:
        return float(close.iloc[-1])
    return float(close.rolling(window).mean().iloc[-1])


def calc_rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) <= period + 1:
        return 50.0

    diff = close.diff(1)
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)

    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()

    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    v = float(rsi.iloc[-1])
    if not np.isfinite(v):
        return 50.0
    return v


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) <= period + 1:
        return 0.0

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]

    if atr is None or not np.isfinite(atr):
        return 0.0
    return float(atr)


def calc_volatility(close: pd.Series, window: int = 20) -> float:
    if len(close) < window + 1:
        return 0.03

    ret = close.pct_change(fill_method=None)
    v = ret.rolling(window).std().iloc[-1]

    if v is None or not np.isfinite(v):
        return 0.03
    return float(v)


def calc_vwap(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    # 日足ベースの簡易VWAP（出来高加重平均）
    if len(close) < window or len(volume) < window:
        return float(close.iloc[-1])
    c = close.tail(window)
    v = volume.tail(window)
    denom = v.sum()
    if denom <= 0:
        return float(c.iloc[-1])
    return float((c * v).sum() / denom)


# ============================================================
# レバレッジ / 建て玉
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    # 地合いスコアから推奨レバ / コメント
    if mkt_score >= 80:
        return 2.0, "攻めMAX（ただしルール外エントリー禁止）"
    if mkt_score >= 70:
        return 1.8, "やや攻め（押し目＋強いブレイク）"
    if mkt_score >= 60:
        return 1.5, "標準〜やや攻め（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "標準（本命押し目のみ）"
    if mkt_score >= 40:
        return 1.1, "やや守り（ロット控えめ）"
    return 1.0, "守り（新規は最小ロット〜様子見）"


def risk_per_trade(mkt_score: int) -> float:
    # 地合いに応じて 1トレードあたりのリスク率を変える
    if mkt_score >= 70:
        return 0.018
    if mkt_score >= 60:
        return 0.013
    if mkt_score >= 50:
        return 0.010
    return 0.007


def calc_max_position(total_asset: float, lev: float) -> int:
    # 今日使っていい建て玉最大金額
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# 動的な最低スコアライン（地合い連動）
# ============================================================
def dynamic_min_score(mkt_score: int) -> float:
    # 地合いが強いほど少し緩く、弱いほど厳しくフィルタ
    if mkt_score >= 75:
        return 75.0
    if mkt_score >= 65:
        return 85.0
    if mkt_score >= 55:
        return 90.0
    if mkt_score >= 45:
        return 93.0
    return 96.0


# ============================================================
# セクター強度（5日モメンタム＋順位）
# ============================================================
def build_sector_strength_map() -> Dict[str, float]:
    # top_sectors_5d() から「順位＋5日騰落」で強度スコア
    secs = top_sectors_5d()
    strength: Dict[str, float] = {}

    for rank, row in enumerate(secs[:10]):
        # 互換性のため name, chg5 / (name, chg5, chg20) 両対応
        if len(row) >= 2:
            name, chg5 = row[0], row[1]
        else:
            continue

        base_rank = max(0, 8 - rank)  # 1位:8, 2位:7, ...
        boost = float(np.clip(chg5 * 1.5, -6.0, 6.0))
        strength[str(name)] = base_rank + boost

    return strength


# ============================================================
# 三階層スコアの重み（地合いで可変）
# ============================================================
def get_score_weights(mkt_score: int) -> Tuple[float, float, float]:
    # Quality / Setup / Regime の重み（地合いで変化）
    if mkt_score >= 75:
        return 0.6, 1.2, 0.7
    if mkt_score >= 60:
        return 0.7, 1.0, 0.7
    if mkt_score >= 50:
        return 0.8, 0.9, 0.8
    if mkt_score >= 40:
        return 0.8, 0.7, 1.0
    return 0.9, 0.6, 1.1


# ============================================================
# 波の大きさ（3〜4%回転波 / 7〜8%波 など）
# ============================================================
def classify_wave(swing_upside: Optional[float]) -> str:
    # swing_upside: エントリーからの想定上値余地（%）
    if swing_upside is None or not np.isfinite(swing_upside):
        return "不明"

    if swing_upside < 0.025:
        return "小さめ波（無理して触らない方が良い）"
    if swing_upside < 0.06:
        return "3〜4%回転波"
    if swing_upside < 0.12:
        return "7〜8%スイング波"
    return "トレンド大波候補"


# ============================================================
# 三階層スコアリング（Quality / Setup / Regime）
# ============================================================
def score_candidate(
    ticker: str,
    name: str,
    sector: str,
    hist: pd.DataFrame,
    score_raw: float,
    mkt_score: int,
    sector_strength: Dict[str, float],
) -> Dict:
    close = hist["Close"].astype(float)
    price = float(close.iloc[-1])

    ma5 = calc_ma(close, 5)
    ma20 = calc_ma(close, 20)
    ma60 = calc_ma(close, 60)
    rsi = calc_rsi(close, 14)
    atr = calc_atr(hist)
    vola20 = calc_volatility(close, 20)

    volume = hist["Volume"].astype(float) if "Volume" in hist.columns else None
    vwap20 = calc_vwap(close, volume, 20) if volume is not None else price

    # Quality（ベースは ACDE + 伸び代/ボラ微調整）
    quality_score = float(score_raw)

    if price > 0 and vwap20 > 0:
        upside_vwap = (vwap20 * 1.05 / price) - 1.0
        if upside_vwap > 0.05:
            quality_score += 3.0
        elif upside_vwap < 0.02:
            quality_score -= 2.0

    # Setup（短期の形・テクニカル）
    setup_score = 0.0

    # 1. トレンド方向（MAの並び）
    if ma5 > ma20 > ma60:
        setup_score += 12.0
    elif ma20 > ma5 > ma60:
        setup_score += 6.0
    elif ma20 > ma60 > ma5:
        setup_score += 3.0

    # 2. RSI
    if 45 <= rsi <= 65:
        setup_score += 10.0
    elif 40 <= rsi < 45 or 65 < rsi <= 70:
        setup_score += 3.0
    else:
        setup_score -= 6.0

    # 3. ボラ・ATR
    if vola20 < 0.015:
        setup_score += 3.0
    elif 0.015 <= vola20 <= 0.04:
        setup_score += 6.0
    else:
        setup_score -= 1.0

    if atr and price > 0:
        atr_ratio = atr / price
        if 0.012 <= atr_ratio <= 0.035:
            setup_score += 6.0
        elif atr_ratio < 0.008 or atr_ratio > 0.06:
            setup_score -= 5.0

    # 4. 出来高
    if volume is not None and len(volume) >= 20:
        v_ma = float(volume.rolling(20).mean().iloc[-1])
        v_now = float(volume.iloc[-1])
        if v_ma > 0:
            ratio = v_now / v_ma
            if ratio >= 1.5:
                setup_score += 4.0
            elif ratio <= 0.5:
                setup_score -= 4.0

    # Regime（地合い・セクター）
    regime_score = 0.0
    regime_score += (mkt_score - 50) * 0.12
    if sector_strength:
        regime_score += sector_strength.get(sector, 0.0)

    # 三階層を合成
    wQ, wS, wR = get_score_weights(mkt_score)
    total_score = quality_score * wQ + setup_score * wS + regime_score * wR

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "price": price,
        "score_quality": quality_score,
        "score_setup": setup_score,
        "score_regime": regime_score,
        "score_final": float(total_score),
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60,
        "rsi": rsi,
        "atr": atr,
        "vola20": vola20,
        "vwap20": vwap20,
        "hist": hist,
    }


# ============================================================
# IN価格ロジック
# ============================================================
def compute_entry_price(
    close: pd.Series,
    ma5: float,
    ma20: float,
    atr: float,
) -> float:
    # 今日から3〜10日スイングで勝ちやすい IN価格
    price = float(close.iloc[-1])
    last_low = float(close.iloc[-5:].min())

    target = ma20

    if atr and atr > 0:
        target = target - atr * 0.5

    if price > ma5 > ma20:
        target = ma20 + (ma5 - ma20) * 0.3

    if target > price:
        target = price * 0.995

    if target < last_low:
        target = last_low * 1.02

    return round(float(target), 1)


# ============================================================
# TP / SL ロジック
# ============================================================
def calc_candidate_tp_sl(
    vola20: float,
    mkt_score: int,
    atr_ratio: Optional[float],
    swing_upside: Optional[float],
) -> Tuple[float, float]:
    # ボラ・地合い・ATR・伸び代から利確 / 損切り
    v = abs(vola20) if np.isfinite(vola20) else 0.03
    ar = abs(atr_ratio) if (atr_ratio is not None and np.isfinite(atr_ratio)) else 0.02

    if v < 0.015 and ar < 0.015:
        tp = 0.035
        sl = -0.02
    elif v < 0.03 and ar < 0.03:
        tp = 0.05
        sl = -0.025
    else:
        tp = 0.08
        sl = -0.035

    if mkt_score >= 70:
        tp += 0.02
    elif mkt_score < 45:
        tp -= 0.01
        sl = max(sl, -0.025)

    if swing_upside is not None and np.isfinite(swing_upside) and swing_upside > 0:
        max_realistic = swing_upside * 0.9
        if tp > max_realistic:
            tp = max(0.025, max_realistic)

    tp = float(np.clip(tp, 0.025, 0.12))
    sl = float(np.clip(sl, -0.05, -0.015))

    return tp, sl


# ============================================================
# 地合い補正付きマクロスコア
# ============================================================
def enhance_market_score() -> Dict:
    base = calc_market_score()
    if isinstance(base, dict):
        score = float(base.get("score", 50))
        comment = str(base.get("comment", ""))
        info = dict(base)
    else:
        score = float(base)
        comment = ""
        info = {"score": int(score), "comment": comment}

    score = float(np.clip(score, 0.0, 100.0))

    # 日経
    try:
        nikkei = yf.Ticker("^N225").history(period="6d")
        if nikkei is not None and not nikkei.empty and len(nikkei) >= 2:
            n_chg = float(nikkei["Close"].iloc[-1] / nikkei["Close"].iloc[0] - 1.0) * 100.0
            score += float(np.clip(n_chg / 2.5, -6.0, 6.0))
    except Exception as e:
        print("[WARN] ^N225 fetch failed:", e)

    # SOX
    try:
        sox = yf.Ticker("^SOX").history(period="6d")
        if sox is not None and not sox.empty and len(sox) >= 2:
            sox_chg = float(sox["Close"].iloc[-1] / sox["Close"].iloc[0] - 1.0) * 100.0
            score += float(np.clip(sox_chg / 3.0, -5.0, 5.0))
    except Exception as e:
        print("[WARN] ^SOX fetch failed:", e)

    # NVDA
    try:
        nvda = yf.Ticker("NVDA").history(period="6d")
        if nvda is not None and not nvda.empty and len(nvda) >= 2:
            nvda_chg = float(nvda["Close"].iloc[-1] / nvda["Close"].iloc[0] - 1.0) * 100.0
            score += float(np.clip(nvda_chg / 4.0, -4.0, 4.0))
    except Exception as e:
        print("[WARN] NVDA fetch failed:", e)

    # 為替 USDJPY（JPY=X）
    try:
        fx = yf.Ticker("JPY=X").history(period="6d")
        if fx is not None and not fx.empty and len(fx) >= 2:
            fx_chg = float(fx["Close"].iloc[-1] / fx["Close"].iloc[0] - 1.0) * 100.0
            score += float(np.clip(fx_chg / 4.0, -3.0, 3.0))
    except Exception as e:
        print("[WARN] FX JPY=X fetch failed:", e)

    score = float(np.clip(round(score), 0, 100))
    info["score"] = int(score)
    if not info.get("comment"):
        if score >= 70:
            info["comment"] = "リスクオン寄り（押し目＋強いテーマに資金集中）"
        elif score >= 50:
            info["comment"] = "中立〜やや追い風（本命押し目のみ厳選）"
        elif score >= 40:
            info["comment"] = "やや逆風（ロット控えめ、ポジション数も絞る）"
        else:
            info["comment"] = "リスクオフ気味（基本は様子見〜縮小）"
    return info


# ============================================================
# スクリーニング（Top15 → 最終5）
# ============================================================
def run_screening(today: datetime.date, mkt_score: int) -> List[Dict]:
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return []

    min_score = dynamic_min_score(mkt_score)
    sector_strength = build_sector_strength_map()

    raw_candidates: List[Dict] = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        base_score = score_stock(hist)
        if base_score is None or not np.isfinite(base_score):
            continue

        if base_score < min_score:
            continue

        info = score_candidate(
            ticker=ticker,
            name=name,
            sector=sector,
            hist=hist,
            score_raw=base_score,
            mkt_score=mkt_score,
            sector_strength=sector_strength,
        )
        raw_candidates.append(info)

    raw_candidates.sort(key=lambda x: x["score_final"], reverse=True)
    topN = raw_candidates[:SCREENING_TOP_N]

    final_list: List[Dict] = []
    for c in topN:
        close = c["hist"]["Close"].astype(float)
        entry = compute_entry_price(close, c["ma5"], c["ma20"], c["atr"])

        price_now = float(c["price"])
        atr_ratio = (c["atr"] / price_now) if (price_now > 0 and c["atr"] is not None and c["atr"] > 0) else None

        # 伸び代：直近20日の高値
        if len(close) >= 20 and entry > 0:
            swing_high = float(close.tail(20).max())
            swing_upside = (swing_high / entry - 1.0) if swing_high > entry else None
        else:
            swing_upside = None

        tp_pct, sl_pct = calc_candidate_tp_sl(c["vola20"], mkt_score, atr_ratio, swing_upside)
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 + sl_pct)

        gap_ratio = abs(price_now - entry) / price_now if price_now > 0 else 1.0
        entry_type = "today" if gap_ratio <= 0.01 else "soon"

        wave_label = classify_wave(swing_upside)

        final_list.append(
            {
                "ticker": c["ticker"],
                "name": c["name"],
                "sector": c["sector"],
                "score": c["score_final"],
                "price": price_now,
                "entry": entry,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "entry_type": entry_type,
                "wave_label": wave_label,
                "swing_upside": float(swing_upside) if swing_upside is not None else None,
            }
        )

    final_list.sort(key=lambda x: x["score"], reverse=True)
    return final_list[:MAX_FINAL_STOCKS]


# ============================================================
# ロット計算（RR / RISK_PER_TRADE ベース）
# ============================================================
def calc_recommended_size(
    total_asset: float,
    mkt_score: int,
    core_list: List[Dict],
) -> None:
    # core_list に推奨株数 / RR を埋め込む
    if not (np.isfinite(total_asset) and total_asset > 0):
        return

    rpt = risk_per_trade(mkt_score)
    risk_amount = total_asset * rpt

    for c in core_list:
        entry = float(c["entry"])
        sl = float(c["sl_price"])
        if entry <= 0:
            c["size"] = None
            c["risk_amount"] = None
            c["rr"] = None
            continue

        stop_distance = entry - sl
        if stop_distance <= 0:
            c["size"] = None
            c["risk_amount"] = None
            c["rr"] = None
            continue

        raw_shares = risk_amount / stop_distance
        shares_100 = int(raw_shares // 100 * 100)
        if shares_100 <= 0:
            c["size"] = None
            c["risk_amount"] = None
            c["rr"] = None
            continue

        c["size"] = shares_100
        c["risk_amount"] = stop_distance * shares_100
        tp = float(c["tp_price"])
        rr = (tp - entry) / stop_distance if stop_distance > 0 else None
        c["rr"] = float(rr) if rr is not None and np.isfinite(rr) else None


# ============================================================
# 今日は触る日か / 休む日か
# ============================================================
def judge_day_mode(core_list: List[Dict]) -> str:
    if not core_list:
        return "rest"

    today_list = [c for c in core_list if c.get("entry_type") == "today"]
    if today_list:
        return "trade"

    soon_list = [c for c in core_list if c.get("entry_type") == "soon"]
    big_waves = [
        c for c in soon_list
        if c.get("swing_upside") is not None and c["swing_upside"] >= 0.06
    ]

    if big_waves:
        return "wait_big_wave"

    return "rest"


# ============================================================
# レポート構築
# ============================================================
def build_report(
    today_str: str,
    today_date: datetime.date,
    mkt: Dict,
    total_asset: float,
    pos_text: str,
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))

    rec_lev, lev_comment = recommend_leverage(mkt_score)
    est_asset = total_asset if np.isfinite(total_asset) and total_asset > 0 else 2_000_000.0
    est_asset_int = int(round(est_asset))
    max_pos = calc_max_position(est_asset, rec_lev)

    # セクター
    secs = top_sectors_5d()
    if secs:
        sec_lines = []
        for i, row in enumerate(secs[:3]):
            if len(row) >= 2:
                name, chg5 = row[0], row[1]
            else:
                continue
            sec_lines.append(f"{i + 1}. {name} ({chg5:+.2f}%)")
        sec_text = "\n".join(sec_lines)
    else:
        sec_text = "算出不可（データ不足）"

    # イベント
    event_lines = build_event_warnings(today_date)
    if not event_lines:
        event_lines = ["- 特筆すべきイベントなし（通常モード）"]

    # スクリーニング
    core_list = run_screening(today_date, mkt_score)
    calc_recommended_size(total_asset, mkt_score, core_list)

    today_list = [c for c in core_list if c.get("entry_type") == "today"]
    soon_list = [c for c in core_list if c.get("entry_type") == "soon"]

    day_mode = judge_day_mode(core_list)

    lines: List[str] = []

    # ヘッダー
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- 推奨レバ: 約{rec_lev:.1f}倍（{lev_comment}）")
    lines.append(f"- 運用資産想定: 約{est_asset_int:,}円")
    lines.append(f"- 同時最大本命銘柄数: {MAX_CORE_POSITIONS}銘柄")
    lines.append("")

    # 今日の姿勢
    if day_mode == "rest":
        lines.append("◆ 今日の姿勢")
        lines.append("今日は休む日。波を待つ。")
        lines.append("無理に触っても未来の7%は出てこない。")
        lines.append("強者: 波が無い日は休む事で勝っている。")
        lines.append("弱者: “なんか取れそうな銘柄” を探してしまう。")
        lines.append("")
    elif day_mode == "wait_big_wave":
        lines.append("◆ 今日の姿勢")
        lines.append("今日は“次の大きな波を待ち構える日”。")
        lines.append("数日以内に取りに行く7〜8%波候補を監視。")
        lines.append("無理なINはせず、INゾーンまで引きつける。")
        lines.append("")

    # セクター
    lines.append("◆ 今日のTOPセクター（5日騰落）")
    lines.append(sec_text)
    lines.append("")

    # イベント
    lines.append("◆ 今日のイベント・警戒")
    for ev in event_lines:
        lines.append(ev)
    lines.append("")

    lines.append("※寄り付きが INゾーン上限より +1.5%以上高い場合は、その日は見送り推奨")
    lines.append("")

    # 今日IN候補
    lines.append(f"◆ Core候補 Aランク（今日IN候補 最大{MAX_FINAL_STOCKS}）")
    if not today_list:
        lines.append("今日INできる本命候補なし")
    else:
        for c in today_list:
            size_txt = f"{int(c['size'])}株" if c.get("size") else "ロット:算出不可"
            rr_txt = f"{c['rr']:.1f}R" if c.get("rr") is not None else "RR: -"
            lines.append(
                f"- {c['ticker']} {c['name']} Score:{c['score']:.1f} 現値:{c['price']:.1f} [{c['sector']}]"
            )
            lines.append(f"    ・INゾーン: {c['entry']*0.995:.1f}〜{c['entry']*1.01:.1f}（中心{c['entry']:.1f}）")
            lines.append(
                f"    ・利確:+{c['tp_pct']*100:.1f}%（{c['tp_price']:.1f}） 損切:{c['sl_pct']*100:.1f}%（{c['sl_price']:.1f}） {rr_txt}"
            )
            if c.get("wave_label"):
                lines.append(f"    ・波のサイズ: {c['wave_label']}")
            lines.append(f"    ・推奨: {size_txt}")
            lines.append("")

    # 数日以内IN候補
    lines.append("◆ Core候補 Aランク（数日以内IN候補）")
    if not soon_list:
        lines.append("数日以内に狙う本命候補なし")
    else:
        for c in soon_list:
            size_txt = f"{int(c['size'])}株" if c.get("size") else "ロット:算出不可"
            rr_txt = f"{c['rr']:.1f}R" if c.get("rr") is not None else "RR: -"
            lines.append(
                f"- {c['ticker']} {c['name']} Score:{c['score']:.1f} 現値:{c['price']:.1f} [{c['sector']}]"
            )
            lines.append(f"    ・理想IN: {c['entry']:.1f}")
            lines.append(
                f"    ・利確:+{c['tp_pct']*100:.1f}% 損切:{c['sl_pct']*100:.1f}% {rr_txt}"
            )
            if c.get("wave_label"):
                lines.append(f"    ・波のサイズ: {c['wave_label']}")
            lines.append(f"    ・推奨: {size_txt}")
            lines.append("")

    # 建て玉
    lines.append("◆ 本日の建て玉最大金額")
    lines.append(f"- 推奨レバ: {rec_lev:.1f}倍")
    lines.append(f"- MAX建て玉: 約{max_pos:,}円")
    lines.append("")

    # ポジション
    lines.append(f"📊 {today_str} ポジション分析")
    lines.append("")
    lines.append("◆ ポジションサマリ")
    lines.append(pos_text.strip())

    long_report = "\n".join(lines)

    # ショート版
    short_lines: List[str] = []
    short_lines.append(f"📅 {today_str} stockbotTOM 要約")
    short_lines.append(f"- 地合い: {mkt_score} / レバ目安: {rec_lev:.1f}倍")
    short_lines.append(f"- 同時最大本命銘柄数: {MAX_CORE_POSITIONS}銘柄")

    if core_list:
        best = core_list[0]
        rr_txt = f"{best['rr']:.1f}R" if best.get("rr") is not None else "RR:-"
        short_lines.append(
            f"- 本命: {best['ticker']} {best['name']} Score:{best['score']:.1f} [{best['sector']}]"
        )
        short_lines.append(
            f"  IN:{best['entry']:.1f} TP:+{best['tp_pct']*100:.1f}% SL:{best['sl_pct']*100:.1f}% {rr_txt}"
        )
    else:
        short_lines.append("- 本命: 今日は休む日（波待ちモード）")

    short_lines.append(f"- MAX建て玉: 約{max_pos:,}円")

    if day_mode == "rest":
        short_lines.append("")
        short_lines.append("今日は休む日。波を待つ事で未来の7%を作る日。")
        short_lines.append("弱者: “なんか取れそう” を探す / 強者: 休む事で勝つ。")

    short_report = "\n".join(short_lines)

    return long_report + "\n\n-----\n\n" + short_report


# ============================================================
# LINE送信（Cloudflare Worker 経由・分割対応）
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL が未設定（print のみ）")
        print(text)
        return

    chunk_size = 3900
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=15)
            print("[LINE RESULT]", r.status_code, r.text)
        except Exception as e:
            print("[ERROR] LINE送信に失敗:", e)
            print(ch)


# ============================================================
# Entry
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev, risk_info = analyze_positions(pos_df)

    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        total_asset=total_asset,
        pos_text=pos_text,
    )

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()