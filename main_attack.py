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
SCREENING_TOP_N = 15           # まずは Top15 まで抽出
MAX_FINAL_STOCKS = 5           # 最終的に LINE に出すのは最大5銘柄

# 決算フィルタ: ±N日
EARNINGS_EXCLUDE_DAYS = 3


# ============================================================
# 日付 / イベント関連
# ============================================================
def jst_today_date() -> datetime.date:
    """JST の今日の日付"""
    return datetime.now(timezone(timedelta(hours=9))).date()


def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    """events.csv -> [{date,label,kind}, ...] を返す（無ければ []）。"""
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
    """通常の日報用：2日前〜翌日までを警戒表示。"""
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


def detect_event_risk(today: datetime.date) -> List[str]:
    """縮小レベルのイベントのみ抽出（当日のマクロ系など）。"""
    events = load_events()
    today_str = today.strftime("%Y-%m-%d")
    msgs: List[str] = []
    for ev in events:
        if ev.get("date") != today_str:
            continue
        kind = str(ev.get("kind", "")).lower()
        if kind in ("macro", "event", "risk") or "fomc" in ev.get("label", "").lower():
            msgs.append(f"イベント: {ev.get('label', '')}")
    return msgs


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

    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


def in_earnings_window(row: pd.Series, today: datetime.date) -> bool:
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    try:
        delta = abs((d - today).days)
    except Exception:
        return False
    return delta <= EARNINGS_EXCLUDE_DAYS


def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    """株価履歴取得（簡易リトライ付き）。"""
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


# ============================================================
# レバレッジ / 建て玉
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
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


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# 動的な最低スコアライン（地合い連動）
# ============================================================
def dynamic_min_score(mkt_score: int) -> float:
    if mkt_score >= 75:
        return 70.0
    if mkt_score >= 65:
        return 73.0
    if mkt_score >= 55:
        return 76.0
    if mkt_score >= 45:
        return 79.0
    return 82.0


# ============================================================
# セクター強度（Top5をブースト）
# ============================================================
def build_sector_strength_map() -> Dict[str, float]:
    secs = top_sectors_5d()
    strength: Dict[str, float] = {}

    for rank, (name, chg) in enumerate(secs[:5]):
        base = 6 - rank  # 1位:6, 2位:5, ...
        boost = max(chg, 0.0) * 0.3
        strength[name] = base + boost

    return strength


# ============================================================
# 三階層スコアの重み（地合いで可変）
# ============================================================
def get_score_weights(mkt_score: int) -> Tuple[float, float, float]:
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
# Top候補用の三階層スコアリング
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

    quality_score = float(score_raw)

    setup_score = 0.0

    if ma5 > ma20 > ma60:
        setup_score += 12.0
    elif ma20 > ma5 > ma60:
        setup_score += 6.0
    elif ma20 > ma60 > ma5:
        setup_score += 3.0

    if 40 <= rsi <= 65:
        setup_score += 10.0
    elif 30 <= rsi < 40 or 65 < rsi <= 70:
        setup_score += 3.0
    else:
        setup_score -= 6.0

    if vola20 < 0.02:
        setup_score += 5.0
    elif vola20 > 0.05:
        setup_score -= 4.0

    if atr and price > 0:
        atr_ratio = atr / price
        if 0.015 <= atr_ratio <= 0.035:
            setup_score += 6.0
        elif atr_ratio < 0.01 or atr_ratio > 0.06:
            setup_score -= 5.0

    if "Volume" in hist.columns:
        vol = hist["Volume"].astype(float)
        if len(vol) >= 20:
            v_ma = float(vol.rolling(20).mean().iloc[-1])
            v_now = float(vol.iloc[-1])
            if v_ma > 0:
                ratio = v_now / v_ma
                if ratio >= 1.5:
                    setup_score += 3.0
                elif ratio <= 0.5:
                    setup_score -= 3.0

    regime_score = 0.0
    regime_score += (mkt_score - 50) * 0.12

    if sector_strength:
        regime_score += sector_strength.get(sector, 0.0)

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
# TP / SL ロジック & RR計算
# ============================================================
def calc_candidate_tp_sl(
    vola20: float,
    mkt_score: int,
    atr_ratio: Optional[float],
    swing_upside: Optional[float],
) -> Tuple[float, float]:
    v = abs(vola20) if np.isfinite(vola20) else 0.03
    ar = abs(atr_ratio) if (atr_ratio is not None and np.isfinite(atr_ratio)) else 0.02

    if v < 0.015 and ar < 0.015:
        tp = 0.06
        sl = -0.03
    elif v < 0.03 and ar < 0.03:
        tp = 0.08
        sl = -0.04
    else:
        tp = 0.12
        sl = -0.055

    if mkt_score >= 70:
        tp += 0.02
    elif mkt_score < 45:
        tp -= 0.02
        sl = max(sl, -0.04)

    if swing_upside is not None and np.isfinite(swing_upside) and swing_upside > 0:
        max_realistic = swing_upside * 0.9
        if tp > max_realistic:
            tp = max(0.05, max_realistic)

    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))

    return tp, sl


def compute_rr(tp_pct: float, sl_pct: float) -> Optional[float]:
    if tp_pct <= 0 or sl_pct >= 0:
        return None
    rr = tp_pct / abs(sl_pct)
    if not np.isfinite(rr):
        return None
    return round(float(rr), 1)


# ============================================================
# 地合いスコア拡張（SOX / NVDA / 為替など）
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

    try:
        nikkei = yf.Ticker("^N225").history(period="6d")
        if nikkei is not None and not nikkei.empty and len(nikkei) >= 2:
            n_chg = float(nikkei["Close"].iloc[-1] / nikkei["Close"].iloc[0] - 1.0) * 100.0
            score += float(np.clip(n_chg / 2.5, -6.0, 6.0))
    except Exception as e:
        print("[WARN] ^N225 fetch failed:", e)

    try:
        sox = yf.Ticker("^SOX").history(period="6d")
        if sox is not None and not sox.empty and len(sox) >= 2:
            sox_chg = float(sox["Close"].iloc[-1] / sox["Close"].iloc[0] - 1.0) * 100.0
            score += float(np.clip(sox_chg / 3.0, -5.0, 5.0))
    except Exception as e:
        print("[WARN] ^SOX fetch failed:", e)

    try:
        nvda = yf.Ticker("NVDA").history(period="6d")
        if nvda is not None and not nvda.empty and len(nvda) >= 2:
            nvda_chg = float(nvda["Close"].iloc[-1] / nvda["Close"].iloc[0] - 1.0) * 100.0
            score += float(np.clip(nvda_chg / 4.0, -4.0, 4.0))
    except Exception as e:
        print("[WARN] NVDA fetch failed:", e)

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
# 波の崩壊検知（縮小用）
# ============================================================
def detect_wave_collapse() -> List[str]:
    msgs: List[str] = []

    try:
        nikkei = yf.Ticker("^N225").history(period="6d")
        if nikkei is not None and not nikkei.empty and len(nikkei) >= 2:
            chg = float(nikkei["Close"].iloc[-1] / nikkei["Close"].iloc[0] - 1.0) * 100.0
            if chg <= -2.0:
                msgs.append(f"日経平均 {chg:.1f}%")
    except Exception as e:
        print("[WARN] detect_wave nikkei:", e)

    try:
        sox = yf.Ticker("^SOX").history(period="6d")
        if sox is not None and not sox.empty and len(sox) >= 2:
            chg = float(sox["Close"].iloc[-1] / sox["Close"].iloc[0] - 1.0) * 100.0
            if chg <= -3.0:
                msgs.append(f"SOX {chg:.1f}%")
    except Exception as e:
        print("[WARN] detect_wave sox:", e)

    try:
        nvda = yf.Ticker("NVDA").history(period="6d")
        if nvda is not None and not nvda.empty and len(nvda) >= 2:
            chg = float(nvda["Close"].iloc[-1] / nvda["Close"].iloc[0] - 1.0) * 100.0
            if chg <= -4.0:
                msgs.append(f"NVDA {chg:.1f}%")
    except Exception as e:
        print("[WARN] detect_wave nvda:", e)

    try:
        fx = yf.Ticker("JPY=X").history(period="6d")
        if fx is not None and not fx.empty and len(fx) >= 2:
            chg = float(fx["Close"].iloc[-1] / fx["Close"].iloc[0] - 1.0) * 100.0
            if chg <= -2.0:
                msgs.append(f"USDJPY {chg:.1f}%（急な円高）")
    except Exception as e:
        print("[WARN] detect_wave fx:", e)

    return msgs


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

        price = float(c["price"])
        atr_ratio = (c["atr"] / price) if (price > 0 and c["atr"] is not None and c["atr"] > 0) else None
        if len(close) >= 20 and entry > 0:
            swing_high = float(close.tail(20).max())
            swing_upside = (swing_high / entry - 1.0) if swing_high > entry else None
        else:
            swing_upside = None

        tp_pct, sl_pct = calc_candidate_tp_sl(c["vola20"], mkt_score, atr_ratio, swing_upside)
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 + sl_pct)

        price_now = float(c["price"])
        gap_ratio = abs(price_now - entry) / price_now if price_now > 0 else 1.0
        entry_type = "today" if gap_ratio <= 0.01 else "soon"

        rr = compute_rr(tp_pct, sl_pct)

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
                "rr": rr,
            }
        )

    final_list.sort(key=lambda x: x["score"], reverse=True)
    return final_list[:MAX_FINAL_STOCKS]


# ============================================================
# 既存 analyze_positions の risk_info から RR情報を抜く
# ============================================================
def extract_position_rr_list(risk_info) -> List[Dict[str, float]]:
    """
    risk_info の形がどうであっても、可能な限り
    [{ticker, rr}, ...] のリストに変換する。
    形が合わなければ空リストを返す（安全側）。
    """
    res: List[Dict[str, float]] = []

    def add(ticker, rr):
        try:
            if not ticker:
                return
            v = float(rr)
            if not np.isfinite(v):
                return
            res.append({"ticker": str(ticker), "rr": v})
        except Exception:
            return

    if risk_info is None:
        return res

    # パターン1: list[dict]
    if isinstance(risk_info, list):
        for item in risk_info:
            if not isinstance(item, dict):
                continue
            t = item.get("ticker") or item.get("code") or item.get("symbol")
            rr = item.get("rr") or item.get("RR") or item.get("rr_current")
            if t is not None and rr is not None:
                add(t, rr)
        return res

    # パターン2: dict で positions 配列を持つ
    if isinstance(risk_info, dict):
        positions = risk_info.get("positions") or risk_info.get("detail") or risk_info.get("list")
        if isinstance(positions, list):
            for item in positions:
                if not isinstance(item, dict):
                    continue
                t = item.get("ticker") or item.get("code") or item.get("symbol")
                rr = item.get("rr") or item.get("RR") or item.get("rr_current")
                if t is not None and rr is not None:
                    add(t, rr)
        else:
            # 直接 ticker/rr を持つ単一の dict の可能性
            t = risk_info.get("ticker") or risk_info.get("code") or risk_info.get("symbol")
            rr = risk_info.get("rr") or risk_info.get("RR") or risk_info.get("rr_current")
            if t is not None and rr is not None:
                add(t, rr)

    return res


# ============================================================
# RR乗り換え候補検知
# ============================================================
def detect_rr_swaps(
    pos_rr_list: List[Dict[str, float]],
    core_list: List[Dict],
    threshold: float = 1.0,
) -> List[Dict]:
    if not pos_rr_list or not core_list:
        return []

    best_core = max(core_list, key=lambda x: (x.get("rr") or 0.0))
    best_rr = best_core.get("rr") or 0.0
    if best_rr <= 0:
        return []

    swaps: List[Dict] = []
    for pos in pos_rr_list:
        rr_pos = pos.get("rr")
        if rr_pos is None or not np.isfinite(rr_pos):
            continue
        if best_rr - rr_pos >= threshold:
            swaps.append(
                {
                    "from_ticker": pos.get("ticker", ""),
                    "from_rr": float(rr_pos),
                    "to_ticker": best_core["ticker"],
                    "to_name": best_core["name"],
                    "to_rr": float(best_rr),
                }
            )
    return swaps


# ============================================================
# レポート構築
# ============================================================
def build_report(
    today_str: str,
    today_date: datetime.date,
    mkt: Dict,
    total_asset: float,
    pos_text: str,
    pos_rr_list: List[Dict[str, float]],
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))

    rec_lev, lev_comment = recommend_leverage(mkt_score)
    est_asset = total_asset if np.isfinite(total_asset) and total_asset > 0 else 2_000_000.0
    est_asset_int = int(round(est_asset))
    max_pos = calc_max_position(est_asset, rec_lev)

    secs = top_sectors_5d()
    if secs:
        sec_lines = [
            f"{i + 1}. {name} ({chg:+.2f}%)"
            for i, (name, chg) in enumerate(secs)
        ]
        sec_text = "\n".join(sec_lines)
    else:
        sec_text = "算出不可（データ不足）"

    event_lines = build_event_warnings(today_date)
    if not event_lines:
        event_lines = ["- 特筆すべきイベントなし（通常モード）"]

    core_list = run_screening(today_date, mkt_score)
    today_list = [c for c in core_list if c.get("entry_type") == "today"]
    soon_list = [c for c in core_list if c.get("entry_type") == "soon"]

    wave_msgs = detect_wave_collapse()
    event_risk_msgs = detect_event_risk(today_date)
    rr_swaps = detect_rr_swaps(pos_rr_list, core_list)

    lines: List[str] = []

    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- 推奨レバ: 約{rec_lev:.1f}倍（{lev_comment}）")
    lines.append(f"- 運用資産想定: 約{est_asset_int:,}円")
    lines.append(f"- 同時最大本命銘柄数: 3銘柄")
    lines.append("")
    lines.append("※寄り付きが INゾーン上限より +1.5%以上高い場合は、その日は見送り推奨")
    lines.append("")

    lines.append("◆ 今日のTOPセクター（5日騰落）")
    lines.append(sec_text)
    lines.append("")

    lines.append("◆ 今日のイベント・警戒")
    for ev in event_lines:
        lines.append(ev)
    lines.append("")

    lines.append(f"◆ Core候補 Aランク（今日IN候補 最大{MAX_FINAL_STOCKS}）")
    if not today_list:
        lines.append("今日INできる本命候補なし")
    else:
        for c in today_list:
            rr_txt = f" RR:{c['rr']:.1f}R" if c.get("rr") else ""
            lines.append(
                f"- {c['ticker']} {c['name']} Score:{c['score']:.1f} 現値:{c['price']:.1f} [{c['sector']}]")
            lines.append(
                f"    ・INゾーン: {c['entry']*0.995:.1f}〜{c['entry']*1.010:.1f}（中心{c['entry']:.1f}）")
            lines.append(
                f"    ・利確:+{c['tp_pct']*100:.1f}%（{c['tp_price']:.1f}） 損切:{c['sl_pct']*100:.1f}%（{c['sl_price']:.1f}）{rr_txt}")
            lines.append("")

    lines.append("◆ Core候補 Aランク（数日以内IN候補）")
    if not soon_list:
        lines.append("数日以内IN候補なし")
    else:
        for c in soon_list:
            rr_txt = f" RR:{c['rr']:.1f}R" if c.get("rr") else ""
            lines.append(
                f"- {c['ticker']} {c['name']} Score:{c['score']:.1f} 現値:{c['price']:.1f} [{c['sector']}]")
            lines.append(
                f"    ・理想IN: {c['entry']:.1f} ゾーン:{c['entry']*0.995:.1f}〜{c['entry']*1.010:.1f}")
            lines.append(
                f"    ・利確:+{c['tp_pct']*100:.1f}% 損切:{c['sl_pct']*100:.1f}%{rr_txt}")
            lines.append("")

    lines.append("◆ 本日の建て玉最大金額")
    lines.append(f"- 推奨レバ: {rec_lev:.1f}倍 / MAX建て玉: 約{max_pos:,}円")
    lines.append("")

    lines.append(f"📊 {today_str} ポジション分析")
    lines.append("")
    lines.append("◆ ポジションサマリ")
    lines.append(pos_text.strip())
    lines.append("")

    lines.append("◆ 縮小・撤退アラート（波崩壊・イベント）")
    if not wave_msgs and not event_risk_msgs:
        lines.append("- 特に無し（波継続。基本は維持でOK）")
    else:
        if wave_msgs:
            lines.append("・波崩壊シグナル:")
            for m in wave_msgs:
                lines.append(f"   - {m}")
        if event_risk_msgs:
            lines.append("・イベント由来リスク:")
            for m in event_risk_msgs:
                lines.append(f"   - {m}")
        lines.append("→ 寄りでロット1段階縮小 or 一部利確を検討（攻めるためのリロード）")
    lines.append("")

    lines.append("◆ 乗り換えアラート（RRベース）")
    if not rr_swaps:
        lines.append("- 乗り換え必須レベルのRR差はなし（現ポジ維持でOK）")
    else:
        for s in rr_swaps:
            diff = s['to_rr'] - s['from_rr']
            lines.append(
                f"- {s['from_ticker']}: 現在RR:{s['from_rr']:.1f}R → 本命 {s['to_ticker']} {s['to_name']} (RR:{s['to_rr']:.1f}R, 差:+{diff:.1f}R) への乗り換え候補"
            )
        lines.append("→ 寄りで部分 or 全乗り換えを検討（攻め型）")
    lines.append("")

    long_report = "\n".join(lines)

    short_lines: List[str] = []
    short_lines.append(f"📅 {today_str} stockbotTOM 要約")
    short_lines.append(f"- 地合い: {mkt_score} / レバ目安: {rec_lev:.1f}倍")
    if core_list:
        best = core_list[0]
        rr_txt = f" RR:{best['rr']:.1f}R" if best.get("rr") else ""
        short_lines.append(
            f"- 本命: {best['ticker']} {best['name']} Score:{best['score']:.1f} [{best['sector']}]")
        short_lines.append(
            f"  IN:{best['entry']:.1f} TP:+{best['tp_pct']*100:.1f}% SL:{best['sl_pct']*100:.1f}%{rr_txt}")
    else:
        short_lines.append("- 本命候補なし（今日は無理に攻めない日）")
    short_lines.append(f"- MAX建て玉: 約{max_pos:,}円")

    if rr_swaps:
        s = rr_swaps[0]
        diff = s['to_rr'] - s['from_rr']
        short_lines.append(
            f"- RR乗換候補: {s['from_ticker']} → {s['to_ticker']} (RR差:+{diff:.1f}R)"
        )

    short_report = "\n".join(short_lines)

    return long_report + "\n\n-----\n\n" + short_report


# ============================================================
# LINE送信
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

    # 地合いスコア
    mkt = enhance_market_score()

    # ポジション
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev, risk_info = analyze_positions(pos_df)

    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    pos_rr_list = extract_position_rr_list(risk_info)

    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        total_asset=total_asset,
        pos_text=pos_text,
        pos_rr_list=pos_rr_list,
    )

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()
