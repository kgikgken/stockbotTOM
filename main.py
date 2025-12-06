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


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

SCREENING_TOP_N = 15
MAX_FINAL_STOCKS = 5
EARNINGS_EXCLUDE_DAYS = 3


# ============================================================
# 日付 / イベント
# ============================================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
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


def build_event_warnings(today: datetime.date) -> Tuple[List[str], bool]:
    events = load_events()
    warns: List[str] = []
    should_rest = False
    for ev in events:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        kind = ev.get("kind", "").lower()
        delta = (d - today).days

        # 表示用
        if -1 <= delta <= 2:
            if delta > 0:
                when = f"{delta}日後"
            elif delta == 0:
                when = "本日"
            else:
                when = "直近"
            warns.append(f"⚠ {ev['label']}（{when}）: ポジションサイズ注意")

        # 休む判定（ざっくり）
        if kind in ("macro", "boj", "fomc", "cpi", "jobs"):
            if delta in (-1, 0):
                should_rest = True

    if not warns:
        warns = ["- 特筆すべきイベントなし（通常モード）"]
    return warns, should_rest


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
# テクニカル
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


def calc_vwap_20(hist: pd.DataFrame) -> Optional[float]:
    if not {"Close", "Volume"}.issubset(hist.columns) or len(hist) < 5:
        return None
    tail = hist.tail(20)
    vol = tail["Volume"].astype(float)
    close = tail["Close"].astype(float)
    denom = vol.sum()
    if denom <= 0:
        return None
    vwap = float((close * vol).sum() / denom)
    return vwap


# ============================================================
# レバ・リスク
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


def calc_risk_per_trade(mkt_score: int) -> float:
    if mkt_score >= 70:
        return 0.018
    if mkt_score >= 60:
        return 0.013
    if mkt_score >= 50:
        return 0.010
    return 0.007


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
# セクター強度 & スコア重み
# ============================================================
def build_sector_strength_map() -> Dict[str, float]:
    secs = top_sectors_5d()
    strength: Dict[str, float] = {}
    for rank, sec in enumerate(secs[:5]):
        if len(sec) >= 2:
            name = sec[0]
            chg5 = float(sec[1])
        else:
            continue
        base = 6 - rank
        boost = max(chg5, 0.0) * 0.3
        strength[str(name)] = base + boost
    return strength


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
# 三階層スコア
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
    vwap20 = calc_vwap_20(hist)

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

    if vola20 < 0.012:
        setup_score -= 3.0
    elif vola20 < 0.02:
        setup_score += 5.0
    elif vola20 > 0.06:
        setup_score -= 4.0
    else:
        setup_score += 2.0

    atr_ratio = 0.0
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
        "atr_ratio": atr_ratio,
        "vola20": vola20,
        "vwap20": vwap20,
        "hist": hist,
    }


# ============================================================
# IN / TP / SL / 伸び代 / ポジションサイズ
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


def calc_swing_upside(
    close: pd.Series,
    entry: float,
    vwap20: Optional[float],
) -> Optional[float]:
    if entry <= 0:
        return None
    local_high = float(close.tail(20).max())
    candidates = [local_high]
    if vwap20 is not None and np.isfinite(vwap20):
        candidates.append(float(vwap20))
    best = max(candidates) if candidates else None
    if best is None or best <= entry:
        return None
    return best / entry - 1.0


def calc_position_size(
    total_asset: float,
    risk_per_trade: float,
    entry: float,
    sl_price: float,
    tp_price: float,
) -> Tuple[int, float, float]:
    if not (np.isfinite(total_asset) and total_asset > 0):
        return 0, 0.0, 0.0
    if risk_per_trade <= 0 or entry <= 0 or sl_price <= 0:
        return 0, 0.0, 0.0
    per_share_risk = abs(entry - sl_price)
    if per_share_risk <= 0:
        return 0, 0.0, 0.0
    max_loss = total_asset * risk_per_trade
    raw_shares = max_loss / per_share_risk
    lots = int(raw_shares // 100)
    shares = lots * 100
    if shares <= 0:
        return 0, 0.0, 0.0
    est_loss = per_share_risk * shares
    reward_per_share = max(tp_price - entry, 0.0)
    est_gain = reward_per_share * shares
    return shares, est_loss, est_gain


# ============================================================
# 地合い（SOX/NVDA/FX込み）
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
# スクリーニング本体
# ============================================================
def run_screening(
    today: datetime.date,
    mkt_score: int,
    total_asset: float,
    risk_per_trade: float,
) -> List[Dict]:
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
        vwap20 = c.get("vwap20")
        swing_upside = calc_swing_upside(close, entry, vwap20)

        tp_pct, sl_pct = calc_candidate_tp_sl(c["vola20"], mkt_score, atr_ratio, swing_upside)
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 + sl_pct)

        rr = 0.0
        denom = abs(sl_pct)
        if denom > 0:
            rr = tp_pct / denom

        shares, est_loss, est_gain = calc_position_size(
            total_asset=total_asset,
            risk_per_trade=risk_per_trade,
            entry=entry,
            sl_price=sl_price,
            tp_price=tp_price,
        )

        price_now = float(c["price"])
        gap_ratio = abs(price_now - entry) / price_now if price_now > 0 else 1.0
        entry_type = "today" if gap_ratio <= 0.01 else "soon"

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
                "rr": rr,
                "entry_type": entry_type,
                "shares": shares,
                "est_loss": est_loss,
                "est_gain": est_gain,
            }
        )

    final_list.sort(key=lambda x: x["score"], reverse=True)
    return final_list[:MAX_FINAL_STOCKS]


# ============================================================
# 既存ポジション RR → 乗り換え提案
# ============================================================
def calc_position_rr_for_existing(
    ticker: str,
    avg_price: float,
    mkt_score: int,
) -> Optional[float]:
    hist = fetch_history(ticker)
    if hist is None or len(hist) < 60:
        return None
    close = hist["Close"].astype(float)
    price_now = float(close.iloc[-1])
    ma5 = calc_ma(close, 5)
    ma20 = calc_ma(close, 20)
    atr = calc_atr(hist)
    vola20 = calc_volatility(close, 20)
    vwap20 = calc_vwap_20(hist)
    atr_ratio = (atr / price_now) if (price_now > 0 and atr is not None and atr > 0) else None
    swing_upside = calc_swing_upside(close, avg_price, vwap20)
    tp_pct, sl_pct = calc_candidate_tp_sl(
        vola20=vola20,
        mkt_score=mkt_score,
        atr_ratio=atr_ratio,
        swing_upside=swing_upside,
    )
    denom = abs(sl_pct)
    if denom <= 0:
        return None
    return float(tp_pct / denom)


def suggest_position_switch(
    pos_df: Optional[pd.DataFrame],
    mkt_score: int,
    best_candidate: Optional[Dict],
) -> List[str]:
    lines: List[str] = []
    if pos_df is None or pos_df.empty or not best_candidate:
        return lines
    best_rr = best_candidate.get("rr")
    if best_rr is None or not np.isfinite(best_rr) or best_rr <= 0:
        return lines
    THRESH = 0.8
    for _, row in pos_df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue
        if "avg_price" in row:
            avg_price = float(row["avg_price"])
        elif "entry_price" in row:
            avg_price = float(row["entry_price"])
        else:
            continue
        current_rr = calc_position_rr_for_existing(ticker, avg_price, mkt_score)
        if current_rr is None or not np.isfinite(current_rr):
            continue
        if best_rr - current_rr >= THRESH:
            lines.append(
                f"- {ticker}: 現在RR:{current_rr:.1f}R → 本命 {best_candidate['ticker']} ({best_rr:.1f}R) への乗り換え候補"
            )
    return lines


# ============================================================
# レポート構築
# ============================================================
def build_report(
    today_str: str,
    today_date: datetime.date,
    mkt: Dict,
    total_asset: float,
    pos_text: str,
    pos_df: Optional[pd.DataFrame],
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))
    rec_lev, lev_comment = recommend_leverage(mkt_score)
    risk_per_trade = calc_risk_per_trade(mkt_score)

    est_asset = total_asset if np.isfinite(total_asset) and total_asset > 0 else 2_000_000.0
    est_asset_int = int(round(est_asset))
    max_pos = calc_max_position(est_asset, rec_lev)

    secs = top_sectors_5d()
    if secs:
        sec_lines = []
        for i, sec in enumerate(secs[:5]):
            if len(sec) >= 2:
                name = sec[0]
                chg5 = float(sec[1])
                sec_lines.append(f"{i + 1}. {name} ({chg5:+.2f}%)")
        sec_text = "\n".join(sec_lines) if sec_lines else "算出不可（データ不足）"
    else:
        sec_text = "算出不可（データ不足）"

    event_lines, should_rest = build_event_warnings(today_date)

    core_list = run_screening(today_date, mkt_score, est_asset, risk_per_trade)
    today_list = [c for c in core_list if c.get("entry_type") == "today"]
    soon_list = [c for c in core_list if c.get("entry_type") == "soon"]

    best = core_list[0] if core_list else None
    switch_lines = suggest_position_switch(pos_df, mkt_score, best)

    lines: List[str] = []

    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- 推奨レバ: 約{rec_lev:.1f}倍（{lev_comment}）")
    lines.append(f"- 運用資産想定: 約{est_asset_int:,}円")
    lines.append("- 同時最大本命銘柄数: 3銘柄")
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
    if should_rest:
        lines.append("※重要イベント前後のため、今日は基本『休む日』寄り。新規INは慎重に。")
    if not today_list:
        lines.append("今日INできる本命候補なし")
    else:
        for c in today_list:
            lines.append(
                f"- {c['ticker']} {c['name']} Score:{c['score']:.1f} 現値:{c['price']:.1f} [{c['sector']}]"
            )
            lines.append(
                f"    ・INゾーン: {c['entry'] * 0.995:.1f}〜{c['entry'] * 1.009:.1f}（中心{c['entry']:.1f}）"
            )
            lines.append(
                f"    ・利確:+{c['tp_pct']*100:.1f}%（{c['tp_price']:.1f}） 損切:{c['sl_pct']*100:.1f}%（{c['sl_price']:.1f}） RR:{c['rr']:.1f}R"
            )
            if c["shares"] > 0:
                notion = int(round(c["shares"] * c["entry"]))
                lines.append(
                    f"    ・推奨: {c['shares']}株 ≒{notion:,}円 / 損失~{int(c['est_loss']):,}円 利確~{int(c['est_gain']):,}円"
                )
            lines.append("")

    lines.append("◆ Core候補 Aランク（数日以内IN候補）")
    if not soon_list:
        lines.append("数日以内IN候補なし")
    else:
        for c in soon_list:
            lines.append(
                f"- {c['ticker']} {c['name']} Score:{c['score']:.1f} 現値:{c['price']:.1f} [{c['sector']}]"
            )
            lines.append(
                f"    ・理想IN: {c['entry']:.1f} ゾーン:{c['entry'] * 0.995:.1f}〜{c['entry'] * 1.009:.1f}"
            )
            lines.append(
                f"    ・利確:+{c['tp_pct']*100:.1f}% 損切:{c['sl_pct']*100:.1f}% RR:{c['rr']:.1f}R"
            )
            if c["shares"] > 0:
                notion = int(round(c["shares"] * c["entry"]))
                lines.append(
                    f"    ・推奨: {c['shares']}株 ≒{notion:,}円 / 損失~{int(c['est_loss']):,}円 利確~{int(c['est_gain']):,}円"
                )
            lines.append("")

    lines.append("◆ 本日の建て玉最大金額")
    lines.append(f"- 推奨レバ: {rec_lev:.1f}倍 / MAX建て玉: 約{max_pos:,}円")
    lines.append("")

    lines.append(f"📊 {today_str} ポジション分析")
    lines.append("")
    lines.append("◆ ポジションサマリ")
    lines.append(pos_text.strip() if pos_text else "ポジション情報なし")
    lines.append("")

    lines.append("◆ ポジション入れ替え候補（RRベース）")
    if switch_lines:
        lines.extend(switch_lines)
    else:
        lines.append("- 本日の時点で明確な乗り換え候補なし（RR差が小さいため）")
    lines.append("")

    lines.append("◆ 今日の行動メモ")
    if today_list and not should_rest:
        lines.append("今日は『攻めてもよい波』が出ている日。ただしINは必ずゾーン内＆ロットはルール通り。")
    else:
        lines.append("今日は休む日。波を待つ。")
        lines.append("無理に触っても未来の7は出てこない。")
        lines.append("")
        lines.append("＝これが強者の行動。")
        lines.append("")
        lines.append("弱者は：")
        lines.append("“なんか取れそうな銘柄探しちゃう”")
        lines.append("")
        lines.append("強者は：")
        lines.append("“波が無い日は休む事で勝ってる”")
        lines.append("")
        lines.append("この違いだけで 資産曲線が別世界になる。")
        lines.append("")
        lines.append("今日の出力はその判断を完全にサポートしてる。")

    long_report = "\n".join(lines)

    short_lines: List[str] = []
    short_lines.append(f"📅 {today_str} stockbotTOM 要約")
    short_lines.append(f"- 地合い: {mkt_score} / レバ目安: {rec_lev:.1f}倍")
    short_lines.append(f"- MAX建て玉: 約{max_pos:,}円")

    if today_list:
        best = today_list[0]
    elif core_list:
        best = core_list[0]
    else:
        best = None

    if best:
        short_lines.append(
            f"- 本命: {best['ticker']} {best['name']} Score:{best['score']:.1f} [{best['sector']}]"
        )
        short_lines.append(
            f"  IN:{best['entry']:.1f} RR:{best['rr']:.1f}R TP:+{best['tp_pct']*100:.1f}% SL:{best['sl_pct']*100:.1f}%"
        )
    else:
        short_lines.append("- 本命: なし（今日は無理に攻めない日）")

    if not today_list or should_rest:
        short_lines.append("- 判定: 今日は休む日寄り。波が来るまでキャッシュ温存が“強者の行動”。")

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
# main
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
        pos_df=pos_df,
    )

    print(report)

    # GitHub Actions で事前生成する時は NO_LINE_SEND=1 をセットして送信スキップ
    if os.getenv("NO_LINE_SEND") != "1":
        send_line(report)


if __name__ == "__main__":
    main()