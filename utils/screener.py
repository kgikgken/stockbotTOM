from __future__ import annotations

import os
from datetime import date
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import safe_float, clamp
from utils.features import add_indicators, calc_liquidity_flags, calc_atr_pct, estimate_pwin, regime_multiplier
from utils.setup import detect_setup
from utils.entry import compute_entry_zone
from utils.rr_ev import dynamic_min_rr, compute_stop, compute_targets, compute_ev, calc_expected_days
from utils.diversify import pick_with_constraints


# ===== Universe filters =====
PRICE_MIN = 200
PRICE_MAX = 15000
ADV20_MIN = 200_000_000  # 200M JPY/day
ATR_PCT_MIN = 1.5
ATR_PCT_MAX = 6.0

EARNINGS_EXCLUDE_DAYS = 3

WEEKLY_NEW_LIMIT = 3


def _fetch_history(ticker: str, period: str = "320d") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is None or df.empty or len(df) < 120:
            return None
        return df
    except Exception:
        return None


def _parse_universe(universe_path: str) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(universe_path)
    if "ticker" in df.columns:
        tcol = "ticker"
    elif "code" in df.columns:
        tcol = "code"
    else:
        raise ValueError("universe_jpx.csv に ticker/code が必要です")
    return df, tcol


def _filter_earnings(df: pd.DataFrame, today_date: date) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df
    parsed = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    out = df.copy()
    out["_earn"] = parsed
    keep = []
    for d in out["_earn"]:
        if d is None or pd.isna(d):
            keep.append(True)
            continue
        try:
            keep.append(abs((d - today_date).days) > EARNINGS_EXCLUDE_DAYS)
        except Exception:
            keep.append(True)
    return out.loc[keep].drop(columns=["_earn"], errors="ignore")


def _classify_macro(sector: str) -> str:
    """
    ざっくり分類（表示用）
    """
    s = str(sector)
    if any(k in s for k in ["銀行", "保険", "その他金融", "証券"]):
        return "rate_sensitive"
    if any(k in s for k in ["食料", "医薬", "水産", "小売"]):
        return "defensive"
    if any(k in s for k in ["機械", "電気", "金属", "化学", "建設", "輸送"]):
        return "cyclical"
    return "other"


def _min_adjev_cut() -> float:
    # (3) AdjEV < 0.5 を切る
    return 0.50


def _no_trade_reason(market: Dict, macro_caution: bool, weekly_new: int) -> List[str]:
    reasons: List[str] = []
    m = int(market["score"])
    d = int(market["delta3d"])

    # NO-TRADE完全機械化（裁量ゼロ）
    if m < 45:
        reasons.append("地合い悪化(MarketScore<45)")
    if d <= -5 and m < 55:
        reasons.append("地合い急悪化(Δ3d<=-5 & MarketScore<55)")
    if macro_caution:
        reasons.append("イベント接近")
    if weekly_new >= WEEKLY_NEW_LIMIT:
        reasons.append("週次上限到達")

    return reasons[:2]


def _recommend_leverage(market: Dict, macro_caution: bool, no_trade: bool) -> Tuple[float, str]:
    m = int(market["score"])
    d = int(market["delta3d"])

    if no_trade:
        return 1.0, "守り（新規禁止）"

    if macro_caution:
        # 本来は新規禁止だが、念のため表現は統一
        return 1.0, "守り（イベント接近）"

    if m >= 75:
        return 2.0, "強気（押し目＋一部ブレイク）"
    if m >= 65:
        return 1.7, "やや強気（押し目メイン）"
    if m >= 55:
        return 1.3, "中立（厳選・押し目中心）"
    if m >= 50:
        return 1.1, "やや守り（新規ロット小さめ）"
    return 1.0, "守り（新規かなり絞る）"


def _calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


def run_swing_screening(
    universe_path: str,
    today_date: date,
    market: Dict,
    macro_caution: bool,
    weekly_new: int,
) -> Dict:
    """
    return dict:
      - no_trade: bool
      - no_trade_reasons: [..]
      - lev, lev_reason
      - max_final: int
      - candidates: list (selected)
      - watchlist: list (rejected)
      - summary: dict (avg metrics)
    """
    if not os.path.exists(universe_path):
        return {
            "no_trade": True,
            "no_trade_reasons": ["ユニバースファイル無し"],
            "lev": 1.0,
            "lev_reason": "守り（データ不足）",
            "max_final": 0,
            "candidates": [],
            "watchlist": [],
            "summary": {},
        }

    uni, tcol = _parse_universe(universe_path)
    uni = _filter_earnings(uni, today_date)

    mkt_score = int(market["score"])
    delta3d = int(market["delta3d"])

    # (2) イベント時の候補数を最大2
    max_final = 2 if macro_caution else 5

    # NO-TRADE判定
    reasons = _no_trade_reason(market, macro_caution, weekly_new)
    no_trade = len(reasons) > 0  # イベント接近もここで新規禁止

    lev, lev_reason = _recommend_leverage(market, macro_caution, no_trade)

    # RR下限（地合い連動）
    rr_min = dynamic_min_rr(mkt_score, delta3d)
    adjev_cut = _min_adjev_cut()

    cands: List[Dict] = []
    watch: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get(tcol, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker)).strip()
        sector = str(row.get("sector", row.get("industry_big", "不明"))).strip()

        hist = _fetch_history(ticker, period="340d")
        if hist is None:
            continue

        df = add_indicators(hist)
        c = float(df["Close"].iloc[-1])

        # Price filter
        if not (np.isfinite(c) and PRICE_MIN <= c <= PRICE_MAX):
            continue

        # Liquidity
        liq_ok, adv = calc_liquidity_flags(df, ADV20_MIN)
        if not liq_ok:
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "流動性不足（売買代金不足）"})
            continue

        # ATR%
        atr_pct = calc_atr_pct(df)
        if not np.isfinite(atr_pct) or atr_pct < ATR_PCT_MIN:
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "値動きが小さい（ボラ不足）"})
            continue
        if atr_pct > ATR_PCT_MAX:
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "値動きが荒すぎる（事故ゾーン）"})
            continue

        # Setup
        setup_type, setup_meta = detect_setup(df)
        if setup_type == "-":
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "チャート形状不一致"})
            continue

        # Entry & action
        entry_zone = compute_entry_zone(df, setup_type, setup_meta)

        # GUなら新規は“監視”へ
        if entry_zone["gu"]:
            entry_zone["action"] = "今日は監視"

        # Stop / Targets / RR
        stop = compute_stop(df, setup_type, entry_zone)
        tp1, tp2 = compute_targets(df, setup_type, entry_zone["center"], stop, mkt_score)

        entry = float(entry_zone["center"])
        risk = entry - stop
        if not (np.isfinite(entry) and np.isfinite(stop) and entry > stop and risk > 0):
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "計算不成立"})
            continue

        rr = (tp2 - entry) / risk
        if not (np.isfinite(rr) and rr >= rr_min):
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": f"利益幅が足りない（RR不足 RR<{rr_min:.2f}）"})
            continue

        # Feature for Pwin
        ma20 = float(df["ma20"].iloc[-1])
        ma50 = float(df["ma50"].iloc[-1])
        trend = 1.0 if (entry > ma20 > ma50) else (0.6 if entry > ma20 else 0.3)

        # pullback quality: dist_atrが小さいほど良い（追いかけ禁止）
        dist = float(entry_zone["dist_atr"])
        pull_q = 1.0 - clamp(dist / 0.8, 0.0, 1.0)

        # volume quality（押しで出来高枯れの簡易）
        vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)
        vol_now = float(vol.iloc[-1]) if np.isfinite(vol.iloc[-1]) else float("nan")
        vol_ma20 = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float("nan")
        volume_q = 0.5
        if np.isfinite(vol_now) and np.isfinite(vol_ma20) and vol_ma20 > 0:
            volume_q = clamp((vol_ma20 / (vol_now + 1e-9)), 0.0, 1.0)  # 押しで薄いほど↑
            volume_q = float(clamp(volume_q, 0.0, 1.0))

        liq_norm = clamp(np.log10(max(adv, 1.0)) / 10.0, 0.0, 1.0)
        gap_safe = 0.0 if entry_zone["gu"] else 1.0

        feat = {
            "trend": trend,
            "pullback_quality": pull_q,
            "volume_quality": volume_q,
            "liquidity": liq_norm,
            "gap_risk": gap_safe,
        }
        pwin = estimate_pwin(feat)

        ev = compute_ev(pwin, rr)
        mult = regime_multiplier(mkt_score, delta3d, macro_caution)
        adjev = ev * mult

        # (3) AdjEV < 0.5 を切る
        if not (np.isfinite(adjev) and adjev >= adjev_cut):
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "期待値が低い（補正EV不足）"})
            continue

        # Speed
        atr = float(entry_zone["atr"])
        exp_days = calc_expected_days(entry, tp2, atr, setup_type)
        rday = rr / exp_days if exp_days > 0 else 0.0

        # (4) R/day分布を広げる：固定の足切りをしない（選抜スコアに反映）
        # “速度主導”なので、最終スコアは R/day を強く効かせる
        final_score = float(adjev + 0.65 * rday)

        macro_tag = _classify_macro(sector)

        cands.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "setup": setup_type,
                "rr": float(rr),
                "ev": float(ev),
                "adjev": float(adjev),
                "rday": float(rday),
                "entry": float(entry),
                "entry_low": float(entry_zone["low"]),
                "entry_high": float(entry_zone["high"]),
                "price_now": float(entry_zone["price_now"]),
                "atr": float(entry_zone["atr"]),
                "gu": bool(entry_zone["gu"]),
                "stop": float(stop),
                "tp1": float(tp1),
                "tp2": float(tp2),
                "exp_days": float(exp_days),
                "action": str(entry_zone["action"]),
                "macro": macro_tag,
                "final_score": final_score,
                "_df_ind": df,  # 相関用
            }
        )

    # sort（速度主導 + 期待値）
    cands.sort(key=lambda x: (x["final_score"], x["adjev"], x["rday"], x["rr"]), reverse=True)

    # NO-TRADEなら新規は全部「今日は監視」
    if no_trade:
        for c in cands:
            c["action"] = "今日は監視"

    # diversification
    picked = pick_with_constraints(cands, max_final=max_final, max_sector=2, corr_limit=0.75)

    # rejected by diversification -> watchlistに回す
    picked_set = set([p["ticker"] for p in picked])
    for c in cands:
        if c["ticker"] not in picked_set:
            reason = c.get("reject_reason")
            if reason:
                watch.append({"ticker": c["ticker"], "name": c["name"], "sector": c["sector"], "reason": reason})

    # summary
    def _avg(vals: List[float]) -> float:
        v = [x for x in vals if np.isfinite(x)]
        return float(np.mean(v)) if v else 0.0

    summary = {
        "count": len(picked),
        "avg_rr": _avg([x["rr"] for x in picked]),
        "avg_ev": _avg([x["ev"] for x in picked]),
        "avg_adjev": _avg([x["adjev"] for x in picked]),
        "avg_rday": _avg([x["rday"] for x in picked]),
        "rr_min": float(rr_min),
        "adjev_cut": float(adjev_cut),
        "max_final": int(max_final),
        "no_trade": bool(no_trade),
    }

    return {
        "no_trade": bool(no_trade),
        "no_trade_reasons": reasons,
        "lev": float(lev),
        "lev_reason": lev_reason,
        "max_final": int(max_final),
        "candidates": picked,
        "watchlist": watch[:10],  # 表示は最大10
        "summary": summary,
    }