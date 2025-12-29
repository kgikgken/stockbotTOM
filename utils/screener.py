import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import Config, safe_float
from utils.events import build_event_warnings, is_major_event_day
from utils.sector import top_sectors_5d, sector_rank_map
from utils.features import compute_features
from utils.setup import judge_setup
from utils.entry import make_entry_plan
from utils.rr_ev import compute_rr_ev
from utils.diversify import select_with_constraints


def _fetch_history(ticker: str, period: str = "320d") -> pd.DataFrame:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty and len(df) >= 80:
                return df
        except Exception:
            time.sleep(0.4)
    return pd.DataFrame()


def _load_universe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def _ticker_col(df: pd.DataFrame) -> str:
    if "ticker" in df.columns:
        return "ticker"
    if "code" in df.columns:
        return "code"
    return ""


def _sector_col(df: pd.DataFrame) -> str:
    if "sector" in df.columns:
        return "sector"
    if "industry_big" in df.columns:
        return "industry_big"
    return ""


def _filter_earnings(df: pd.DataFrame, today_date, days: int) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df
    try:
        parsed = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    except Exception:
        return df
    out = df.copy()
    out["_earn"] = parsed
    keep = []
    for d in out["_earn"]:
        if d is None or pd.isna(d):
            keep.append(True)
            continue
        try:
            keep.append(abs((d - today_date).days) > days)
        except Exception:
            keep.append(True)
    return out[keep].drop(columns=["_earn"], errors="ignore")


def _recommend_leverage(mkt_score: int, delta3d: int) -> Tuple[float, str]:
    # delta3d 悪化はレバ抑制
    if mkt_score >= 70 and delta3d >= 0:
        return 2.0, "強気（押し目＋一部ブレイク）"
    if mkt_score >= 60:
        return 1.7, "やや強気（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "中立（厳選・押し目中心）"
    if mkt_score >= 40:
        return 1.1, "やや守り（新規ロット小さめ）"
    return 1.0, "守り（新規かなり絞る）"


def _calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


def _notrade_by_market(cfg: Config, mkt_score: int, delta3d: int) -> Tuple[bool, str]:
    if mkt_score < cfg.NOTRADE_SCORE_LT:
        return True, "MarketScore<45"
    if delta3d <= cfg.NOTRADE_DELTA3D_LE and mkt_score < cfg.NOTRADE_SCORE_LT_2:
        return True, "Δ3d<=-5 & MarketScore<55"
    return False, ""


def run_screening(
    universe_path: str,
    events_path: str,
    today_date,
    market: Dict,
    total_asset: float,
    cfg: Config = Config(),
) -> Dict:
    mkt_score = int(market.get("score", 50))
    delta3d = int(market.get("delta3d", 0))

    lev, lev_comment = _recommend_leverage(mkt_score, delta3d)
    max_pos = _calc_max_position(total_asset, lev)

    # セクター（補助）
    sectors_top = top_sectors_5d(universe_path, top_n=cfg.SECTOR_TOP_N)
    sec_rank = sector_rank_map(universe_path)

    # イベント
    events = build_event_warnings(events_path, today_date)
    major_event = is_major_event_day(events_path, today_date)

    # Market NO-TRADE（最優先）
    notrade_mkt, notrade_reason = _notrade_by_market(cfg, mkt_score, delta3d)

    uni = _load_universe(universe_path)
    tcol = _ticker_col(uni)
    scol = _sector_col(uni)

    if uni.empty or not tcol:
        return {
            "notrade": True,
            "notrade_reason": "ユニバース読込失敗",
            "leverage": lev,
            "leverage_comment": lev_comment,
            "max_position": max_pos,
            "sectors": sectors_top,
            "events": events,
            "picks": [],
            "watch": [],
            "stats": {},
        }

    # 決算±3 完全回避（新規）
    uni_trade = _filter_earnings(uni, today_date, cfg.EARNINGS_EXCLUDE_DAYS)

    candidates: List[dict] = []
    watch: List[dict] = []

    # 指数（RS算出用：TOPIX）
    index_close = None
    try:
        topx = yf.Ticker("^TOPX").history(period="120d", auto_adjust=True)
        if topx is not None and not topx.empty:
            index_close = topx["Close"].astype(float)
    except Exception:
        index_close = None

    # 上位セクターの集合（補助）
    top_sector_names = {s for s, _ in sectors_top}

    for _, row in uni_trade.iterrows():
        ticker = str(row.get(tcol, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker)).strip() or ticker
        sector = str(row.get(scol, "不明")).strip() if scol else "不明"

        hist = _fetch_history(ticker, period="320d")
        if hist.empty:
            continue

        feat = compute_features(hist)
        if feat is None:
            continue

        # Universe足切り（トレード候補のみ）
        # ただし母集団は全銘柄：落ちたものは監視理由で出せる
        price = feat.close
        adv20 = feat.turnover_ma20
        atrpct = feat.atr_pct

        # 価格帯
        if not (cfg.PRICE_MIN <= price <= cfg.PRICE_MAX):
            continue

        # 流動性
        if np.isfinite(adv20) and adv20 < cfg.ADV20_WARN_JPY:
            # 監視へ
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "流動性弱(ADV20<100M)"})
            continue
        if np.isfinite(adv20) and adv20 < cfg.ADV20_MIN_JPY:
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "流動性弱(ADV20<200M)"})
            continue

        # ボラ
        if np.isfinite(atrpct) and atrpct < cfg.ATRPCT_MIN:
            continue
        if np.isfinite(atrpct) and atrpct >= cfg.ATRPCT_MAX:
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "事故ゾーン(ATR%高すぎ)"})
            continue

        # セクター補助：上位5セクター以外は減点（理由ではない）
        sector_rank = sec_rank.get(sector, 999)
        sector_bonus = 0.0
        if sector in top_sector_names:
            sector_bonus = 0.05
        elif sector_rank <= 10:
            sector_bonus = 0.02
        else:
            sector_bonus = -0.02

        setup = judge_setup(feat)
        if setup.setup_type == "-":
            # 形が弱いものは監視しない（ノイズ削減）
            continue

        entry = make_entry_plan(feat, setup.setup_type)
        rr = compute_rr_ev(
            hist=hist,
            feat=feat,
            setup_type=setup.setup_type,
            in_center=entry.in_center,
            in_low=entry.in_low,
            market_score=mkt_score,
            delta3d=delta3d,
            major_event=major_event,
        )

        # 足切り：RR/EV/速度（仕様書）
        if rr.rr < cfg.RR_MIN:
            continue
        if rr.ev_r < cfg.EV_MIN_R:
            continue
        if rr.expected_days > cfg.EXPECTED_DAYS_MAX:
            continue
        if rr.r_per_day < cfg.RPDAY_MIN:
            continue

        # Pwinにセクター補正
        pwin = float(np.clip(rr.pwin + sector_bonus, 0.15, 0.65))
        ev = pwin * rr.rr - (1 - pwin) * 1.0
        adj_ev = ev * rr.regime_mult

        c = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "sector_rank": int(sector_rank),
            "setup": setup.setup_type,
            "in_center": float(round(entry.in_center, 1)),
            "in_low": float(round(entry.in_low, 1)),
            "in_high": float(round(entry.in_high, 1)),
            "price_now": float(round(feat.close, 1)),
            "atr": float(round(feat.atr14, 1)),
            "gu": "Y" if entry.gu_flag else "N",
            "action": entry.action,
            "stop": rr.stop,
            "tp1": rr.tp1,
            "tp2": rr.tp2,
            "rr": float(rr.rr),
            "pwin": float(pwin),
            "ev": float(ev),
            "adj_ev": float(adj_ev),
            "r_per_day": float(rr.r_per_day),
            "expected_days": float(rr.expected_days),
            "_close_series": hist["Close"].astype(float).copy(),
        }

        # 追いかけ禁止：WATCH_ONLYは本命から除外（監視へ）
        if entry.action == "WATCH_ONLY":
            c2 = dict(c)
            c2["reason"] = "追いかけ禁止(乖離/GU)"
            watch.append(c2)
            continue

        candidates.append(c)

    # ソート：AdjEV → R/day → RR
    candidates.sort(key=lambda x: (x["adj_ev"], x["r_per_day"], x["rr"]), reverse=True)

    picked, watch2 = select_with_constraints(
        candidates=candidates,
        max_final=cfg.SWING_MAX_FINAL,
        max_same_sector=cfg.MAX_SAME_SECTOR,
        corr_max=cfg.CORR_MAX,
    )

    # watch整形（最大10に寄せる）
    watch_all = watch + watch2
    # 監視は「理由が濃い順」：相関/セクター/流動性/追いかけ
    prio = {"相関高": 4, "セクター上限": 3, "流動性弱": 2, "追いかけ禁止": 1}
    def _wkey(w):
        r = str(w.get("reason", ""))
        p = 0
        for k, v in prio.items():
            if k in r:
                p = max(p, v)
        return (p, safe_float(w.get("adj_ev", 0), 0), safe_float(w.get("r_per_day", 0), 0))

    watch_all.sort(key=_wkey, reverse=True)
    watch_all = watch_all[: cfg.WATCH_MAX]

    # NO-TRADE最終確定（候補の質でも）
    notrade = notrade_mkt
    reasons: List[str] = []
    if notrade_mkt:
        reasons.append(notrade_reason)

    # 候補平均AdjEV
    if picked:
        avg_adj_ev = float(np.mean([c["adj_ev"] for c in picked]))
        gu_ratio = float(np.mean([1.0 if c.get("gu") == "Y" else 0.0 for c in picked]))
    else:
        avg_adj_ev = 0.0
        gu_ratio = 0.0

    if avg_adj_ev < 0.30:
        notrade = True
        reasons.append("平均AdjEV<0.3R")
    if gu_ratio >= 0.60:
        notrade = True
        reasons.append("GU比率>=60%")

    # ロット事故警告（想定最大損失）
    # 上位採用を全部入れたと仮定して、1トレード1.5%リスク
    risk_per_trade = cfg.RISK_PER_TRADE
    max_risk_yen = total_asset * lev * risk_per_trade
    assumed_risk_sum = max_risk_yen * max(len(picked), 0)
    worst_loss_pct = assumed_risk_sum / total_asset if total_asset > 0 else 0.0

    risk_warn = worst_loss_pct > cfg.MAX_PORTFOLIO_RISK_PCT

    stats = {
        "avg_rr": float(np.mean([c["rr"] for c in picked])) if picked else 0.0,
        "avg_ev": float(np.mean([c["ev"] for c in picked])) if picked else 0.0,
        "avg_adj_ev": float(avg_adj_ev),
        "avg_r_per_day": float(np.mean([c["r_per_day"] for c in picked])) if picked else 0.0,
        "gu_ratio": float(gu_ratio),
        "risk_warn": bool(risk_warn),
        "worst_loss_pct": float(worst_loss_pct),
        "assumed_risk_yen": float(assumed_risk_sum),
        "notrade_reasons": reasons,
    }

    # 本命の内部seriesを落とす（出力軽量化）
    for c in picked:
        c.pop("_close_series", None)
    for c in watch_all:
        c.pop("_close_series", None)

    return {
        "notrade": bool(notrade),
        "notrade_reason": " & ".join(reasons) if reasons else "",
        "leverage": float(lev),
        "leverage_comment": lev_comment,
        "max_position": int(max_pos),
        "sectors": sectors_top,
        "events": events,
        "picks": picked,
        "watch": watch_all,
        "stats": stats,
        "meta": {
            "market_score": mkt_score,
            "delta3d": delta3d,
            "major_event": bool(major_event),
        },
    }