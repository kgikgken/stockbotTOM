from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.setup import detect_setup_type
from utils.entry import calc_in_band
from utils.rr_ev import (
    adv20_from_df,
    gu_flag,
    calc_stop_tp,
    estimate_pwin,
    calc_ev,
    regime_multiplier,
    expected_days,
    r_per_day,
)
from utils.diversify import apply_diversify, DiversifyConfig
from utils.util import clamp


@dataclass
class ScreenerConfig:
    earnings_exclude_days: int = 3
    close_min: float = 200.0
    close_max: float = 15000.0
    adv20_min: float = 200_000_000.0
    atrp_min: float = 0.015
    atrp_max: float = 0.060
    rr_min: float = 1.8
    ev_min: float = 0.30
    expdays_max: float = 5.0
    rperday_min: float = 0.50

    max_final: int = 5
    max_watch: int = 10

    # NO-TRADE条件
    no_trade_mkt_score: int = 45
    no_trade_delta3d: float = -5.0
    no_trade_delta3d_mktcap: int = 55
    no_trade_avg_adjev: float = 0.30
    no_trade_gu_ratio: float = 0.60


def _fetch_hist(ticker: str, period: str = "260d") -> pd.DataFrame | None:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.35)
    return None


def _get_ticker_col(df: pd.DataFrame) -> str | None:
    if "ticker" in df.columns:
        return "ticker"
    if "code" in df.columns:
        return "code"
    return None


def _sector_col(df: pd.DataFrame) -> str:
    if "sector" in df.columns:
        return "sector"
    if "industry_big" in df.columns:
        return "industry_big"
    return "sector"


def _filter_earnings(df: pd.DataFrame, today_date, days: int) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df
    tmp = df.copy()
    d = pd.to_datetime(tmp["earnings_date"], errors="coerce").dt.date
    tmp["_earn"] = d

    keep = []
    for x in tmp["_earn"]:
        if x is None or pd.isna(x):
            keep.append(True)
            continue
        try:
            keep.append(abs((x - today_date).days) > days)
        except Exception:
            keep.append(True)
    return tmp.loc[keep].drop(columns=["_earn"], errors="ignore")


def _universe_filter(df: pd.DataFrame, hist: pd.DataFrame, cfg: ScreenerConfig) -> Tuple[bool, str, Dict]:
    """
    Universe足切り
    - Close範囲
    - ADV20
    - ATR%
    """
    close = float(hist["Close"].astype(float).iloc[-1])
    if not (cfg.close_min <= close <= cfg.close_max):
        return False, "価格レンジ外", {}

    adv20 = adv20_from_df(hist)
    if not np.isfinite(adv20) or adv20 < cfg.adv20_min:
        return False, f"流動性弱(ADV20<{int(cfg.adv20_min/1e6)}M)", {"adv20": adv20}

    # ATR%
    high = hist["High"].astype(float)
    low = hist["Low"].astype(float)
    close_s = hist["Close"].astype(float)
    tr = pd.concat([high - low, (high - close_s.shift(1)).abs(), (low - close_s.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 15 else float("nan")
    if not np.isfinite(atr14) or atr14 <= 0:
        return False, "ATR不明", {"adv20": adv20}

    atrp = float(atr14 / close)
    if atrp < cfg.atrp_min:
        return False, "ボラ不足(ATR%<1.5)", {"adv20": adv20, "atr": atr14, "atrp": atrp}
    if atrp > cfg.atrp_max:
        # 事故ゾーンは監視送り（除外ではなく落とす）
        return False, "ボラ過大(事故ゾーン)", {"adv20": adv20, "atr": atr14, "atrp": atrp}

    return True, "", {"adv20": adv20, "atr": atr14, "atrp": atrp}


def _calc_candidate(
    ticker: str,
    name: str,
    sector: str,
    sector_rank: int | None,
    hist: pd.DataFrame,
    mkt: Dict,
    cfg: ScreenerConfig,
    event_risk: bool,
) -> Dict | None:
    mkt_score = int(mkt.get("score", 50))
    delta3d = float(mkt.get("delta3d", 0.0))

    setup_type = detect_setup_type(hist)
    if setup_type not in ("A", "B"):
        return None

    # IN帯
    in_center, in_low, in_high = calc_in_band(hist, setup_type)

    # GU
    gu = gu_flag(hist)

    # 行動分類（追いかけ禁止を機械化）
    close = float(hist["Close"].astype(float).iloc[-1])
    atr14 = float(cfg.atrp_min)  # dummy
    # 正しいATRは universe_filter の meta から入れたいので、後で上書き（ここは軽く）
    try:
        high = hist["High"].astype(float)
        low = hist["Low"].astype(float)
        close_s = hist["Close"].astype(float)
        tr = pd.concat([high - low, (high - close_s.shift(1)).abs(), (low - close_s.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])
    except Exception:
        atr14 = max(close * 0.01, 1.0)

    dist_atr = abs(close - in_center) / max(atr14, 1e-9)

    if gu or dist_atr > 0.8:
        action = "WATCH_ONLY"
    elif in_low <= close <= in_high:
        action = "EXEC_NOW"
    else:
        action = "LIMIT_WAIT"

    # Stop/TP
    stop, tp1, tp2 = calc_stop_tp(hist, setup_type, in_center, in_low)

    rr = (tp2 - in_center) / max(in_center - stop, 1e-9)
    rr = float(rr)

    if rr < cfg.rr_min:
        return None

    # Pwin/EV
    adv20 = adv20_from_df(hist)
    pwin = estimate_pwin(hist, sector_rank=sector_rank, adv20=adv20, gu=gu, mkt_score=mkt_score)
    ev = calc_ev(rr, pwin)
    if ev < cfg.ev_min:
        return None

    mult = regime_multiplier(mkt_score, delta3d, event_risk=event_risk)
    adjev = float(ev * mult)

    # 速度
    exp_days = expected_days(tp2, in_center, atr14)
    rpd = r_per_day(rr, exp_days)
    if exp_days > cfg.expdays_max or rpd < cfg.rperday_min:
        return None

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "setup": setup_type,
        "in_center": float(round(in_center, 1)),
        "in_low": float(round(in_low, 1)),
        "in_high": float(round(in_high, 1)),
        "price_now": float(round(close, 1)),
        "atr": float(round(atr14, 1)),
        "gu": "Y" if gu else "N",
        "stop": float(round(stop, 1)),
        "tp1": float(round(tp1, 1)),
        "tp2": float(round(tp2, 1)),
        "rr": float(round(rr, 2)),
        "pwin": float(round(pwin, 3)),
        "ev": float(round(ev, 2)),
        "adjev": float(round(adjev, 2)),
        "exp_days": float(round(exp_days, 1)),
        "r_per_day": float(round(rpd, 2)),
        "action": action if action != "WATCH_ONLY" else "指値待ち" if action == "LIMIT_WAIT" else "監視のみ",
        "action_code": action,
        "sector_rank": sector_rank,
    }


def run_swing_screening(
    today_date,
    universe_path: str,
    mkt: Dict,
    positions_df: pd.DataFrame | None = None,
    cfg: ScreenerConfig = ScreenerConfig(),
) -> Dict:
    """
    戻り:
      {
        no_trade: bool,
        no_trade_reason: str,
        candidates: [..selected..],
        watch: [..watch..],
        avg_rr, avg_ev, avg_adjev, avg_rpd,
        lot_warning_text: str | None
      }
    """
    try:
        uni = pd.read_csv(universe_path)
    except Exception:
        return {"no_trade": True, "no_trade_reason": "ユニバース読込失敗", "candidates": [], "watch": []}

    t_col = _get_ticker_col(uni)
    if not t_col:
        return {"no_trade": True, "no_trade_reason": "ticker列不明", "candidates": [], "watch": []}

    sec_col = _sector_col(uni)

    # 決算±3除外（新規）
    uni2 = _filter_earnings(uni, today_date, cfg.earnings_exclude_days)

    # 市況NO-TRADE（先に判定）
    mkt_score = int(mkt.get("score", 50))
    delta3d = float(mkt.get("delta3d", 0.0))

    no_trade = False
    no_trade_reason = ""

    if mkt_score < cfg.no_trade_mkt_score:
        no_trade = True
        no_trade_reason = f"MarketScore<{cfg.no_trade_mkt_score}"
    if (delta3d <= cfg.no_trade_delta3d) and (mkt_score < cfg.no_trade_delta3d_mktcap):
        no_trade = True
        no_trade_reason = f"Δ3d<={cfg.no_trade_delta3d} & MarketScore<{cfg.no_trade_delta3d_mktcap}"

    # セクターrank（補助）
    # ※Universeに sector が入ってる前提で、簡易に 5日上位を report 側で表示するだけでもOKだが、rankはここで暫定生成
    # ここでは「銘柄セクターが分かるなら上位を軽く補正する」程度。
    sector_rank_map: Dict[str, int] = {}
    try:
        # セクターごとの平均5dを粗く推定（最大30銘柄/セクター）
        tmp = uni2[[t_col, sec_col]].dropna()
        sector_scores: Dict[str, float] = {}
        for sec, sub in tmp.groupby(sec_col):
            tickers = sub[t_col].astype(str).tolist()[:30]
            vals = []
            for t in tickers[:12]:
                h = _fetch_hist(t, "6d")
                if h is None or len(h) < 2:
                    continue
                c = h["Close"].astype(float)
                vals.append(float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0))
            if vals:
                sector_scores[str(sec)] = float(np.mean(vals))
        ranked = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (sec, _) in enumerate(ranked, start=1):
            sector_rank_map[str(sec)] = i
    except Exception:
        sector_rank_map = {}

    candidates_raw: List[Dict] = []
    watch_raw: List[Dict] = []
    price_hist_map: Dict[str, pd.DataFrame] = {}

    # スキャン
    for _, row in uni2.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get(sec_col, "不明"))
        sector_rank = sector_rank_map.get(sector)

        hist = _fetch_hist(ticker, "260d")
        if hist is None or len(hist) < 80:
            continue

        ok, reason, meta = _universe_filter(uni2, hist, cfg)
        if not ok:
            # 監視へ（理由つけ）
            watch_raw.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "setup": "-",
                "rr": 0.0,
                "r_per_day": 0.0,
                "reject_reason": reason,
                "gu": "N",
                "action_code": "WATCH_ONLY",
            })
            continue

        price_hist_map[ticker] = hist

        # イベントリスク（ここは events.csv ベースで main で警告するが、EV補正にだけ使う）
        # “重要イベント前日”などの厳密は events 側でやる前提。ここはFalse固定でOK（必要なら後で拡張）。
        event_risk = False

        c = _calc_candidate(
            ticker=ticker,
            name=name,
            sector=sector,
            sector_rank=sector_rank,
            hist=hist,
            mkt=mkt,
            cfg=cfg,
            event_risk=event_risk,
        )

        if c is None:
            continue

        # action_code で WATCH_ONLY は監視へ
        if c.get("action_code") == "WATCH_ONLY":
            c2 = dict(c)
            c2["reject_reason"] = "GU/乖離"
            watch_raw.append(c2)
            continue

        candidates_raw.append(c)

    # まずAdjustedEV→R/day→RRでソート
    candidates_raw.sort(key=lambda x: (x.get("adjev", 0), x.get("r_per_day", 0), x.get("rr", 0)), reverse=True)

    # 分散制約
    selected, rejected = apply_diversify(
        candidates_raw,
        price_hist_map=price_hist_map,
        cfg=DiversifyConfig(max_per_sector=2, corr_lookback=20, corr_limit=0.75),
    )

    # 監視枠に rejected を入れる
    for r in rejected:
        r2 = dict(r)
        r2["reject_reason"] = r2.get("reject_reason", "制約落ち")
        watch_raw.append(r2)

    # 最終本命
    final = selected[:cfg.max_final]

    # 監視枠
    # （理由があるものだけ上位）
    watch_raw = watch_raw[: cfg.max_watch]

    # 平均
    def _avg(key: str) -> float:
        if not final:
            return 0.0
        vals = [float(x.get(key, 0.0)) for x in final]
        return float(np.mean(vals))

    avg_rr = _avg("rr")
    avg_ev = _avg("ev")
    avg_adjev = _avg("adjev")
    avg_rpd = _avg("r_per_day")

    # NO-TRADEの“最終確定”（候補平均AdjustedEV / GU比率）
    if final:
        if avg_adjev < cfg.no_trade_avg_adjev:
            no_trade = True
            no_trade_reason = f"平均AdjEV<{cfg.no_trade_avg_adjev}"
        gu_ratio = float(sum(1 for x in final if x.get("gu") == "Y")) / float(len(final))
        if gu_ratio >= cfg.no_trade_gu_ratio:
            no_trade = True
            no_trade_reason = f"GU比率>={int(cfg.no_trade_gu_ratio*100)}%"

    # NO-TRADEなら新規枠は空にして“監視だけ”にする
    if no_trade:
        # ただし候補自体は情報として残す（レポート側で「🚫」にする）
        pass

    return {
        "no_trade": bool(no_trade),
        "no_trade_reason": no_trade_reason,
        "candidates": final,
        "watch": watch_raw,
        "avg_rr": float(round(avg_rr, 2)),
        "avg_ev": float(round(avg_ev, 2)),
        "avg_adjev": float(round(avg_adjev, 2)),
        "avg_rpd": float(round(avg_rpd, 2)),
        "lot_warning_text": None,  # レポート側で positions/stop から出すならここを拡張
    }