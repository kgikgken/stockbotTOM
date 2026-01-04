# utils/screener.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import MarketState, EventState, safe_float
from utils.features import compute_tech
from utils.setup import decide_setup
from utils.entry import build_entry_plan
from utils.rr_ev import compute_rr_ev, rr_min_by_market
from utils.diversify import corr_filter


# ============================================================
# 設定（Swing専用）
# ============================================================
EARNINGS_EXCLUDE_DAYS = 3

# 流動性（売買代金）
ADV20_MIN = 200_000_000  # 2億円/日
ADV20_SOFT = 100_000_000  # 1億円/日（弱いなら理由付けで除外）

# ボラ
ATR_PCT_MIN = 0.015
ATR_PCT_MAX = 0.060

# 候補数
MAX_FINAL_NORMAL = 5
MAX_FINAL_EVENT = 2

# EV足切り（c）
ADJ_EV_MIN = 0.5

# 週次新規（別実装とぶつけないため、ここでは「表示用」だけ残す）
WEEKLY_NEW_LIMIT = 3


@dataclass
class Candidate:
    ticker: str
    name: str
    sector: str
    setup: str
    rr: float
    ev: float
    adj_ev: float
    rpd: float
    exp_days: float
    entry_center: float
    entry_low: float
    entry_high: float
    close: float
    atr: float
    gu: bool
    stop: float
    tp1: float
    tp2: float
    action: str
    macro_tag: str
    reject_reason: str = ""


def _load_universe(path: str) -> pd.DataFrame:
    """
    universe_jpx.csv 必須列（最低）:
      - ticker (例: 6503.T)
    任意:
      - name
      - sector
      - earnings_date (YYYY-MM-DD)
    """
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("universe_jpx.csv needs 'ticker' column")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if "name" not in df.columns:
        df["name"] = ""
    if "sector" not in df.columns:
        df["sector"] = ""
    if "earnings_date" not in df.columns:
        df["earnings_date"] = ""
    return df


def _earnings_near(_date_str: str) -> bool:
    # 簡易：空はFalse
    s = str(_date_str).strip()
    if not s:
        return False
    # 本気版は営業日計算が必要だが、ここでは「±3営業日」を保守的に近似
    # → 7日以内は危険として除外
    try:
        d = pd.to_datetime(s).to_pydatetime()
        now = pd.Timestamp.now(tz="Asia/Tokyo").to_pydatetime()
        return abs((d - now).days) <= 7
    except Exception:
        return False


def _macro_tag(sector: str) -> str:
    # 超簡易：本来は金利感応・景気敏感など分類テーブルを持つ
    s = str(sector)
    if "銀行" in s or "保険" in s or "不動産" in s:
        return "rate_sensitive"
    if "機械" in s or "化学" in s or "金属" in s:
        return "cyclical"
    if "食料" in s or "医薬" in s:
        return "defensive"
    return "other"


def _fetch_ohlcv(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        df = df.dropna()
        if len(df) < 80:
            return None
        return df
    except Exception:
        return None


def _adv20_jpy(df: pd.DataFrame) -> float:
    # 売買代金 = Close * Volume を近似（JPY建て）
    try:
        v = (df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1]
        return float(v)
    except Exception:
        return 0.0


def run_screening(
    universe_path: str,
    positions_path: str,
    events: EventState,
    market: MarketState,
    sectors: List[Tuple[str, float]],
) -> Dict[str, Any]:
    dfu = _load_universe(universe_path)

    macro_event_near = bool(events.macro_event_near)
    rr_min = rr_min_by_market(market.score)

    no_trade = bool(market.no_trade)
    reasons: List[str] = []
    if no_trade:
        reasons.append(market.reason)

    # まず候補を全部作る（落とす理由も残す）
    candidates: List[Candidate] = []
    rejected: List[Candidate] = []

    for _, r in dfu.iterrows():
        ticker = str(r["ticker"]).strip()
        name = str(r.get("name", "")).strip()
        sector = str(r.get("sector", "")).strip()

        # 決算±3営業日（近似）回避
        if _earnings_near(str(r.get("earnings_date", ""))):
            rejected.append(
                Candidate(
                    ticker=ticker, name=name, sector=sector, setup="-",
                    rr=0, ev=0, adj_ev=0, rpd=0, exp_days=0,
                    entry_center=0, entry_low=0, entry_high=0,
                    close=0, atr=0, gu=False, stop=0, tp1=0, tp2=0,
                    action="今日は見送り", macro_tag=_macro_tag(sector),
                    reject_reason="決算近接(±3営業日回避)",
                )
            )
            continue

        ohlc = _fetch_ohlcv(ticker)
        if ohlc is None:
            continue

        tech = compute_tech(ohlc)
        if tech is None:
            continue

        # Universe: 価格レンジ（任意）
        if not (200 <= tech.close <= 15000):
            continue

        # ADV20
        adv20 = _adv20_jpy(ohlc)
        if adv20 < ADV20_MIN:
            rejected.append(
                Candidate(
                    ticker=ticker, name=name, sector=sector, setup="-",
                    rr=0, ev=0, adj_ev=0, rpd=0, exp_days=0,
                    entry_center=0, entry_low=0, entry_high=0,
                    close=tech.close, atr=tech.atr, gu=False, stop=0, tp1=0, tp2=0,
                    action="今日は見送り", macro_tag=_macro_tag(sector),
                    reject_reason="流動性不足(売買代金不足)",
                )
            )
            continue

        # ATR%
        if tech.atr_pct < ATR_PCT_MIN:
            rejected.append(
                Candidate(
                    ticker=ticker, name=name, sector=sector, setup="-",
                    rr=0, ev=0, adj_ev=0, rpd=0, exp_days=0,
                    entry_center=0, entry_low=0, entry_high=0,
                    close=tech.close, atr=tech.atr, gu=False, stop=0, tp1=0, tp2=0,
                    action="今日は見送り", macro_tag=_macro_tag(sector),
                    reject_reason="値動きが小さい（ボラ不足）",
                )
            )
            continue
        if tech.atr_pct >= ATR_PCT_MAX:
            rejected.append(
                Candidate(
                    ticker=ticker, name=name, sector=sector, setup="-",
                    rr=0, ev=0, adj_ev=0, rpd=0, exp_days=0,
                    entry_center=0, entry_low=0, entry_high=0,
                    close=tech.close, atr=tech.atr, gu=False, stop=0, tp1=0, tp2=0,
                    action="今日は見送り", macro_tag=_macro_tag(sector),
                    reject_reason="ボラ高すぎ（事故ゾーン）",
                )
            )
            continue

        # Setup（A1/A2）
        setup = decide_setup(tech)
        if not setup.ok:
            rejected.append(
                Candidate(
                    ticker=ticker, name=name, sector=sector, setup="-",
                    rr=0, ev=0, adj_ev=0, rpd=0, exp_days=0,
                    entry_center=0, entry_low=0, entry_high=0,
                    close=tech.close, atr=tech.atr, gu=False, stop=0, tp1=0, tp2=0,
                    action="今日は見送り", macro_tag=_macro_tag(sector),
                    reject_reason="チャート形状不一致",
                )
            )
            continue

        # Entry
        ep = build_entry_plan(tech, setup.kind)
        if ep.action == "今日は見送り":
            rejected.append(
                Candidate(
                    ticker=ticker, name=name, sector=sector, setup=setup.kind,
                    rr=0, ev=0, adj_ev=0, rpd=0, exp_days=0,
                    entry_center=ep.in_center, entry_low=ep.in_low, entry_high=ep.in_high,
                    close=tech.close, atr=tech.atr, gu=ep.gu, stop=0, tp1=0, tp2=0,
                    action="今日は見送り", macro_tag=_macro_tag(sector),
                    reject_reason="追いかけ禁止（IN帯外/GU/乖離）",
                )
            )
            continue

        # rr/ev
        trend_strength = 1.0  # ここは今後強化できる（MA傾き等）
        rs = clamp(tech.ret20, -0.10, 0.10) / 0.10  # 簡易正規化
        volume_quality = 0.0
        if tech.vol_ma20 > 0:
            volume_quality = clamp((tech.vol / tech.vol_ma20 - 1.0), -0.5, 1.0) / 1.0
        liquidity = clamp((adv20 - ADV20_MIN) / ADV20_MIN, 0.0, 1.0)
        gap_risk = 1.0 if ep.gu else 0.0

        rrev = compute_rr_ev(
            entry=ep.in_center,
            in_low=ep.in_low,
            atr=tech.atr,
            market=market,
            macro_event_near=macro_event_near,
            trend_strength=float(trend_strength),
            rs=float(rs),
            volume_quality=float(volume_quality),
            liquidity=float(liquidity),
            gap_risk=float(gap_risk),
            ret20=float(tech.ret20),
            atr_pct=float(tech.atr_pct),
        )

        # RR下限（地合い連動）
        if rrev.rr < rr_min:
            rejected.append(
                Candidate(
                    ticker=ticker, name=name, sector=sector, setup=setup.kind,
                    rr=rrev.rr, ev=rrev.ev, adj_ev=rrev.adj_ev, rpd=rrev.r_per_day, exp_days=rrev.exp_days,
                    entry_center=ep.in_center, entry_low=ep.in_low, entry_high=ep.in_high,
                    close=tech.close, atr=tech.atr, gu=ep.gu,
                    stop=rrev.stop, tp1=rrev.tp1, tp2=rrev.tp2,
                    action="今日は見送り", macro_tag=_macro_tag(sector),
                    reject_reason=f"RR不足(RR<{rr_min:.1f})",
                )
            )
            continue

        # AdjEV足切り（c）
        if rrev.adj_ev < ADJ_EV_MIN:
            rejected.append(
                Candidate(
                    ticker=ticker, name=name, sector=sector, setup=setup.kind,
                    rr=rrev.rr, ev=rrev.ev, adj_ev=rrev.adj_ev, rpd=rrev.r_per_day, exp_days=rrev.exp_days,
                    entry_center=ep.in_center, entry_low=ep.in_low, entry_high=ep.in_high,
                    close=tech.close, atr=tech.atr, gu=ep.gu,
                    stop=rrev.stop, tp1=rrev.tp1, tp2=rrev.tp2,
                    action="今日は見送り", macro_tag=_macro_tag(sector),
                    reject_reason="補正EV不足(AdjEV<0.5)",
                )
            )
            continue

        # 速度優先：R/日
        c = Candidate(
            ticker=ticker,
            name=name,
            sector=sector,
            setup=setup.kind,
            rr=rrev.rr,
            ev=rrev.ev,
            adj_ev=rrev.adj_ev,
            rpd=rrev.r_per_day,
            exp_days=rrev.exp_days,
            entry_center=ep.in_center,
            entry_low=ep.in_low,
            entry_high=ep.in_high,
            close=tech.close,
            atr=tech.atr,
            gu=ep.gu,
            stop=rrev.stop,
            tp1=rrev.tp1,
            tp2=rrev.tp2,
            action=ep.action,
            macro_tag=_macro_tag(sector),
        )
        # 相関判定用リターン系列
        ret_series = ohlc["Close"].pct_change().tail(30)
        setattr(c, "ret_series", ret_series)
        candidates.append(c)

    # NO-TRADEなら新規候補は出すが、行動は「今日は見送り」に寄せる（レポート上）
    # ただし計算自体は維持（強い候補を観測する価値はある）
    # イベント接近時は候補数制限
    max_final = MAX_FINAL_EVENT if macro_event_near else MAX_FINAL_NORMAL

    # ソート：補正EV → R/日（速度）を優先しつつ
    # ※あなたの方針で「速度主導」にしたいなら R/日 を主キーにできる
    candidates.sort(key=lambda x: (x.adj_ev, x.rpd), reverse=True)

    # 相関フィルタ（採用を絞る）
    # Candidateをdictっぽく扱うため変換
    cand_dicts: List[Dict[str, Any]] = []
    for c in candidates:
        d = c.__dict__.copy()
        d["ret_series"] = getattr(c, "ret_series", None)
        cand_dicts.append(d)

    picked_dicts, removed_corr = corr_filter(cand_dicts, corr_limit=0.75)
    picked: List[Candidate] = []
    removed: List[Candidate] = []

    for d in picked_dicts[:max_final]:
        picked.append(Candidate(**{k: d[k] for k in Candidate.__dataclass_fields__.keys() if k in d}))

    for d in removed_corr:
        if "reject_reason" not in d:
            d["reject_reason"] = "相関高"
        removed.append(Candidate(**{k: d[k] for k in Candidate.__dataclass_fields__.keys() if k in d}))

    # 平均AdjEVが低い場合は「見送り」判定（あなたの運用思想に合わせる）
    avg_adj_ev = float(np.mean([c.adj_ev for c in picked])) if picked else 0.0
    force_no_trade = False
    if macro_event_near and avg_adj_ev < 1.2:
        force_no_trade = True
        reasons.append("イベント接近 & 平均補正EVが弱い")

    return {
        "picked": picked,
        "rejected": rejected,
        "removed": removed,
        "no_trade": no_trade or force_no_trade,
        "reasons": reasons,
        "weekly_new": 0,
        "weekly_limit": WEEKLY_NEW_LIMIT,
        "macro_event_near": macro_event_near,
        "max_final": max_final,
        "rr_min": rr_min,
        "avg_rr": float(np.mean([c.rr for c in picked])) if picked else 0.0,
        "avg_ev": float(np.mean([c.ev for c in picked])) if picked else 0.0,
        "avg_adj_ev": avg_adj_ev,
        "avg_rpd": float(np.mean([c.rpd for c in picked])) if picked else 0.0,
    }