"""topdown 保有ポジションの監視 v2.0 — 固定利確の撤廃に伴う全面改訂.

v1.0の「固定利確2R + 時間ストップ」から、仕様書v2.0の設計へ:
  ① 初期ストップ(構造ベース)割れ — 終値で判定 → 翌日寄り成行を推奨
  ② +1R到達 — 半分利確の検討(2単元以上のときのみ。1単元ならトレーリング一本)
  ③ +1R到達後 — ストップを直近の構造(押し安値)の下へ引き上げ。★建値には上げない
  ④ トレーリング水準の提示 — トリガー別(シャンデリア / 終値ATR / 移動平均)
  ⑤ 時間ストップ — トリガー別(材料反応20 / 高値ブレイク10 / 押し目10営業日)
自動決済はしない。到達時は警告のみ(発注は人間が行う)。

positions_topdown.csv の列:
  ticker,shares,entry_price,entry_date,stop_price,trigger,hold_tag
  ※v1.0にあった target_price は不要(固定利確を撤廃したため)。あっても無視する。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import Config
from .data import fetch_ohlcv
from .ta import atr_wilder, sma
from .universe import norm_ticker

POSITIONS_PATH = "positions_topdown.csv"
COLS = ["ticker", "shares", "entry_price", "entry_date", "stop_price", "trigger", "hold_tag"]

TRIG_GAP, TRIG_PULL, TRIG_BREAK = "材料反応", "押し目", "高値ブレイク"


def load_positions_topdown(path: str = POSITIONS_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(p, dtype=str).fillna("")
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    df["ticker"] = df["ticker"].map(norm_ticker)
    return df


def _to_float(v) -> float | None:
    try:
        s = str(v).strip()
        return float(s) if s and s.lower() != "nan" else None
    except Exception:
        return None


def _bdays_since(entry_date: str, today: str) -> int | None:
    try:
        d0, d1 = pd.Timestamp(entry_date).date(), pd.Timestamp(today).date()
    except Exception:
        return None
    return int(np.busday_count(d0, d1)) if d1 >= d0 else None


def _time_stop_for(trigger: str, cfg: Config) -> int:
    return {TRIG_GAP: cfg.time_stop_gap, TRIG_BREAK: cfg.time_stop_break,
            TRIG_PULL: cfg.time_stop_pull}.get(trigger, cfg.time_stop_pull)


def _trailing_level(trigger: str, df: pd.DataFrame, atr_now: float, cfg: Config):
    """トリガー別のトレーリング水準を返す。(水準, 説明) / 算出不可なら (None, "")"""
    try:
        c, h = df["Close"].dropna(), df["High"].dropna()
        if trigger == TRIG_GAP:
            n = cfg.trail_chandelier_days
            if len(h) < n:
                return None, ""
            lvl = float(h.iloc[-n:].max()) - cfg.trail_chandelier_atr * atr_now
            return lvl, f"シャンデリア({n}日高値−{cfg.trail_chandelier_atr:.1f}ATR)"
        if trigger == TRIG_BREAK:
            lvl = float(c.max()) - cfg.trail_close_atr_mult * atr_now
            # 終値ベースの最高値から k×ATR
            lvl = float(c.cummax().iloc[-1]) - cfg.trail_close_atr_mult * atr_now
            return lvl, f"終値ベース(最高終値−{cfg.trail_close_atr_mult:.1f}ATR)"
        n = cfg.trail_ma_days
        ma = sma(c, n)
        if pd.isna(ma.iloc[-1]):
            return None, ""
        return float(ma.iloc[-1]), f"{n}日移動平均"
    except Exception:
        return None, ""


def _structure_stop(df: pd.DataFrame, lookback: int, atr_now: float, cfg: Config):
    """+1R到達後にストップを引き上げる先(直近の押し安値 − 緩衝)。"""
    try:
        l = df["Low"].dropna()
        if len(l) < lookback:
            return None
        return float(l.iloc[-lookback:].min()) - cfg.stop_buffer_atr_mult * atr_now
    except Exception:
        return None


def check_held_positions(pos_df: pd.DataFrame, universe: pd.DataFrame,
                         cfg: Config, today: str) -> List[dict]:
    if pos_df is None or len(pos_df) == 0:
        return []
    tickers = [str(t).strip() for t in pos_df["ticker"].tolist() if str(t).strip()]
    ohlcv, _ = fetch_ohlcv(tickers, cfg.history_days, dryrun=cfg.dryrun)
    uni_idx = universe.drop_duplicates("ticker").set_index("ticker") if len(universe) else None

    alerts: List[dict] = []
    for _, p in pos_df.iterrows():
        ticker = str(p["ticker"]).strip()
        if not ticker:
            continue
        code = ticker.replace(".T", "")
        name = (str(uni_idx.loc[ticker]["name"])
                if (uni_idx is not None and ticker in uni_idx.index) else ticker)
        try:
            df = ohlcv.get(ticker)
            if df is None:
                alerts.append({"code": code, "name": name, "hit": None,
                               "note": "データ取得不可(コード表記・上場廃止等、要確認)"})
                continue
            close_s = df["Close"].dropna() if "Close" in df.columns else pd.Series(dtype=float)
            if not len(close_s):
                alerts.append({"code": code, "name": name, "hit": None,
                               "note": f"データ不足(算出不可・取得{len(df)}行)"})
                continue
            close_now = float(close_s.iloc[-1])
            atr_now = float(atr_wilder(df["High"], df["Low"], df["Close"], cfg.atr_period).iloc[-1])

            trigger = str(p.get("trigger", "")).strip() or TRIG_PULL
            entry = _to_float(p.get("entry_price"))
            stop = _to_float(p.get("stop_price"))
            shares = _to_float(p.get("shares")) or 0
            entry_date = str(p.get("entry_date", "")).strip()
            days = _bdays_since(entry_date, today) if entry_date else None
            limit = _time_stop_for(trigger, cfg)

            notes: List[str] = []
            hit = None
            current_r = None

            # ① 構造ストップ割れ(終値ベース判定 → 翌日寄り成行)
            if stop is not None and close_now <= stop:
                hit = "stop"
                notes.append(f"ストップ({stop:,.0f}円)を終値{close_now:,.0f}円で割れ — "
                             "翌日寄り成行で手仕舞い(値幅制限・寄りギャップ対策)")

            # ② R倍数の算出と +1R 到達判定
            if hit is None and entry is not None and stop is not None and entry > stop:
                risk_w = entry - stop
                current_r = (close_now - entry) / risk_w
                if current_r >= cfg.partial_tp_r:
                    if shares >= 200:
                        notes.append(f"{current_r:.1f}R到達 — 半分({int(shares//200*100)}株)利確を検討")
                    else:
                        notes.append(f"{current_r:.1f}R到達 — 1単元のため分割せず"
                                     "トレーリング+時間ストップで運用")
                    # ③ ストップの引き上げ先(★建値ではなく構造)
                    st = _structure_stop(df, 10, atr_now, cfg)
                    if st is not None and st > stop:
                        notes.append(f"ストップを構造({st:,.0f}円=直近10日安値の下)へ引き上げ検討"
                                     "。建値には上げない(早すぎる刈られを避けるため)")
                    hit = "partial"

            # ④ トレーリング水準の提示(+1R到達後の残玉向け)
            if hit in (None, "partial") and current_r is not None and current_r >= cfg.partial_tp_r:
                lvl, how = _trailing_level(trigger, df, atr_now, cfg)
                if lvl is not None:
                    notes.append(f"トレーリング水準 {lvl:,.0f}円({how})")

            # ⑤ 時間ストップ
            if hit is None and days is not None and days >= limit:
                hit = "time"
                notes.append(f"時間ストップ到達({trigger}: {days}営業日経過・上限{limit})"
                             " — 翌日寄りで手仕舞い検討")
            elif hit is None and days is not None and limit - days <= 2:
                notes.append(f"⚠時間ストップまで残り{limit - days}営業日")
            elif days is None and not entry_date:
                notes.append("entry_date未記録のため時間ストップ判定不可(要追記)")

            alerts.append({
                "code": code, "name": name, "hit": hit, "close": close_now,
                "days_held": days, "time_stop": limit, "trigger": trigger,
                "current_r": current_r,
                "note": " / ".join(notes) if notes else "平常(ストップ・+1R・時間ストップいずれも未到達)",
            })
        except Exception as e:
            print(f"[WARN] topdown保有チェックで例外(ticker={ticker}): {type(e).__name__}: {e}")
            alerts.append({"code": code, "name": name, "hit": None,
                           "note": "処理中に例外(コード表記等を要確認・Actionsログ参照)"})
    return alerts
