"""候補の自動追跡 — 手入力なしで実現R・MFE・MAEを貯める(2026-07-25).

plan_log(その日どの候補をどんなスコアで出したか)に載った銘柄を、翌日以降の
日次OHLCVで前向きに追い、v2.0の出口ルール(構造ストップ・+1R半利確・トレーリング・
時間ストップ)で「もし建てていたら」の実現Rを機械的に計算する。

なぜ必要か:
  - 実運用の手入力(result_log)は続きにくく、材料反応は週0〜2件でサンプルが貯まらない。
  - この自動追跡は「実際に建てなかった候補」も含めて全件を追えるので、
    期待度スコアと実現Rの相関を最短で検証できる。

重要な限界(結果を読むときに必ず意識する):
  - カタリストの中身を見ていない。好材料も悪材料も区別せず全件エントリーした場合の
    成績なので、これは実運用の「上限ではなく下限に近い参照値」。人間の確認フィルターが
    入る実運用はこれより良くなりうるが、それはこの数字からは分からない。
  - ゾーン方式と即時方式の両方を計算し、どちらが良かったかを後で比較できるようにする。
  - 日中の値動きは分からないため、ストップ・利確・トレーリングの判定はすべて「日足の
    高値/安値/終値」で行う近似。寄りギャップや値幅制限は再現しない。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .ta import atr_wilder, sma

TRACK_PATH = "topdown_track_log.csv"

TRACK_COLS = [
    "date_listed", "code", "name", "trigger", "tag", "score",
    "method",              # zone(ゾーン指値) / close(即時=提示日終値)
    "entry_price", "entry_date",
    "stop_initial", "risk_width",
    "exit_date", "exit_price", "exit_reason",   # stop/partial_then_trail/trail/time/open
    "bars_held",
    "realized_r", "mfe_r", "mae_r",
    "reached",             # ゾーンに到達して約定成立したか(zroのみ意味を持つ)
    "status",              # open / closed
]


def _f(v):
    try:
        s = str(v).strip()
        return float(s) if s and s.lower() != "nan" else None
    except Exception:
        return None


def _time_stop(trigger: str, cfg) -> int:
    return {"材料反応": cfg.time_stop_gap, "高値ブレイク": cfg.time_stop_break,
            "押し目": cfg.time_stop_pull}.get(trigger, cfg.time_stop_pull)


def _simulate(df_future: pd.DataFrame, entry: float, stop: float, trigger: str,
              cfg, atr_at_entry: float) -> dict:
    """建玉後の日足を1本ずつ辿り、v2.0の出口で手仕舞う。

    df_future: エントリー日の翌営業日以降のOHLCV(古い→新しい)
    戻り値: exit_price, exit_date, exit_reason, bars_held, realized_r, mfe_r, mae_r, status
    """
    risk = entry - stop
    if risk <= 0 or df_future is None or len(df_future) == 0:
        return {"status": "open", "exit_reason": "invalid", "realized_r": None,
                "mfe_r": None, "mae_r": None, "bars_held": 0,
                "exit_price": None, "exit_date": None}

    limit = _time_stop(trigger, cfg)
    partial_done = False
    cur_stop = stop
    mfe = 0.0     # 最大含み益(R)
    mae = 0.0     # 最大含み損(R)
    # 半利確した場合、残玉の実現Rは (利確分0.5*1R + 残り0.5*出口R)。単純化のため単元は2とみなす。
    booked_r = 0.0
    booked_frac = 0.0

    highs = df_future["High"].values
    lows = df_future["Low"].values
    closes = df_future["Close"].values
    idx = df_future.index

    for i in range(len(df_future)):
        hi, lo, cl = float(highs[i]), float(lows[i]), float(closes[i])
        mfe = max(mfe, (hi - entry) / risk)
        mae = min(mae, (lo - entry) / risk)

        # ① ストップ割れ(日中安値がストップ以下)。半利確済みなら残玉だけ決済。
        if lo <= cur_stop:
            exit_r = (cur_stop - entry) / risk
            r = booked_r + (1 - booked_frac) * exit_r
            return _res("closed", "partial_then_stop" if partial_done else "stop",
                        cur_stop, idx[i], i + 1, r, mfe, mae)

        # ② +1R到達 → 半分利確(まだなら)。以降ストップを建値=構造へ引上げ、トレーリング開始。
        if not partial_done and hi >= entry + cfg.partial_tp_r * risk:
            partial_done = True
            booked_r += 0.5 * cfg.partial_tp_r
            booked_frac = 0.5
            cur_stop = max(cur_stop, entry)   # 建値へ(簡易。構造の再計算はしない)

        # ③ トレーリング(半利確後の残玉)。トリガー別の水準を日足終値で更新。
        if partial_done:
            trail = _trail_level(df_future.iloc[:i + 1], trigger, atr_at_entry, cfg)
            if trail is not None:
                cur_stop = max(cur_stop, trail)

        # ④ 時間ストップ(その日の終値で手仕舞い)
        if i + 1 >= limit:
            exit_r = (cl - entry) / risk
            r = booked_r + (1 - booked_frac) * exit_r
            return _res("closed", "partial_then_time" if partial_done else "time",
                        cl, idx[i], i + 1, r, mfe, mae)

    # 追跡データが尽きた(まだ手仕舞い条件に達していない) → 未決済
    last_cl = float(closes[-1])
    unreal = booked_r + (1 - booked_frac) * (last_cl - entry) / risk
    return _res("open", "still_open", last_cl, idx[-1], len(df_future), unreal, mfe, mae)


def _res(status, reason, price, date, bars, r, mfe, mae):
    return {"status": status, "exit_reason": reason, "exit_price": round(price, 1),
            "exit_date": str(date.date()) if hasattr(date, "date") else str(date),
            "bars_held": bars, "realized_r": round(r, 3),
            "mfe_r": round(mfe, 3), "mae_r": round(mae, 3)}


def _trail_level(hist: pd.DataFrame, trigger: str, atr: float, cfg):
    try:
        c, h = hist["Close"].dropna(), hist["High"].dropna()
        if trigger == "材料反応":
            n = cfg.trail_chandelier_days
            if len(h) < 1:
                return None
            return float(h.iloc[-min(n, len(h)):].max()) - cfg.trail_chandelier_atr * atr
        if trigger == "高値ブレイク":
            return float(c.cummax().iloc[-1]) - cfg.trail_close_atr_mult * atr
        n = cfg.trail_ma_days
        if len(c) < n:
            return None
        return float(sma(c, n).iloc[-1])
    except Exception:
        return None


def load_plan_log(outdir: str) -> pd.DataFrame:
    p = Path(outdir) / "topdown_plan_log.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, dtype=str, encoding="utf-8-sig").fillna("")
        df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def _already(outdir: str):
    """追跡済み(date_listed, code, method)の集合。冪等な再計算のため。"""
    p = Path(outdir) / TRACK_PATH
    done = set()
    if p.exists():
        try:
            with p.open("r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    if row.get("status") == "closed":
                        done.add((row["date_listed"], row["code"], row["method"]))
        except Exception:
            pass
    return done, p


def track(outdir: str, ohlcv: Dict[str, pd.DataFrame], today: str, cfg) -> dict:
    """plan_log の各候補を追跡し、決済済みになったものを track_log に確定追記する。

    まだ決済に至らない候補は毎回再評価する(open行は書かず、closedになった時点で確定)。
    """
    plan = load_plan_log(outdir)
    if plan is None or len(plan) == 0:
        return {"tracked": 0, "closed": 0, "note": "plan_logが空"}

    done, path = _already(outdir)
    new_rows: List[dict] = []
    closed = 0

    for _, r in plan.iterrows():
        listed = str(r.get("date", "")).strip()
        code = str(r.get("code", "")).strip()
        ticker = code + ".T"
        trigger = str(r.get("trigger", "")).strip()
        df = ohlcv.get(ticker)
        if df is None or listed == "":
            continue
        try:
            after = df[df.index > pd.Timestamp(listed)]
        except Exception:
            continue
        if len(after) == 0:
            continue

        zone_hi = _f(r.get("zone_hi")); zone_lo = _f(r.get("zone_lo"))
        stop = _f(r.get("stop")); close0 = _f(r.get("close_at_listing"))
        atr0 = _f(r.get("atr")) or 0.0

        # --- ゾーン方式: 3営業日内に安値がゾーン上端以下へ来たら約定。約定価格はゾーン内 ---
        if (listed, code, "zone") not in done and zone_hi and zone_lo and stop:
            fill_i = None
            for i in range(min(cfg.zone_expire_days, len(after))):
                lo = float(after["Low"].values[i])
                cl = float(after["Close"].values[i])
                if cl < zone_lo:      # 終値が下端割れ → 失効
                    break
                if lo <= zone_hi:     # ゾーン到達
                    fill_i = i
                    entry = min(max(lo, zone_lo), zone_hi)
                    break
            if fill_i is not None:
                sim = _simulate(after.iloc[fill_i + 1:], entry, stop, trigger, cfg, atr0)
                if sim["status"] == "closed":
                    new_rows.append(_row(listed, r, "zone", entry,
                                         str(after.index[fill_i].date()), stop, sim, True))
                    closed += 1

        # --- 即時方式: 提示日の終値で即エントリー(反実仮想の対照) ---
        if (listed, code, "close") not in done and close0 and stop and close0 > stop:
            sim = _simulate(after, close0, stop, trigger, cfg, atr0)
            if sim["status"] == "closed":
                new_rows.append(_row(listed, r, "close", close0, listed, stop, sim, True))
                closed += 1

    if new_rows:
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=TRACK_COLS)
            if new_file:
                w.writeheader()
            for row in new_rows:
                w.writerow(row)

    return {"tracked": len(plan), "closed": closed,
            "note": f"新たに確定 {closed}件" if closed else "新規決済なし"}


def _row(listed, r, method, entry, entry_date, stop, sim, reached) -> dict:
    return {
        "date_listed": listed, "code": r.get("code", ""), "name": r.get("name", ""),
        "trigger": r.get("trigger", ""), "tag": r.get("tag", ""), "score": r.get("score", ""),
        "method": method, "entry_price": round(entry, 1), "entry_date": entry_date,
        "stop_initial": round(stop, 1), "risk_width": round(entry - stop, 1),
        "exit_date": sim["exit_date"], "exit_price": sim["exit_price"],
        "exit_reason": sim["exit_reason"], "bars_held": sim["bars_held"],
        "realized_r": sim["realized_r"], "mfe_r": sim["mfe_r"], "mae_r": sim["mae_r"],
        "reached": int(reached), "status": "closed",
    }


def summarize(outdir: str) -> dict:
    """track_log の要約(レポート表示・検証用)。"""
    p = Path(outdir) / TRACK_PATH
    if not p.exists():
        return {"n": 0}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {"n": 0}
    out = {"n": len(df)}
    for method in ("zone", "close"):
        sub = df[df["method"] == method]
        if len(sub):
            out[method] = {
                "n": len(sub),
                "avg_r": round(float(sub["realized_r"].mean()), 3),
                "win_rate": round(float((sub["realized_r"] > 0).mean()), 3),
                "avg_mfe": round(float(sub["mfe_r"].mean()), 3),
            }
    # トリガー別・期待度別の平均R(検証の中心)
    if len(df):
        by_trig = df.groupby("trigger")["realized_r"].agg(["count", "mean"]).round(3)
        out["by_trigger"] = {k: {"n": int(v["count"]), "avg_r": float(v["mean"])}
                             for k, v in by_trig.iterrows()}
    return out
