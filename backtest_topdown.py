"""バックテスト — 過去データで設計ルールを検証する(2026-07-25).

目的を3つに絞る。調査(2026-07-25のレポート)で「反証あり」「内部矛盾」と判定された
項目だけを検証する。総合成績を出して一喜一憂するためのものではない。

  Q1. 流動性フィルター5億円は正しいか
      日本のPEADは小型・低注目株に残存する(Okada&Saeki 2014, Jinushi 2023)のに、
      5億円フィルターがその小型株を除外している。売買代金の帯ごとに実現Rを比較する。

  Q2. ゾーン指値で待つのは正しいか
      Bulkowskiは「押し戻しを待つと最良のトレードを逃す」と明言している。
      同じ候補を「ゾーン指値」と「即時(提示日終値)」の両方で建てて比較する。

  Q3. +1Rの半分利確は期待Rを下げるか
      数学的には下がるはず(1:2設計で2.0R→1.5R)。実データで確かめる。

【先読みバイアスの防止】
各営業日tについて、その日までのデータだけを渡して候補を選び、t+1以降で決済を判定する。
未来の値を参照する余地を構造的に無くすため、スライスは必ず df.loc[:t] で行う。

【結果を読むときの限界(必ず意識する)】
- カタリストの中身を見ていない。好材料も悪材料も区別せず全件エントリーした成績。
  実運用は人間が悪材料を外すのでこれより良くなりうるが、その差はこの数字からは分からない。
- yfinanceは上場廃止銘柄を欠く。生存バイアスにより結果は上振れる方向。
- 日中の値動きは分からない。約定・ストップ判定は日足の高値/安値/終値による近似。
- 手数料・スリッページ・値幅制限・寄りギャップは再現しない。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from topdown.config import load_config
from topdown.indicators import compute_topdown_features
from topdown.screen import build_zone, TRIG_GAP, TRIG_PULL, TRIG_BREAK
from topdown.ta import sma


@dataclass
class Trade:
    date: str
    code: str
    trigger: str
    method: str          # zone / immediate
    partial: str         # on / off
    adv_bucket: str      # 売買代金の帯
    entry: float
    stop: float
    exit_price: float
    exit_reason: str
    bars: int
    r: float
    mfe: float
    mae: float


def _time_stop(trigger: str, cfg) -> int:
    return {TRIG_GAP: cfg.time_stop_gap, TRIG_BREAK: cfg.time_stop_break,
            TRIG_PULL: cfg.time_stop_pull}.get(trigger, cfg.time_stop_pull)


def _trail(hist: pd.DataFrame, trigger: str, atr: float, cfg):
    """トリガー別のトレーリング水準。histはその時点までの履歴のみ。"""
    try:
        c, h = hist["Close"], hist["High"]
        if trigger == TRIG_GAP:
            n = min(cfg.trail_chandelier_days, len(h))
            return float(h.iloc[-n:].max()) - cfg.trail_chandelier_atr * atr
        if trigger == TRIG_BREAK:
            return float(c.max()) - cfg.trail_close_atr_mult * atr
        n = cfg.trail_ma_days
        if len(c) < n:
            return None
        return float(sma(c, n).iloc[-1])
    except Exception:
        return None


def simulate(future: pd.DataFrame, entry: float, stop: float, trigger: str,
             cfg, atr: float, partial_on: bool):
    """建玉後を1日ずつ辿って手仕舞う。futureはエントリー翌日以降のみ(未来参照なし)。"""
    risk = entry - stop
    if risk <= 0 or len(future) == 0:
        return None
    limit = _time_stop(trigger, cfg)
    cur_stop = stop
    partial_done = False
    booked_r, booked_frac = 0.0, 0.0
    mfe, mae = 0.0, 0.0

    for i in range(len(future)):
        hi = float(future["High"].values[i])
        lo = float(future["Low"].values[i])
        cl = float(future["Close"].values[i])
        mfe = max(mfe, (hi - entry) / risk)
        mae = min(mae, (lo - entry) / risk)

        if lo <= cur_stop:
            r = booked_r + (1 - booked_frac) * (cur_stop - entry) / risk
            reason = "partial_then_stop" if partial_done else "stop"
            return dict(exit=cur_stop, reason=reason, bars=i + 1, r=r, mfe=mfe, mae=mae)

        if partial_on and not partial_done and hi >= entry + cfg.partial_tp_r * risk:
            partial_done = True
            booked_r += 0.5 * cfg.partial_tp_r
            booked_frac = 0.5
            cur_stop = max(cur_stop, entry)

        # トレーリングは「半分利確後」または「利確なし運用で+1R超え後」に開始
        started = partial_done or (not partial_on and mfe >= cfg.partial_tp_r)
        if started:
            lvl = _trail(future.iloc[:i + 1], trigger, atr, cfg)
            if lvl is not None:
                cur_stop = max(cur_stop, lvl)

        if i + 1 >= limit:
            r = booked_r + (1 - booked_frac) * (cl - entry) / risk
            reason = "partial_then_time" if partial_done else "time"
            return dict(exit=cl, reason=reason, bars=i + 1, r=r, mfe=mfe, mae=mae)

    cl = float(future["Close"].values[-1])
    r = booked_r + (1 - booked_frac) * (cl - entry) / risk
    return dict(exit=cl, reason="unfinished", bars=len(future), r=r, mfe=mfe, mae=mae)


def _bucket(adv: float) -> str:
    """売買代金の帯。Q1(流動性フィルターの妥当性)の検証軸。"""
    if adv < 1e8: return "A: 1億未満"
    if adv < 2e8: return "B: 1-2億"
    if adv < 5e8: return "C: 2-5億"
    if adv < 2e9: return "D: 5-20億"
    return "E: 20億以上"


def run(ohlcv: Dict[str, pd.DataFrame], cfg, start: str, end: str,
        min_adv: float, step: int = 1) -> List[Trade]:
    """営業日ごとに候補を選び、その後を追う。

    min_adv を引数にしたのは Q1 の検証のため(本番の5億円を緩めて小型を含める)。
    """
    dates = None
    for df in ohlcv.values():
        idx = df.loc[start:end].index
        dates = idx if dates is None else dates.union(idx)
    if dates is None or len(dates) == 0:
        return []
    dates = sorted(dates)[::step]

    trades: List[Trade] = []
    for t in dates:
        for ticker, df in ohlcv.items():
            hist = df.loc[:t]                       # ★その日までのデータのみ
            if len(hist) < 260:
                continue
            feat = compute_topdown_features(hist, cfg)
            if feat is None:
                continue
            if feat["close"] < cfg.min_price or feat["close"] > cfg.max_price:
                continue
            if feat["adv20_jpy"] < min_adv:
                continue
            if feat.get("spiked"):
                continue

            if feat.get("gap_found"):
                trigger = TRIG_GAP
            elif feat.get("pullback_state_a"):
                trigger = TRIG_PULL
            elif feat.get("breakout_found"):
                trigger = TRIG_BREAK        # 順風条件はバックテストでは省く(セクター再現が困難)
            else:
                continue

            zone, _ = build_zone(trigger, feat, cfg)
            if zone is None:
                continue
            zone_hi, zone_lo, stop, _ = zone
            atr = feat["atr"]
            code = ticker.replace(".T", "")
            bucket = _bucket(feat["adv20_jpy"])
            after = df.loc[df.index > t]
            if len(after) < 5:
                continue

            for partial_on in (True, False):
                ptag = "on" if partial_on else "off"

                # --- ゾーン指値(3営業日内に到達したら約定) ---
                fill_i, entry_z = None, None
                for i in range(min(cfg.zone_expire_days, len(after))):
                    lo = float(after["Low"].values[i])
                    cl = float(after["Close"].values[i])
                    if cl < zone_lo:
                        break
                    if lo <= zone_hi:
                        fill_i = i
                        entry_z = min(max(lo, zone_lo), zone_hi)
                        break
                if fill_i is not None:
                    s = simulate(after.iloc[fill_i + 1:], entry_z, stop, trigger, cfg, atr, partial_on)
                    if s:
                        trades.append(Trade(str(t.date()), code, trigger, "zone", ptag, bucket,
                                            entry_z, stop, s["exit"], s["reason"],
                                            s["bars"], s["r"], s["mfe"], s["mae"]))

                # --- 即時(提示日の終値で建てる) ---
                entry_i = feat["close"]
                if entry_i > stop:
                    s = simulate(after, entry_i, stop, trigger, cfg, atr, partial_on)
                    if s:
                        trades.append(Trade(str(t.date()), code, trigger, "immediate", ptag, bucket,
                                            entry_i, stop, s["exit"], s["reason"],
                                            s["bars"], s["r"], s["mfe"], s["mae"]))
    return trades


def report(trades: List[Trade]) -> str:
    if not trades:
        return "トレードなし"
    df = pd.DataFrame([t.__dict__ for t in trades])
    L = []
    ap = L.append
    ap(f"総トレード {len(df)}件 / 銘柄 {df['code'].nunique()} / "
       f"期間 {df['date'].min()} 〜 {df['date'].max()}")
    ap("")

    def agg(g):
        return pd.Series({
            "件数": len(g),
            "平均R": round(g["r"].mean(), 3),
            "中央R": round(g["r"].median(), 3),
            "勝率": round((g["r"] > 0).mean(), 3),
            "平均MFE": round(g["mfe"].mean(), 3),
        })

    ap("【Q2】ゾーン指値 vs 即時エントリー")
    ap(df[df["partial"] == "off"].groupby("method").apply(agg).to_string())
    ap("")
    ap("【Q3】+1R半分利確 あり vs なし")
    ap(df[df["method"] == "zone"].groupby("partial").apply(agg).to_string())
    ap("")
    ap("【Q1】売買代金の帯ごと(材料反応のみ・ゾーン・利確なし)")
    sub = df[(df["trigger"] == TRIG_GAP) & (df["method"] == "zone") & (df["partial"] == "off")]
    if len(sub):
        ap(sub.groupby("adv_bucket").apply(agg).to_string())
    else:
        ap("  材料反応のサンプルなし")
    ap("")
    ap("【参考】トリガー別(ゾーン・利確なし)")
    ap(df[(df["method"] == "zone") & (df["partial"] == "off")]
       .groupby("trigger").apply(agg).to_string())
    ap("")
    ap("【参考】手仕舞い理由の内訳(ゾーン・利確なし)")
    ap(df[(df["method"] == "zone") & (df["partial"] == "off")]["exit_reason"]
       .value_counts().to_string())
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--min-adv", type=float, default=1e8,
                   help="売買代金の下限(既定1億円。本番の5億より緩めてQ1を検証)")
    p.add_argument("--universe", default="universe_jpx.csv")
    p.add_argument("--limit", type=int, default=0, help="銘柄数の上限(0=全件)")
    p.add_argument("--step", type=int, default=5, help="何営業日ごとに判定するか")
    p.add_argument("--out", default="backtest_result.csv")
    args = p.parse_args()

    cfg = load_config()
    from topdown.universe import load_universe
    from topdown.data import fetch_ohlcv

    uni = load_universe(args.universe)
    if args.limit:
        uni = uni.head(args.limit)
    tickers = uni["ticker"].tolist()
    print(f"[1/3] ユニバース {len(tickers)}銘柄 / 期間 {args.start}〜{args.end}")

    # 取得範囲は「今日から遡る日数」で指定する仕様なので、開始日を判定するのに必要な
    # 助走期間(200日移動平均のため約260営業日≒380暦日)を start より前に確保する。
    # これを忘れると開始年の前半が「履歴不足」で静かにスキップされる(2026-07-25に発見)。
    warmup_days = 400
    lead = pd.Timestamp(args.start) - pd.Timedelta(days=warmup_days)
    days = (pd.Timestamp.today().normalize() - lead).days
    ohlcv, meta = fetch_ohlcv(tickers, days, dryrun=cfg.dryrun)
    print(f"[2/3] OHLCV取得 {meta.get('data_ok','?')}/{meta.get('data_total','?')} "
          f"(助走含め {lead.date()} 以降)")

    # 助走が実際に足りているかを点検し、足りない銘柄が多ければ警告する
    short = sum(1 for df in ohlcv.values()
                if len(df.loc[:args.start]) < 260)
    if short:
        print(f"      ⚠ 開始日時点で履歴260行未満の銘柄 {short}/{len(ohlcv)}件"
              f" — その銘柄は開始直後の期間が判定されない")

    trades = run(ohlcv, cfg, args.start, args.end, args.min_adv, args.step)
    print(f"[3/3] トレード {len(trades)}件")

    if trades:
        pd.DataFrame([t.__dict__ for t in trades]).to_csv(args.out, index=False,
                                                          encoding="utf-8-sig")
        print(f"      → {args.out}")
    print()
    print(report(trades))
    print()
    print("※カタリスト未確認・生存バイアスあり・手数料/スリッページ未考慮。")
    print("  絶対値ではなく『AとBのどちらが良いか』の比較にのみ使うこと。")


if __name__ == "__main__":
    main()
