"""v6 Layer 7 — 出口ルール。固定の目標株価は設定しない。

旧仕様からの最大の変更: 「一律N営業日で区切る」を廃止した。トレンドフォローの利益は
少数の大きく伸びたトレードから生まれるため、順行中のポジションを日数で切ってはならない。
代わりに「動かないポジションだけ」を25日で切る(条件7)。

条件3(日足ステージ2への移行)が実質的なトレーリングの主軸。エントリーとエグジットを
同じステージ理論で扱うことで判断の一貫性が保たれる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .filters import ema, weekly_stage, daily_stage

# 優先度順(小さいほど優先)
EXIT_RULES = [
    (1, "ストップ割れ", "翌営業日寄付で全売却"),
    (2, "週足ステージ3/4へ移行", "翌営業日寄付で全売却"),
    (3, "日足ステージ2へ移行", "翌営業日寄付で全売却"),
    (4, "10日EMA割れが2日連続", "翌営業日寄付で全売却"),
    (5, "+1R到達", "ストップを建値へ引き上げ"),
    (6, "+2R到達", "建玉の1/3を利確"),
    (7, "25日経過かつ±0.5R以内", "時間切れとして手仕舞い"),
]


def evaluate_exit(daily: pd.DataFrame, position: dict, cfg, today: str) -> dict:
    """保有中ポジションについて出口条件を評価する。

    position: {entry, stop, shares, entry_date, partial_done(bool), be_moved(bool)}
    戻り値: {"hits": [(優先度, 条件名, 動作), ...], "top": 最優先の該当, "metrics": {...}}
    """
    c = daily["Close"].dropna()
    if len(c) < 60:
        return {"hits": [], "top": None, "metrics": {}, "error": "データ不足"}

    close_now = float(c.iloc[-1])
    entry = float(position["entry"])
    stop = float(position["stop"])
    risk = entry - stop
    r_now = (close_now - entry) / risk if risk > 0 else 0.0

    hits = []

    # ① ストップ割れ(終値判定 → 翌営業日寄付)
    if close_now < stop:
        hits.append((1, "ストップ割れ",
                     f"終値{close_now:,.0f} < ストップ{stop:,.0f} → 翌営業日寄付で全売却"))

    # ② 週足ステージが3/4へ移行
    ws = weekly_stage(daily, cfg)
    if ws and ws["stage_w"] in (3, 4):
        hits.append((2, "週足ステージ悪化",
                     f"週足ステージ{ws['stage_w']}へ移行 → 翌営業日寄付で全売却"))

    # ③ 日足ステージが2へ移行(5日線が20日線を下抜け)
    ds = daily_stage(daily, cfg)
    if ds and ds["stage_d"] == 2:
        hits.append((3, "日足ステージ2へ移行",
                     "5日線が20日線を下抜け → 翌営業日寄付で全売却"))

    # ④ 10日EMA割れが2営業日連続
    e = ema(c, cfg.exit_ema_days)
    if len(e) >= cfg.exit_ema_consecutive:
        below = [float(c.iloc[-i]) < float(e.iloc[-i])
                 for i in range(1, cfg.exit_ema_consecutive + 1)]
        if all(below):
            hits.append((4, f"{cfg.exit_ema_days}日EMA割れ{cfg.exit_ema_consecutive}日連続",
                         "翌営業日寄付で全売却"))

    # ⑤ +1R到達 → ストップを建値へ
    if r_now >= 1.0 and not position.get("be_moved"):
        hits.append((5, "+1R到達", f"ストップを建値{entry:,.0f}円へ引き上げ"))

    # ⑥ +2R到達 → 1/3利確
    if r_now >= cfg.partial_tp_r and not position.get("partial_done"):
        n = int(position.get("shares", 0) * cfg.partial_tp_frac // cfg.unit_shares) * cfg.unit_shares
        hits.append((6, f"+{cfg.partial_tp_r:.0f}R到達",
                     f"建玉の1/3({n}株)を利確、残りは条件3・4で追随"))

    # ⑦ 時間切れ(動かない場合のみ)
    days = _bdays(position.get("entry_date"), today)
    if days is not None and days >= cfg.exit_time_stop_days and abs(r_now) <= cfg.exit_time_stop_r:
        hits.append((7, "時間切れ",
                     f"{days}営業日経過で{r_now:+.2f}R(±{cfg.exit_time_stop_r}R以内) → 手仕舞い"))

    hits.sort(key=lambda x: x[0])
    return {
        "hits": hits,
        "top": hits[0] if hits else None,
        "metrics": {"close": close_now, "r": r_now, "days_held": days,
                    "stage_w": ws["stage_w"] if ws else None,
                    "stage_d": ds["stage_d"] if ds else None},
    }


def _bdays(entry_date, today) -> int | None:
    try:
        d0 = pd.Timestamp(entry_date).date()
        d1 = pd.Timestamp(today).date()
        return int(np.busday_count(d0, d1)) if d1 >= d0 else None
    except Exception:
        return None
