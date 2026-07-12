"""バックテスト結果の集計指標。調査レポート(2026-07-11「期待値評価」)が最低限必要とした
指標(勝率・期待R・プロフィットファクター・最大DD・トレード数)を計算する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult


def summarize(result: BacktestResult) -> dict:
    trades = [t for t in result.trades if t.r_multiple is not None]
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "note": "トレードが0件のため集計不可(閾値が厳しすぎる/期間が短い可能性)"}

    rs = np.array([t.r_multiple for t in trades])
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    win_rate = len(wins) / n
    expected_r = float(rs.mean())
    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf")

    # 累積R推移から最大ドローダウン(R換算)を計算
    cum = np.cumsum(rs)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    max_dd_r = float(dd.min()) if len(dd) else 0.0

    by_reason = pd.Series([t.exit_reason for t in trades]).value_counts().to_dict()
    by_engine = pd.Series([t.engine for t in trades]).value_counts().to_dict()
    days_held = [t.days_held for t in trades if t.days_held is not None]

    return {
        "n_trades": n,
        "win_rate": round(win_rate, 3),
        "expected_r": round(expected_r, 3),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf(負けトレード無し)",
        "max_drawdown_r": round(max_dd_r, 2),
        "avg_days_held": round(float(np.mean(days_held)), 1) if days_held else None,
        "exit_reason_breakdown": by_reason,
        "engine_breakdown": by_engine,
        "skipped_gap_chase": result.skipped_gap_chase,
        "signals_fired_total": result.signals_fired,
        "note": ("サンプルサイズ" + (
            "が100件超で最低限の目安を満たす" if n >= 100 else f"が{n}件と少なく、統計的な結論は時期尚早(目安100件超)")),
    }


def trades_to_dataframe(result: BacktestResult) -> pd.DataFrame:
    rows = []
    for t in result.trades:
        rows.append({
            "ticker": t.ticker, "engine": t.engine, "entry_date": t.entry_date,
            "entry_price": t.entry_price, "planned_entry": t.planned_entry, "stop": t.stop,
            "target": t.target, "shares": t.shares, "exit_date": t.exit_date,
            "exit_price": t.exit_price, "exit_reason": t.exit_reason,
            "r_multiple": round(t.r_multiple, 3) if t.r_multiple is not None else None,
            "days_held": t.days_held,
        })
    return pd.DataFrame(rows)
