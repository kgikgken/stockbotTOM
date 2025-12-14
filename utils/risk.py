from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


COOLDOWN_PATH_DEFAULT = "cooldown_tp.csv"


@dataclass
class AbnormalFlag:
    flag: bool
    reasons: List[str]


def sl_cluster_filter(candidates: List[Dict], tol: float = 0.003) -> List[Dict]:
    """
    ⑥ SL相関クラスタ制御:
    sl_pct（例:-0.02）が近い銘柄は同時採用しない
    tol=0.003 は 0.3% 相当（小数）
    """
    kept: List[Dict] = []
    for c in candidates:
        sl = float(c.get("sl_pct", np.nan))
        if not np.isfinite(sl):
            kept.append(c)
            continue
        if all(abs(sl - float(k.get("sl_pct", sl))) > tol for k in kept):
            kept.append(c)
    return kept


def entry_unreachable(entry: float, price_now: float, max_gap: float = 0.015) -> bool:
    """
    ⑦ Entry未到達の自動見送り:
    現在が entry から +1.5% 超なら「追わない」確定。
    """
    if not (np.isfinite(entry) and np.isfinite(price_now) and entry > 0 and price_now > 0):
        return False
    return (price_now / entry - 1.0) > max_gap


def count_resistance_pivots(close: pd.Series, entry: float, tp_price: float, window: int = 2) -> int:
    """
    ⑧ RRの質（到達可能性）フィルタ用：抵抗帯数の簡易推定。
    - pivot high: i が近傍 window で局所最大
    - entry〜tp_price の範囲にある pivot の数をカウント
    """
    if close is None or len(close) < (window * 2 + 5):
        return 0
    c = close.astype(float).dropna()
    if len(c) < (window * 2 + 5):
        return 0

    lo = min(entry, tp_price)
    hi = max(entry, tp_price)
    if hi <= 0:
        return 0

    vals = c.values
    cnt = 0
    for i in range(window, len(vals) - window):
        v = vals[i]
        if not np.isfinite(v):
            continue
        if not (lo * 1.002 <= v <= hi * 0.998):
            continue
        left = vals[i - window:i]
        right = vals[i + 1:i + 1 + window]
        if np.all(v >= left) and np.all(v >= right):
            cnt += 1
    return int(cnt)


def rr_quality_ok(rr: float, resistance_count: int, max_res: int = 2) -> bool:
    """
    ⑧ RRの質フィルタ:
    resistance_count <= 2 を必須にする
    """
    if not np.isfinite(rr):
        return False
    return (rr >= 0.0) and (resistance_count <= max_res)


def load_cooldown(path: str = COOLDOWN_PATH_DEFAULT) -> Dict[str, date]:
    if not os.path.exists(path):
        return {}
    out: Dict[str, date] = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                t = str(row.get("ticker", "")).strip()
                d = str(row.get("last_tp_date", "")).strip()
                if not t or not d:
                    continue
                try:
                    out[t] = datetime.strptime(d, "%Y-%m-%d").date()
                except Exception:
                    continue
    except Exception:
        return {}
    return out


def save_cooldown(data: Dict[str, date], path: str = COOLDOWN_PATH_DEFAULT) -> None:
    rows = [{"ticker": t, "last_tp_date": d.strftime("%Y-%m-%d")} for t, d in sorted(data.items())]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "last_tp_date"])
        w.writeheader()
        w.writerows(rows)


def cooldown_ok(ticker: str, today: date, cooldown_map: Dict[str, date], cooldown_days: int = 3) -> bool:
    """
    ⑨ 利確後クールダウン: last_tp_date から cooldown_days 未満ならNG
    """
    last = cooldown_map.get(ticker)
    if last is None:
        return True
    return (today - last).days >= cooldown_days


def update_cooldown_if_tp_hit(ticker: str, today: date, price_now: float, tp_price: float,
                             cooldown_map: Dict[str, date]) -> None:
    """
    ⑨ last_tp_date 更新（近似）:
    price_now >= tp_price で「TP到達」とみなし、today を記録。
    """
    if not ticker:
        return
    if not (np.isfinite(price_now) and np.isfinite(tp_price)):
        return
    if tp_price <= 0:
        return
    if price_now >= tp_price:
        cooldown_map[ticker] = today


def downgrade_al(al: int, steps: int = 1) -> int:
    v = int(al) - int(steps)
    return max(0, v)
