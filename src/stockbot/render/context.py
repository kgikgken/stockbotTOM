"""表示内容の組み立て（docs/SCREENER.md §4.5）。

配信記録（`delivered_*.csv`）とその日の要約（`screen_summary_*.json`）から、
テンプレートに渡す値だけを作る。**ここでは株価も指標も計算し直さない。**
描画を差し替えても表示内容が変わらないよう、整形はすべてこの層に閉じる（§4.3）。

Playwright も Jinja2 も import しないので、ブラウザ無しでテストできる。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..screener.conditions import CONDITION_LABELS

TOP_FAILS = 3
MA_LABELS = {"SMA5": "5日線", "SMA25": "25日線", "SMA75": "75日線", "SMA200": "200日線"}

# 各カードに固定で載せる出口ルール（SPEC.md §1 の出口④に対応。銘柄ごとに変えない）
EXIT_RULE = "押し安値割れで撤退／5日線回復で手仕舞い"
# 健全性ブロックの注記。A4 のカバー率が低い日でも読み手が誤解しないように固定で出す
EARNINGS_NOTE = "決算日が取れない銘柄は、決算前でも候補に出ます。"
DISCLAIMER = "AI候補提示で投資助言ではない。最終判断と結果責任はユーザーにある。"
ORDER_NOTE = "並びは20日平均売買代金の降順。順位ではありません。"


def _f(value) -> Optional[float]:
    """欠損を None に潰す（テンプレート側で分岐しやすくする）。"""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _yen(value) -> str:
    v = _f(value)
    return "—" if v is None else f"{v:,.1f}"


def _pct(value, digits: int = 2) -> str:
    v = _f(value)
    return "—" if v is None else f"{v:.{digits}f}%"


def _signed_pct(value, digits: int = 1) -> str:
    v = _f(value)
    return "—" if v is None else f"{v:+.{digits}f}%"


def _gap_pct(target, close) -> Optional[float]:
    """終値から見た乖離率（%）。上なら +、下なら −。"""
    t, c = _f(target), _f(close)
    if t is None or c is None or c == 0:
        return None
    return (t - c) / c * 100.0


def _earnings(row) -> dict:
    """決算までの日数。取れていない銘柄は赤字で明示する（§2.6）。"""
    if bool(row.get("a4_earnings_unknown")):
        return {"text": "決算日未取得", "unknown": True}
    days = _f(row.get("earnings_days"))
    if days is None:
        return {"text": "決算日未取得", "unknown": True}
    return {"text": f"決算まで{int(days)}営業日", "unknown": False}


def _streak(row) -> Optional[str]:
    try:
        n = int(row.get("streak") or 1)
    except (TypeError, ValueError):
        return None
    return f"連続{n}日目" if n > 1 else None


def build_card(rank: int, row) -> dict:
    """1銘柄ぶんの表示内容。値はすべて配信記録の列から来る（§4.3）。"""
    close = _f(row.get("close_t"))
    lp, h0 = _f(row.get("lp")), _f(row.get("h0_high"))
    return {
        "rank": rank,
        "ticker": str(row.get("ticker") or ""),
        "name": str(row.get("name") or ""),
        "sector33": str(row.get("sector33") or ""),
        "state": str(row.get("state") or "—"),
        "close": _yen(close),
        "landing_ma": MA_LABELS.get(str(row.get("landing_ma") or ""), "—"),
        "landing_dist": (f"{_f(row.get('landing_dist_atr')):.2f} ATR"
                         if _f(row.get("landing_dist_atr")) is not None else "—"),
        "lp": _yen(lp),
        "lp_gap": _signed_pct(_gap_pct(lp, close)),
        "h0_high": _yen(h0),
        "h0_gap": _signed_pct(_gap_pct(h0, close)),
        "depth": _pct(_f(row.get("depth_pct")) * 100 if _f(row.get("depth_pct")) is not None else None),
        "pullback_days": (f"{int(row['pullback_days'])}日"
                          if _f(row.get("pullback_days")) is not None else "—"),
        "adv": (f"{_f(row.get('adv_jpy')) / 1e8:,.1f}億円"
                if _f(row.get("adv_jpy")) is not None else "—"),
        "earnings": _earnings(row),
        "streak": _streak(row),
        "exit_rule": EXIT_RULE,
    }


def build_health(summary: dict) -> dict:
    """健全性ブロック。取得成功率と決算日カバー率（§4.5）。"""
    ok, total = summary.get("fetch_ok"), summary.get("fetch_total")
    rate = (ok / total) if _f(ok) is not None and _f(total) not in (None, 0) else None
    cov = _f(summary.get("earnings_coverage"))
    return {
        "n_evaluated": int(summary.get("n_evaluated") or 0),
        "fetch_text": (f"{int(ok):,}/{int(total):,}（{rate:.1%}）"
                       if rate is not None else "—"),
        "earnings_text": (f"{int(summary.get('earnings_known') or 0):,}"
                          f"/{int(summary.get('n_evaluated') or 0):,}"
                          + (f"（{cov:.1%}）" if cov is not None else "")),
        "earnings_note": EARNINGS_NOTE,
    }


def build_zero_day(summary: dict) -> dict:
    """候補0件の日に出す内訳。落ちた条件の上位3つと延べ件数（§4.2）。"""
    fails = summary.get("fail_counts") or {}
    top = sorted(fails.items(), key=lambda kv: -kv[1])[:TOP_FAILS]
    return {
        "n_evaluated": int(summary.get("n_evaluated") or 0),
        "total": int(sum(fails.values())) if fails else 0,
        "rows": [{"cid": cid, "label": CONDITION_LABELS.get(cid, ""), "n": int(n)}
                 for cid, n in top],
    }


def build_context(delivered: Optional[pd.DataFrame], summary: dict) -> dict:
    """テンプレートに渡す全体（§4.5）。

    delivered は screener.record.load_delivered() の出力（0行でもよい）。
    summary は screener.screen.build_summary() が書いた JSON を読んだ dict。
    """
    n = 0 if delivered is None else len(delivered)
    score = summary.get("regime_score")
    cards = ([build_card(i, row) for i, (_idx, row) in enumerate(delivered.iterrows(), start=1)]
             if n else [])
    return {
        "delivered_on": str(summary.get("delivered_on") or ""),
        "asof": str(summary.get("asof") or ""),
        "regime": {
            "level": summary.get("regime_level") or "不明",
            "score": score,
            "text": (f"{summary.get('regime_level') or '不明'}（{score}/6）"
                     if score is not None else (summary.get("regime_level") or "不明")),
        },
        "n_candidates": n,
        "health": build_health(summary),
        # 候補0件の日は E1 の注記を出さない（適用する対象が無い、§4.2）
        "e1_note": (f"E1（相対力の上位10%除外）は母集団 {summary.get('n_pool', 0)}件のため本日は未適用"
                    if n and summary.get("e1_skipped") else None),
        "cards": cards,
        "zero_day": None if n else build_zero_day(summary),
        "order_note": ORDER_NOTE,
        "disclaimer": DISCLAIMER,
    }
