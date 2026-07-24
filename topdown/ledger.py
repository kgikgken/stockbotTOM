"""topdown 構造化ログ v2.0 — 検証設計に沿った記録項目.

仕様書v2.0「6.3 記録項目」に対応。特に次を後から検証できるようにする:
  - ゾーン指値方式 vs 即時エントリー方式(close_at_listing との比較)
  - 固定2Rなら取り逃していたテール(MFEのR分布)
  - 時間ストップの最適日数(保有日数別の追加リターン)
  - 小型ほど成績が良いか(売買代金 — 流動性フィルター引き下げ判断の材料)
"""

from __future__ import annotations

import csv
from pathlib import Path

PLAN_COLS = [
    "date", "code", "name", "sector", "trigger", "tag", "score", "score_reason",
    "zone_hi", "zone_lo", "stop", "risk_shallow", "risk_deep",
    "risk_pct_shallow", "risk_pct_deep", "atr", "unit_cost",
    "close_at_listing",      # 即時方式の約定価格(反実仮想)
    "expire_date", "time_stop", "zone_width_atr", "zone_floored",
    "tailwind", "headwind", "hivol", "gap_date", "earnings_est_days",
    "adv20_jpy",             # 売買代金(流動性フィルター判断の材料)
]

# 手仕舞い後に手入力する結果ログ。MFE/MAEはテール取り逃しの定量化に必須。
RESULT_COLS = [
    "date_listed", "date_entry", "code", "name", "trigger", "tag", "score",
    "entry_exec",            # 実際の約定価格
    "zone_hi", "zone_lo", "stop_initial", "risk_width",
    "close_at_listing",      # 即時方式ならいくらで入っていたか
    "shares", "units",
    "partial_tp_done(Y/N)", "partial_tp_price",
    "stop_raised_to", "exit_date", "exit_price",
    "exit_reason(stop/partial/trail/time/manual)",
    "realized_r", "mfe_r", "mae_r",
    "gap_through_stop",      # 寄りギャップで想定ストップをどれだけ割ったか
    "limit_move_blocked(Y/N)",  # 値幅制限で約定不能だったか
    "fee", "memo",
]

REJECT_COLS = ["date", "code", "name", "stage", "reason"]


def _num(v, nd=1):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return ""


def write_plan_log(outdir: str, today: str, picked: list) -> str:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    path = Path(outdir) / f"topdown_plan_log_{today}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(PLAN_COLS)
        for c in picked:
            feat = c.feat or {}
            w.writerow([
                today, c.code, c.name, c.sector, c.trigger, c.tag,
                f"{c.score:.0f}", c.score_reason,
                _num(c.zone_hi), _num(c.zone_lo), _num(c.stop),
                _num(c.risk_shallow), _num(c.risk_deep),
                _num(c.risk_pct_shallow, 2), _num(c.risk_pct_deep, 2),
                _num(feat.get("atr"), 2), _num(c.unit_cost, 0),
                _num(feat.get("close")),
                c.expire_date, c.time_stop,
                _num((c.zone_hi - c.zone_lo) / feat["atr"], 2) if feat.get("atr") else "",
                int(any("下限" in f for f in c.flags)),
                int(c.tailwind), int(c.headwind), int(c.hivol),
                c.gap_date or "", c.earnings_est_days if c.earnings_est_days is not None else "",
                _num(feat.get("adv20_jpy"), 0),
            ])
    return str(path)


def ensure_result_template(outdir: str) -> str:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    path = Path(outdir) / "topdown_result_log_template.csv"
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(RESULT_COLS)
    return str(path)


def append_reject_ledger(outdir: str, today: str, rejects: list) -> str:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    path = Path(outdir) / "topdown_reject_ledger.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(REJECT_COLS)
        for r in rejects:
            w.writerow([today, r.get("code", ""), r.get("name", ""),
                        r.get("stage", ""), r.get("reason", "")])
    return str(path)
