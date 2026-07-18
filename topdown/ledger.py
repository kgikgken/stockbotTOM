"""topdown 構造化ログ — 計画ログ(日次)+結果ログ雛形(検証用)。
将来の有効性検証(実現R集計)のため、本命候補を毎日1行ずつ記録する。
"""

from __future__ import annotations

import csv
from pathlib import Path


PLAN_COLS = ["date", "code", "name", "sector", "trigger", "tag", "confidence",
             "entry", "stop", "target", "time_stop", "risk_pct", "shares",
             "tailwind", "headwind", "hivol", "score"]

RESULT_COLS = ["date_entry", "code", "name", "trigger", "tag", "confidence",
               "entry_exec", "exit_date", "exit_price", "exit_reason(target/stop/time/manual)",
               "realized_r", "memo"]


def write_plan_log(outdir: str, today: str, picked: list) -> str:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    path = Path(outdir) / f"topdown_plan_log_{today}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(PLAN_COLS)
        for c in picked:
            w.writerow([today, c.code, c.name, c.sector, c.trigger, c.tag, c.confidence,
                        c.entry, c.stop, c.target, c.time_stop, c.risk_pct, c.shares,
                        int(c.tailwind), int(c.headwind), int(c.hivol), round(c.score, 2)])
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
            w.writerow(["date", "code", "name", "stage", "reason"])
        for r in rejects:
            w.writerow([today, r.get("code", ""), r.get("name", ""), r.get("stage", ""), r.get("reason", "")])
    return str(path)
