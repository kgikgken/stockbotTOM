"""STEP5相当: モメンタム構造化ログ."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

PLAN_COLS = ["取引ID", "日付", "コード", "銘柄名", "セクター", "状態(A/B)",
             "モメンタムスコア", "エントリー", "初期ストップ", "シャンデリア水準",
             "リスク%", "リスク幅%", "レジーム", "候補プール規模", "ステータス"]

RESULT_COLS = ["取引ID", "実IN", "実決済", "実現損益", "実現R", "最大含み益到達R",
               "エグジット理由(初期ストップ/シャンデリア/裁量)", "実保有日数", "結果ラベル"]

REJECT_COLS = ["棄却日", "コード", "銘柄名", "ゲート", "理由", "棄却時終値"]


def write_plan_log(outdir: str, today: str, picked: list, regime: dict, pool_stats: dict) -> str:
    path = Path(outdir) / f"momentum_plan_log_{today}.csv"
    rows = []
    for i, c in enumerate(picked, 1):
        rows.append({
            "取引ID": f"MOM-{today}-{i:02d}", "日付": today, "コード": c.code, "銘柄名": c.name,
            "セクター": c.sector, "状態(A/B)": c.state, "モメンタムスコア": round(c.score, 2),
            "エントリー": c.entry, "初期ストップ": c.stop, "シャンデリア水準": c.chandelier,
            "リスク%": c.risk_pct, "リスク幅%": round(c.risk_w / c.entry * 100, 2),
            "レジーム": regime.get("mode", "-"), "候補プール規模": pool_stats.get("pool_size", 0),
            "ステータス": "仮点灯(未確認・単一ソース)",
        })
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_COLS)
        w.writeheader()
        w.writerows(rows)
    return str(path)


def ensure_result_template(outdir: str) -> str:
    path = Path(outdir) / "momentum_result_log_template.csv"
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(RESULT_COLS)
    return str(path)


def append_reject_ledger(outdir: str, today: str, rejects: List[dict]) -> str:
    path = Path(outdir) / "momentum_reject_ledger.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(REJECT_COLS)
        for r in rejects:
            w.writerow([today, r["code"], r["name"], r["stage"], r["reason"], r.get("close", "")])
    return str(path)
