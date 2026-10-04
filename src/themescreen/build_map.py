"""Build the semiconductor front-end industry map from saved raw data.

Reads data/theme/raw/<asof>/ (written by fetch.py) and the universe CSV; writes
data/theme/out/semicon_frontend_<asof>.csv and .md (docs/theme/README.md). No network.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import OUT_DIR, RAW_DIR, UNIVERSE_DIR
from .metrics import (
    PROCESSES, cutoff, latest_annual, market_cap, median_trading_value,
    next_period_end_passed,
)

OK, NG = "計算可", "計算不可"


def build(asof: str, universe_path: Path) -> pd.DataFrame:
    """One row per company with computed values, reasons and cutoff status."""
    with open(universe_path, encoding="utf-8") as f:
        universe = list(csv.DictReader(f))
    raw = RAW_DIR / asof
    meta = json.loads((raw / "info.json").read_text(encoding="utf-8"))
    recs = {r["code"]: r for r in meta["records"]}
    bars = pd.read_csv(raw / "history.csv", dtype={"code": str, "date": str})
    rows: list[dict] = []
    for u in universe:
        code = u["code"]
        rec = recs.get(code, {})
        info = rec.get("info")
        mc = market_cap(info)
        tv = median_trading_value(bars.loc[bars["code"] == code], asof)
        income = rec.get("income_annual") or {}
        op_end, op = latest_annual(income.get("Operating Income"))
        rep_end, rep = latest_annual(income.get("Total Operating Income As Reported"))
        status, reasons = cutoff(tv, op)
        rows.append({
            "code": code, "name": u["name"], "process": u["process"],
            "basis": u["basis"], "other_processes": u["other_processes"],
            "yf_long_name": (info or {}).get("longName"),
            "market_cap_jpy": mc.value, "market_cap_reason": mc.reason,
            "median_tv_60d_jpy": tv.value, "median_tv_reason": tv.reason,
            "op_period_end": op_end, "op_income_jpy": op.value, "op_reason": op.reason,
            "op_as_reported_period_end": rep_end, "op_as_reported_jpy": rep.value,
            "status": status, "exclusion_reasons": ";".join(reasons),
        })
    df = pd.DataFrame(rows)
    df["process"] = pd.Categorical(df["process"], categories=list(PROCESSES), ordered=True)
    return df.sort_values(["process", "code"]).reset_index(drop=True)


def _oku(v: float | None, digits: int) -> str:
    if v is None or pd.isna(v):
        return "計算不可"
    return f"{v / 1e8:,.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render(df: pd.DataFrame, asof: str) -> str:
    """Markdown with tables only."""
    n = len(df)
    mc_ok = int(df["market_cap_jpy"].notna().sum())
    tv_ok = int(df["median_tv_60d_jpy"].notna().sum())
    op_ok = int(df["op_income_jpy"].notna().sum())
    both = df["op_income_jpy"].notna() & df["op_as_reported_jpy"].notna() & (
        df["op_period_end"] == df["op_as_reported_period_end"])
    sign_diff = int(((df.loc[both, "op_income_jpy"] < 0)
                     != (df.loc[both, "op_as_reported_jpy"] < 0)).sum())
    st = Counter(df["status"])
    parts: list[str] = []
    parts.append("### 計算可能性\n\n" + _table(
        ["項目", "計算可", "計算不可"],
        [["時価総額", mc_ok, n - mc_ok],
         ["60日中央値売買代金", tv_ok, n - tv_ok],
         ["直近期の営業損益", op_ok, n - op_ok]]))
    parts.append("### 足切りの結果\n\n" + _table(
        ["列挙", "掲載", "除外", "判定不能"],
        [[n, st.get("掲載", 0), st.get("除外", 0), st.get("判定不能", 0)]]))
    shown = df[df["status"] == "掲載"]
    parts.append(f"### 工程別の表（基準日 {asof}）\n\n" + _table(
        ["証券コード", "社名", "工程", "時価総額（億円）", "60日中央値売買代金（億円）"],
        [[r.code, r.name, r.process, _oku(r.market_cap_jpy, 0), _oku(r.median_tv_60d_jpy, 2)]
         for r in shown.itertuples()]))
    ex = df[df["status"] == "除外"]
    parts.append("### 足切りで除外\n\n" + _table(
        ["証券コード", "社名", "工程", "60日中央値売買代金（億円）", "直近期の営業利益（億円）", "直近期", "理由"],
        [[r.code, r.name, r.process, _oku(r.median_tv_60d_jpy, 2), _oku(r.op_income_jpy, 1),
          r.op_period_end, r.exclusion_reasons.replace(";", "・")] for r in ex.itertuples()]))
    ng_rows: list[list[str]] = []
    for r in df.itertuples():
        for item, reason in (("時価総額", r.market_cap_reason),
                             ("60日中央値売買代金", r.median_tv_reason),
                             ("直近期の営業損益", r.op_reason)):
            if reason:
                ng_rows.append([r.code, r.name, item, reason, r.status])
    parts.append("### 計算不可\n\n" + _table(
        ["証券コード", "社名", "項目", "理由", "足切り"], ng_rows))
    ends = Counter(df["op_period_end"].dropna())
    parts.append("### 直近期の期末日（yfinance 通期損益計算書の最新列）\n\n" + _table(
        ["期末日", "社数"], [[k, v] for k, v in sorted(ends.items(), reverse=True)]))
    stale_rows: list[list[str]] = []
    for r in df.itertuples():
        p = next_period_end_passed(r.op_period_end if isinstance(r.op_period_end, str) else None, asof)
        if p:
            stale_rows.append([r.code, r.name, r.op_period_end, p[0], p[1], r.status])
    parts.append("### yfinance の最新列より後の期末日が基準日までに来ている会社\n\n" + _table(
        ["証券コード", "社名", "yfinance の最新列", "その次の期末日", "期末日から基準日までの日数", "足切り"],
        stale_rows))
    parts.append("### 営業損益の項目差（同じ期末日で両方ある社）\n\n" + _table(
        ["比較", "社数"],
        [["Operating Income と Total Operating Income As Reported が両方ある", int(both.sum())],
         ["うち赤字・黒字の符号が食い違う", sign_diff]]))
    multi = df[df["other_processes"].fillna("") != ""]
    parts.append("### 工程が1つに決まらない会社（表には主工程で1行）\n\n" + _table(
        ["証券コード", "社名", "主工程（表の工程）", "根拠の製品", "ほかの工程"],
        [[r.code, r.name, r.process, r.basis, r.other_processes.replace(";", "・")]
         for r in multi.itertuples()]))
    with open(UNIVERSE_DIR / "semicon_frontend_boundary.csv", encoding="utf-8") as f:
        boundary = list(csv.DictReader(f))
    parts.append("### 列挙から外した会社（取得していない）\n\n" + _table(
        ["証券コード", "社名", "製品", "理由"],
        [[b["code"], b["name"], b["product"], b["reason"]] for b in boundary]))
    return "\n\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True)
    ap.add_argument("--universe", default=str(UNIVERSE_DIR / "semicon_frontend.csv"))
    args = ap.parse_args()
    df = build(args.asof, Path(args.universe))
    out = OUT_DIR
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"semicon_frontend_{args.asof}.csv", index=False, encoding="utf-8")
    md = render(df, args.asof)
    (out / f"semicon_frontend_{args.asof}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
