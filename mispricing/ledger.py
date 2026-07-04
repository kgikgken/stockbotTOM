"""STEP5: 構造化ログ(計画/結果枠/棄却台帳) — CSV出力とN営業日後リターン自動追記."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PLAN_COLS = ["取引ID", "日付", "コード", "銘柄名", "方向", "タイプ", "根拠(正規化/縮退)",
             "トリガー実測値", "計画IN", "損切り", "第1利確(+1R)", "2R参照水準",
             "計画R", "計画ブレンドR参考", "リスク%", "リスク%根拠", "確信度", "確信度減点履歴",
             "地合いスコア", "VI", "保有期間", "時間ストップN", "候補有効期限",
             "ステータス", "反証要点"]

RESULT_COLS = ["取引ID", "実IN", "実決済", "実現損益", "手数料スリッページ実額",
               "実現R(加重平均)", "第1利確到達YN", "残玉エグジット理由",
               "結果ラベル", "エグジット理由(最終)", "実保有日数", "寄りギャップ",
               "ゲート0_3事後的中タグ", "確信度x結果"]

REJECT_COLS = ["棄却日", "コード", "銘柄名", "ゲート", "理由", "棄却時終値",
               "N営業日後リターン%", "事後正誤タグ"]


def write_plan_log(outdir: str, today: str, picked: list, macro: dict, cfg) -> str:
    path = Path(outdir) / f"plan_log_{today}.csv"
    rows = []
    for i, c in enumerate(picked, 1):
        rows.append({
            "取引ID": f"{today}-{i:02d}", "日付": today, "コード": c.code, "銘柄名": c.name,
            "方向": c.direction, "タイプ": c.mtype, "根拠(正規化/縮退)": c.basis,
            "トリガー実測値": c.trigger_text,
            "計画IN": c.entry, "損切り": c.stop, "第1利確(+1R)": c.tp1, "2R参照水準": c.ref2r,
            "計画R": 2.0, "計画ブレンドR参考": "+0.5R(第1のみ)/+1.5R(2Rトレール)",
            "リスク%": c.risk_pct,
            "リスク%根拠": f"確信度{c.conf}×地合い係数{macro['lot_factor']}",
            "確信度": c.conf, "確信度減点履歴": " / ".join(c.conf_trail),
            "地合いスコア": macro["score"], "VI": f"{macro['vi']:.1f}" if macro["vi"] else "欠落",
            "保有期間": "短期スイング", "時間ストップN": c.hold_days,
            "候補有効期限": f"{c.expiry_days}営業日",
            "ステータス": "仮点灯(未確認・単一ソース)",
            "反証要点": "ゲート0/3はチャット側(パス2)で実施",
        })
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_COLS)
        w.writeheader()
        w.writerows(rows)
    return str(path)


def ensure_result_template(outdir: str) -> str:
    path = Path(outdir) / "result_log_template.csv"
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(RESULT_COLS)
    return str(path)


def append_reject_ledger(outdir: str, today: str, rejects: List[dict]) -> str:
    path = Path(outdir) / "reject_ledger.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(REJECT_COLS)
        for r in rejects:
            w.writerow([today, r["code"], r["name"], r["stage"], r["reason"],
                        r.get("close", ""), "", ""])
    return str(path)


def backfill_reject_returns(outdir: str, ohlcv: Dict[str, pd.DataFrame],
                            bdays: int) -> int:
    """棄却台帳の空欄「N営業日後リターン%」を、価格データが揃った行に自動追記."""
    path = Path(outdir) / "reject_ledger.csv"
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    except Exception:
        return 0
    if len(df) == 0:
        return 0
    filled = 0
    for i, row in df.iterrows():
        if row["N営業日後リターン%"] != "" or row["棄却時終値"] in ("", "nan"):
            continue
        tkr = f"{row['コード']}.T"
        hist = ohlcv.get(tkr)
        if hist is None:
            continue
        try:
            d0 = pd.Timestamp(row["棄却日"])
            idx = hist.index[hist.index > d0]
            if len(idx) < bdays:
                continue
            c_then = float(row["棄却時終値"])
            c_n = float(hist.loc[idx[bdays - 1], "Close"])
            ret = (c_n / c_then - 1.0) * 100.0
            if np.isfinite(ret):
                df.at[i, "N営業日後リターン%"] = f"{ret:+.1f}"
                filled += 1
        except Exception:
            continue
    if filled:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return filled
