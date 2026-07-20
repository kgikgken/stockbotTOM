"""候補の持ち越しと到達判定 — 3営業日ルールを実際に機能させるための状態管理.

botは毎朝ゼロから候補を作り直すため、昨日「ゾーンで待つ」と決めた候補を翌日は覚えていない。
このモジュールが pending_topdown.csv に候補を保存し、翌日以降に次を行う:

  1. ゾーン到達の自動判定 — その日の安値がゾーン上端以下まで下がったか(指値が約定しうる水準か)
  2. 失効判定 — 3営業日経過、または終値がゾーン下端を割った時点(構造の否定)
  3. 反実仮想の記録 — 提示日の終値(=旧方式の即時エントリー価格)と、その後の値動き

3の記録により「ゾーン指値方式 vs 即時エントリー方式」のどちらが良かったかを、
手入力ゼロで後から比較できる。調査(2026-07-19)では両方式の優劣は決着していないため、
片方を捨てずに実データで判定する設計とした。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PENDING_PATH = "pending_topdown.csv"

COLS = [
    "listed_date",      # 候補として提示した日
    "ticker", "code", "name", "sector", "trigger", "tag",
    "zone_hi", "zone_lo", "stop", "time_stop", "expire_date",
    "close_at_listing",  # 提示日の終値(= 即時エントリー方式の約定価格・反実仮想用)
    "status",            # pending / reached / expired / broken
    "reached_date", "reached_low",   # ゾーン到達時の日付とその日の安値
    "days_waited",
    "confidence", "gap_date", "unit_cost",
]


def load_pending(path: str = PENDING_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(p, dtype=str).fillna("")
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    return df


def save_pending(df: pd.DataFrame, path: str = PENDING_PATH) -> str:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _f(v) -> float | None:
    try:
        s = str(v).strip()
        return float(s) if s and s.lower() != "nan" else None
    except Exception:
        return None


def _bdays(d0: str, d1: str) -> int | None:
    try:
        return int(np.busday_count(pd.Timestamp(d0).date(), pd.Timestamp(d1).date()))
    except Exception:
        return None


def update_pending(pending: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
                   today: str, cfg) -> tuple[pd.DataFrame, List[dict]]:
    """保留中の候補を今日のデータで更新する。

    戻り値: (更新後のDataFrame, 今日の変化のリスト)
    変化 = {"code","name","event": reached/expired/broken, "note": ...}
    """
    if pending is None or len(pending) == 0:
        return pd.DataFrame(columns=COLS), []

    events: List[dict] = []
    rows = []
    for _, p in pending.iterrows():
        status = str(p.get("status", "")).strip() or "pending"
        if status != "pending":
            rows.append(p.to_dict())
            continue

        ticker = str(p.get("ticker", "")).strip()
        code = str(p.get("code", "")).strip()
        name = str(p.get("name", "")).strip()
        listed = str(p.get("listed_date", "")).strip()
        zone_hi, zone_lo = _f(p.get("zone_hi")), _f(p.get("zone_lo"))
        df = ohlcv.get(ticker)

        waited = _bdays(listed, today)
        rec = p.to_dict()
        rec["days_waited"] = waited if waited is not None else ""

        if df is None or zone_hi is None or zone_lo is None:
            rows.append(rec)
            continue
        try:
            low_today = float(df["Low"].dropna().iloc[-1])
            close_today = float(df["Close"].dropna().iloc[-1])
        except Exception:
            rows.append(rec)
            continue

        # ① 構造の否定(終値がゾーン下端割れ) → 即失効。到達判定より優先する
        if close_today < zone_lo:
            rec["status"] = "broken"
            events.append({"code": code, "name": name, "event": "broken",
                           "note": f"終値{close_today:,.0f}円がゾーン下端{zone_lo:,.0f}円割れ → 即失効"})
            rows.append(rec); continue

        # ② ゾーン到達(その日の安値がゾーン上端以下 = 指値が約定しうる水準まで下げた)
        if low_today <= zone_hi:
            rec["status"] = "reached"
            rec["reached_date"] = today
            rec["reached_low"] = f"{low_today:.1f}"
            fill = min(max(low_today, zone_lo), zone_hi)  # ゾーン内での想定約定価格
            events.append({"code": code, "name": name, "event": "reached",
                           "note": f"ゾーン到達(安値{low_today:,.0f}円)— 想定約定{fill:,.0f}円付近"})
            rows.append(rec); continue

        # ③ 期限切れ
        if waited is not None and waited >= cfg.zone_expire_days:
            rec["status"] = "expired"
            events.append({"code": code, "name": name, "event": "expired",
                           "note": f"{waited}営業日ゾーン未到達 → 失効"})
            rows.append(rec); continue

        rows.append(rec)

    return pd.DataFrame(rows, columns=COLS), events


def add_new_candidates(pending: pd.DataFrame, picked: list, today: str,
                       resolved_today: List[dict] | None = None) -> pd.DataFrame:
    """本日の本命候補を保留リストに追加する。

    次の銘柄は追加しない:
      - すでに保留中(pending)の銘柄 — 二重登録になるため
      - 本日決着した銘柄(到達・下端割れ・失効) — 同じレポートに二重で出て紛らわしいため、
        1営業日のクールダウンを置く。翌日以降に再点灯すれば改めて候補になる。
    """
    existing = set()
    if pending is not None and len(pending):
        for _, p in pending.iterrows():
            if str(p.get("status", "")).strip() == "pending":
                existing.add(str(p.get("ticker", "")).strip())
    for e in (resolved_today or []):
        code = str(e.get("code", "")).strip()
        if code:
            existing.add(code + ".T")

    new_rows = []
    for c in picked:
        if c.ticker in existing:
            continue
        new_rows.append({
            "listed_date": today, "ticker": c.ticker, "code": c.code, "name": c.name,
            "sector": c.sector, "trigger": c.trigger, "tag": c.tag,
            "zone_hi": f"{c.zone_hi:.1f}", "zone_lo": f"{c.zone_lo:.1f}",
            "stop": f"{c.stop:.1f}", "time_stop": c.time_stop, "expire_date": c.expire_date,
            "close_at_listing": f"{c.feat['close']:.1f}",
            "status": "pending", "reached_date": "", "reached_low": "", "days_waited": 0,
            "confidence": c.confidence, "gap_date": c.gap_date or "",
            "unit_cost": f"{c.unit_cost:.0f}",
        })
    if not new_rows:
        return pending if pending is not None else pd.DataFrame(columns=COLS)
    add = pd.DataFrame(new_rows, columns=COLS)
    if pending is None or len(pending) == 0:
        return add
    return pd.concat([pending, add], ignore_index=True)


def prune(pending: pd.DataFrame, keep_days: int = 90, today: str | None = None) -> pd.DataFrame:
    """古い決着済みレコードを落とす(検証用に一定期間は残す)。"""
    if pending is None or len(pending) == 0:
        return pending
    if today is None:
        return pending
    keep = []
    for _, p in pending.iterrows():
        if str(p.get("status", "")).strip() == "pending":
            keep.append(p.to_dict()); continue
        d = _bdays(str(p.get("listed_date", "")), today)
        if d is None or d <= keep_days:
            keep.append(p.to_dict())
    return pd.DataFrame(keep, columns=COLS)


def summarize(pending: pd.DataFrame) -> dict:
    """保留・到達・失効の件数サマリー(レポート表示用)。"""
    if pending is None or len(pending) == 0:
        return {"pending": 0, "reached": 0, "expired": 0, "broken": 0}
    s = pending["status"].astype(str).str.strip()
    return {"pending": int((s == "pending").sum()), "reached": int((s == "reached").sum()),
            "expired": int((s == "expired").sum()), "broken": int((s == "broken").sum())}
