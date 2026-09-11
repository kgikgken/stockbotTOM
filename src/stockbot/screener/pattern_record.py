"""パターンの配信記録と日次スナップショット（docs/PATTERN.md §3）。

**記録するのは成立（ネックライン上抜け）だけ。監視（未抜け）は記録しない**（§3.1）。
成立は事象（抜けた日 1 日）、監視は状態（同じ形が何日も続く）で、単位が違う。
監視を台帳に書くと分母が「形の数」ではなく「形 × 滞留日数」になる。

監視の推移は**日次スナップショット**に残す（§3.2）。`pattern_summary_<配信日>_asof<判定日>.json`
に、その日の監視銘柄と `breakout_pct` を書く。**これは観測値であって台帳ではない** ——
上書きしてよいし、結果付けの対象にもしない。

ファイル名の規約（`stamped_name`）と入出力は `record.py` のものをそのまま使う。
判定日をファイル名に持つので、同じ配信日に引け前と引け後の 2 回走っても両方残る
（2026-09-03 の記録消失と同じ形を作らない）。

**古い `screen_summary_*.json` とは別の prefix にしてある。** 19 条件のスクリーナーの
配信記録は保全する決まりで（SCREENER_CLOSING.md）、同じ名前空間に書くと混ざる。
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from .record import as_calendar_date, parse_stamped_name, stamped_name

PATTERN_SUMMARY_PREFIX = "pattern_summary_"
PATTERN_SUMMARY_SUFFIX = ".json"

# 配信記録の列（docs/PATTERN.md §3.3。「具体的な列名は配信の実装時に決める」に対する回答）。
# **成立した行だけがここに入る。** 値はすべて T の引けまでで決まるものだけで、
# 結果（T+1 以降）は outcome 側が持つ
PATTERN_DELIVERED_COLS = [
    "delivered_on",       # 配信日（LINE に流した日）
    "asof",               # 判定日 T
    "ticker",
    "name",               # 銘柄名（取れなければ空）
    "sector33",           # 33業種。**表示のみ。並び順にも条件にも使わない**（§6.5）
    "pattern",            # double_bottom / ascending_triangle / ...
    "neckline",           # 反転系は水平なネックライン、保ち合い系は上辺の T での値
    "close_t",            # Close[T]
    "atr_t",              # ATR14[T]。**MFE / MAE を ATR 単位で出すために記録する**
                          # （§3.4）。判定には使わない
    "breakout_pct",       # (終値 / ネックライン − 1) × 100。成立なら正
    "pattern_low",        # 撤退の目安（パターンの最安値）
    "height",             # ネックライン − 最安値
    "target",             # 測定目標（ネックライン + 高さ）。**文献上の目安**（§2.3）
    "breakout_h",         # 抜け幅 ÷ 高さ（比率の b）
    "up_pct",             # 目標までの距離（%）
    "down_pct",           # 撤退までの距離（%・負）
    "rr",                 # 比率 = (1 − b) / (1 + b)
    "breakeven_win_rate",  # 1 / (1 + 比率)
    "span",               # 最初の極値から T までの本数
    # 極値。**日付と価格の両方を持つ**（チャートで形を確認するため・§6.2）
    "l1_date", "l1", "l2_date", "l2", "l3_date", "l3",
    "h1_date", "h1", "h2_date", "h2", "h3_date", "h3",
    # 保ち合い系の線（反転系では欠損）
    "upper_slope", "lower_slope", "pole_pct",
    "upper_scatter",      # 上辺の散らばり。山 3 点以上のときだけ（§2.2・D-13）
    # 表示用
    "adv_jpy",            # 20日平均売買代金
    "earnings_days",      # 決算発表までの営業日数（取れなければ NaN）
    "earnings_unknown",   # True ならカードに「決算日未取得」と出す。9 割方これになる
]

PATTERN_DATE_COLS = ["delivered_on", "asof",
                     "l1_date", "l2_date", "l3_date",
                     "h1_date", "h2_date", "h3_date"]

PATTERN_STR_COLS = ["ticker", "name", "sector33", "pattern"]

# 監視の連続日数を遡る上限。反転系の未抜けは数日〜数週間そのまま残るので、
# 押し目型（60）より長くは要らないが、同じ値にしておく
MAX_WATCH_LOOKBACK = 60


def pattern_summary_path(daily_dir: Path, delivered_on, asof) -> Path:
    """日次スナップショットのパス（§3.2）。"""
    return Path(daily_dir) / stamped_name(PATTERN_SUMMARY_PREFIX, delivered_on, asof,
                                          PATTERN_SUMMARY_SUFFIX)


def save_pattern_summary(summary: dict, daily_dir: Path, delivered_on, asof) -> Path:
    """日次スナップショットを書く。**既存を上書きする**（§3.2）。

    台帳ではなく観測値なので、同じ配信日・同じ判定日で 2 回走ったら新しい方が正しい。
    `save_delivered`（上書きしない）とは扱いが違う —— 判定日がファイル名に入っている
    ので、2026-09-03 のように「朝の 0 件と引け後の 5 件が同じ名前を取り合う」ことは
    起きない。
    """
    daily_dir = Path(daily_dir)
    daily_dir.mkdir(parents=True, exist_ok=True)
    path = pattern_summary_path(daily_dir, delivered_on, asof)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    return path


class PatternSummaryFile(NamedTuple):
    delivered_on: pd.Timestamp
    asof: Optional[pd.Timestamp]
    path: Path


def list_pattern_summaries(daily_dir: Path) -> list[PatternSummaryFile]:
    """スナップショットを (配信日, 判定日) の昇順で返す。"""
    daily_dir = Path(daily_dir)
    if not daily_dir.exists():
        return []
    out: list[PatternSummaryFile] = []
    for f in sorted(daily_dir.glob(f"{PATTERN_SUMMARY_PREFIX}*{PATTERN_SUMMARY_SUFFIX}")):
        parsed = parse_stamped_name(f.name, PATTERN_SUMMARY_PREFIX, PATTERN_SUMMARY_SUFFIX)
        if parsed is None:
            continue
        delivered_on, asof = parsed
        out.append(PatternSummaryFile(delivered_on, asof, f))
    out.sort(key=lambda x: (x.delivered_on,
                            x.asof if x.asof is not None else x.delivered_on))
    return out


def latest_pattern_summary(daily_dir: Path,
                           delivered_on=None) -> Optional[PatternSummaryFile]:
    """その配信日で最も新しい判定のスナップショット（省略時は全体で最新）。"""
    files = list_pattern_summaries(daily_dir)
    if delivered_on is not None:
        target = as_calendar_date(delivered_on)
        files = [f for f in files if f.delivered_on == target]
    return files[-1] if files else None


def _watch_keys(path: Path) -> set:
    """スナップショット 1 ファイルに載っている監視の (銘柄, パターン) の集合。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()       # 読めないファイルは「その日は出なかった」扱い（例外にしない）
    return {(str(r.get("ticker") or ""), str(r.get("pattern") or ""))
            for r in (data.get("watch") or [])}


def watch_streaks(daily_dir: Path, keys, delivered_on,
                  max_files: int = MAX_WATCH_LOOKBACK) -> dict:
    """監視の連続日数（docs/PATTERN.md §6.2）。今日を 1 日目として数える。

    keys は `(ticker, pattern)` の並び。**銘柄ではなく (銘柄, パターン) 単位**で数える
    —— 1909.T が C1 と C2 の両方に出る日があるので、銘柄だけで数えると「C1 として
    5 日目」と「C2 として 1 日目」が同じ数字になってしまう。カードは 1 行 1 パターン
    なので、行に対応する単位で数える。

    数え方は押し目型の `record.lookback_stats` と同じ（§3.2）。**配信日で数え、同じ
    配信日に判定が 2 つある日は和集合を取る** —— 1 日を 2 日と数えないため。
    スナップショットが欠けている日（ワークフローが失敗した日）は「その日は出なかった」
    扱いになるので、連続日数は実際より短くなりうる。遡って直さない。
    """
    keys = [(str(t), str(p)) for t, p in keys]
    streak = {k: 1 for k in keys}
    if not keys:
        return streak

    delivered_on = as_calendar_date(delivered_on)
    by_day: dict = {}
    for f in list_pattern_summaries(daily_dir):
        if f.delivered_on < delivered_on:
            by_day.setdefault(f.delivered_on, []).append(f.path)
    past = sorted(by_day.items(), key=lambda kv: kv[0], reverse=True)

    unbroken = set(keys)
    for _day, paths in past[:max_files]:
        present: set = set()
        for path in paths:
            present |= _watch_keys(path)
        for k in list(unbroken):
            if k in present:
                streak[k] += 1
            else:
                unbroken.discard(k)
        if not unbroken:
            break
    return streak


def build_delivered(rows: pd.DataFrame, delivered_on, asof) -> pd.DataFrame:
    """成立した行を配信記録の形に整える（§3.1）。

    rows は `cli.step_pattern` が作る表（`features.pattern.PATTERN_COLS` +
    ticker / 極値の日付 / 表示用の列）。**ここでは何も計算し直さない** —— 列を選んで
    並べるだけで、値はすべて検出時のものをそのまま持つ（§4.3 と同じ方針）。
    """
    out = pd.DataFrame(index=rows.index)
    out["delivered_on"] = as_calendar_date(delivered_on)
    out["asof"] = as_calendar_date(asof)
    for col in PATTERN_DELIVERED_COLS:
        if col in ("delivered_on", "asof"):
            continue
        if col in rows.columns:
            out[col] = rows[col].values
        elif col in PATTERN_DATE_COLS:
            out[col] = pd.NaT
        elif col in PATTERN_STR_COLS:
            out[col] = ""
        elif col == "earnings_unknown":
            out[col] = True
        else:
            out[col] = np.nan
    return out[PATTERN_DELIVERED_COLS].reset_index(drop=True)


def load_pattern_delivered(path: Path) -> pd.DataFrame:
    """パターンの配信記録を読む。日付列は Timestamp、文字列列は空文字を保つ。"""
    df = pd.read_csv(path, dtype={c: str for c in PATTERN_STR_COLS},
                     keep_default_na=False, na_values=[""])
    for col in PATTERN_DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in PATTERN_STR_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("")
    if "earnings_unknown" in df.columns:
        df["earnings_unknown"] = df["earnings_unknown"].map(
            lambda v: v if isinstance(v, bool) else str(v) in ("True", "true")
        ).astype(bool)
    return df


def is_pattern_record(df: pd.DataFrame) -> bool:
    """パターンの配信記録か（押し目型のものと区別する）。

    結果付け（`resolver.py`）は押し目型の列（`lp` / `h0_high`）を前提にしているので、
    パターンの記録を渡すと落ちる。**どちらの記録かを列で判定する** —— ファイル名は
    共通の規約なので、名前では区別できない。
    """
    return "pattern" in df.columns and "lp" not in df.columns
