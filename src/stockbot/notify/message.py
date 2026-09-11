"""配信本文の組み立て（docs/PATTERN.md §6）。

**入力は成立の配信記録（`delivered_*.csv`）とその日のスナップショット
（`pattern_summary_*.json`）だけ。** 株価も指標もここでは計算し直さない。配信した
内容と台帳が食い違わないようにするため、出す値はすべて記録の列をそのまま整形する。

**通常はこの本文を送らない。** 画像カード 2 枚だけを送り（§6・SCREENER.md §4.5）、
描画か送信に失敗した日だけここに落ちる。先頭で失敗した旨を断る。

順位は付けない。監視の並びは上値抵抗線に近い順（明日抜けるかもしれない順）で、
**優劣ではない**。業種は表示のみで並び順に使わない（§6.5）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..render.context import PATTERN_LABELS, PATTERN_ORDER, WATCH_MAX

MAX_TEXT = 4900  # Worker 側で切られる上限（src/worker.js）。ここで超えないようにする

# 通常は画像カード2枚だけを送る。この本文が流れるのは描画か送信に失敗した日だけなので、
# 受け取った側が「いつもと違う」と分かるように先頭で断る
FALLBACK_NOTE = "画像生成に失敗（テキストで配信）"

TARGET_NOTE = "測定目標は文献上の目安（統計的な裏付けは未確認）。"
DISCLAIMER = "AI候補提示で投資助言ではない。最終判断と結果責任はユーザーにある。"


def _num(value, digits: int = 1) -> str:
    """欠損を「—」にして桁区切りで整形する。"""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):,.{digits}f}"


def _signed(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):+.{digits}f}%"


def _oku(adv_jpy) -> str:
    if adv_jpy is None or (isinstance(adv_jpy, float) and not np.isfinite(adv_jpy)):
        return "—"
    return f"{float(adv_jpy) / 1e8:,.1f}億円"


def _earnings(row) -> str:
    """決算までの営業日数。**9 割方「未取得」になる前提**で、それでも出す（§6.2）。"""
    days = row.get("earnings_days")
    if bool(row.get("earnings_unknown")) or days is None or (
            isinstance(days, float) and not np.isfinite(days)):
        return "決算日未取得"
    return f"決算まで{int(days)}営業日"


def _label(pattern) -> str:
    return PATTERN_LABELS.get(str(pattern), str(pattern))


def _rows(df: Optional[pd.DataFrame]) -> list:
    if df is None or len(df) == 0:
        return []
    return [row for _i, row in df.iterrows()]


def _done_line(row) -> list[str]:
    return [
        f"{row.get('ticker', '')} {row.get('name', '') or ''}"
        f"［{_label(row.get('pattern'))}］".rstrip(),
        f"  終値 {_num(row.get('close_t'))} / 抜け幅 {_signed(row.get('breakout_pct'), 2)}",
        f"  撤退 {_num(row.get('pattern_low'))}（{_signed(row.get('down_pct'))}）"
        f" / 目標 {_num(row.get('target'))}（{_signed(row.get('up_pct'))}）"
        f" / 比率 {_num(row.get('rr'), 2)}",
        f"  {row.get('sector33', '') or '—'} / {_oku(row.get('adv_jpy'))}"
        f" / {_earnings(row)}",
    ]


def _watch_line(row) -> str:
    streak = ""
    try:
        n = int(row.get("watch_streak") or 1)
        streak = f" 監視{n}日目" if n > 1 else ""
    except (TypeError, ValueError):
        pass
    return (f"{row.get('ticker', '')}［{_label(row.get('pattern'))}］"
            f" 抜けまで {_signed(-float(row.get('breakout_pct') or 0.0), 2)}{streak}")


def build_message(delivered: Optional[pd.DataFrame], watch: Optional[pd.DataFrame],
                  summary: dict, fallback: bool = False) -> str:
    """テキスト本文（画像に失敗した日だけ流れる）。

    **成立 0 件の日も本文を作る**（§6.1）。その旨を明記して監視だけを出す。
    """
    done_rows = _rows(delivered)
    done_tickers = {str(r.get("ticker") or "") for r in done_rows}
    # 成立している銘柄は監視から外す（§6.4）。既に抜けた銘柄を「待つ」側に出さない
    watch_rows = [r for r in _rows(watch)
                  if str(r.get("ticker") or "") not in done_tickers][:WATCH_MAX]

    lines = [FALLBACK_NOTE] if fallback else []
    lines += [
        f"チャートパターン {summary.get('delivered_on', '')}"
        f"（判定 {summary.get('asof', '')} の引け）",
        f"成立 {len(done_rows)}件 / 監視 {int(summary.get('n_watch') or 0)}件"
        f" / 判定対象 {int(summary.get('n_evaluated') or 0):,}銘柄",
        "",
    ]

    if done_rows:
        lines.append("■ 候補（上値抵抗線を抜けた）")
        for pattern in PATTERN_ORDER:
            for r in [x for x in done_rows if str(x.get("pattern")) == pattern]:
                lines += _done_line(r)
        lines.append("")
    else:
        lines += ["■ 本日の成立はありません。", ""]

    if watch_rows:
        lines.append(f"■ 監視（形は揃い、まだ抜けていない／上位 {len(watch_rows)}件）")
        lines += [f"・{_watch_line(r)}" for r in watch_rows]
        lines.append("")

    lines += [TARGET_NOTE, "監視は記録に残しません。", DISCLAIMER]
    text = "\n".join(lines)
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT - 1] + "…"
