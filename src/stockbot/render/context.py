"""表示内容の組み立て（docs/PATTERN.md §6）。

配信記録（成立）とその日の監視・要約から、テンプレートに渡す値だけを作る。
**ここでは株価も指標も計算し直さない。** 描画を差し替えても表示内容が変わらないよう、
整形はすべてこの層に閉じる（SCREENER.md §4.3 と同じ方針）。

Playwright も Jinja2 も import しないので、ブラウザ無しでテストできる。

構成は §6.1 のとおり **パターン別にセクションを分け**、各セクションの中を
**候補（成立）** と **監視（未抜け）** に分ける。業種は表示のみで、並び順に使わない
（§6.5）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# パターン名の表示。順番は docs/PATTERN.md §2 の並び（反転系 → 保ち合い系）。
# **優劣でも順位でもない。** 文書と同じ順に並べると読み手が引きやすいというだけ
PATTERN_LABELS = {
    "double_bottom": "ダブルボトム",
    "triple_bottom": "トリプルボトム",
    "inverse_hs": "逆三尊",
    "ascending_triangle": "上昇三角",
    "ascending_box": "上昇ボックス",
    "bull_flag": "上昇フラッグ",
    "bull_pennant": "上昇ペナント",
}
PATTERN_ORDER = list(PATTERN_LABELS)

# 監視に出す上限（§6.1）。1枚目のカードが縦に長くなりすぎない上限として 10 件
# （設計責任者の判断）。**絞り込みの基準ではなく、1 枚に収まる枚数**である
WATCH_MAX = 10

# 反転系は「ネックライン」、保ち合い系は「上値抵抗線」と呼ぶ（§6.2）
NECKLINE_LABELS = {"double_bottom": "ネックライン", "triple_bottom": "ネックライン",
                   "inverse_hs": "ネックライン"}
DEFAULT_NECKLINE_LABEL = "上値抵抗線"

TARGET_NOTE = "測定目標は文献上の目安で、統計的な裏付けは調べていません。"
NO_DONE_NOTE = "本日の成立はありません。"
# 注記は HTML にそのまま入る。**Markdown の強調記号を書かない**（文字として出る）
WATCH_NOTE = ("監視は形が揃って上抜けを待っている銘柄です。記録には残しません。"
              "並びは上値抵抗線に近い順で、優劣ではありません。")
SECTOR_NOTE = "業種は表示のみで、並び順にも判定にも使っていません。"
DISCLAIMER = "AI候補提示で投資助言ではない。最終判断と結果責任はユーザーにある。"


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


def _signed_pct(value, digits: int = 1) -> str:
    v = _f(value)
    return "—" if v is None else f"{v:+.{digits}f}%"


def _num(value, digits: int = 2) -> str:
    v = _f(value)
    return "—" if v is None else f"{v:,.{digits}f}"


def _extremes(row, keys) -> list:
    """極値を「日付 価格」の並びにする。欠損は飛ばす（§6.2）。

    **チャートで形を確認するための表示。** 押し目型と決定的に違う点なので、
    位置（バー番号）ではなく日付で出す。
    """
    out = []
    for key in keys:
        date, value = row.get(f"{key}_date"), _f(row.get(key))
        if value is None or date is None:
            continue
        # スナップショット経由だと文字列、配信記録経由だと Timestamp。どちらも受ける
        try:
            date = pd.Timestamp(date)
        except (TypeError, ValueError):
            continue
        if pd.isna(date):
            continue
        out.append({"date": f"{date:%m-%d}", "value": _yen(value)})
    return out


def _earnings(row) -> dict:
    """決算までの日数（§6.2）。**9 割方「未取得」になる前提**で、それでも載せる。

    載せないと、取れている 1 割の情報まで消える。赤字が並ぶのは実態の表示である。
    """
    days = _f(row.get("earnings_days"))
    if bool(row.get("earnings_unknown")) or days is None:
        return {"text": "決算日未取得", "unknown": True}
    return {"text": f"決算まで{int(days)}営業日", "unknown": False}


def _streak(row) -> Optional[str]:
    """監視の連続日数（§6.2）。初日は何も出さない。"""
    try:
        n = int(row.get("watch_streak") or 1)
    except (TypeError, ValueError):
        return None
    return f"監視{n}日目" if n > 1 else None


def build_card(row, watch: bool = False) -> dict:
    """1 行ぶんの表示内容。値はすべて記録・検出の列から来る（何も計算し直さない）。"""
    pattern = str(row.get("pattern") or "")
    close = _f(row.get("close_t"))
    scatter = _f(row.get("upper_scatter"))
    return {
        "pattern": pattern,
        "pattern_label": PATTERN_LABELS.get(pattern, pattern),
        "ticker": str(row.get("ticker") or ""),
        "name": str(row.get("name") or ""),
        "sector33": str(row.get("sector33") or ""),
        "watch": bool(watch),
        "close": _yen(close),
        "neckline": _yen(row.get("neckline")),
        "neckline_label": NECKLINE_LABELS.get(pattern, DEFAULT_NECKLINE_LABEL),
        "breakout": _signed_pct(row.get("breakout_pct"), 2),
        "stop": _yen(row.get("pattern_low")),
        "stop_gap": _signed_pct(row.get("down_pct")),
        "target": _yen(row.get("target")),
        "target_gap": _signed_pct(row.get("up_pct")),
        "rr": _num(row.get("rr")),
        "troughs": _extremes(row, ("l1", "l2", "l3")),
        "peaks": _extremes(row, ("h1", "h2", "h3")),
        "span": (f"{int(row['span'])}本" if _f(row.get("span")) is not None else "—"),
        # 山 3 点以上のときだけ。**V 字を水平線と取り違えないための表示**（§2.2・D-13）
        "scatter": (None if scatter is None else f"{scatter:.2f}%"),
        "scatter_wide": bool(scatter is not None and scatter > 0.75),
        "adv": (f"{_f(row.get('adv_jpy')) / 1e8:,.1f}億円"
                if _f(row.get("adv_jpy")) is not None else "—"),
        "earnings": _earnings(row),
        "streak": _streak(row) if watch else None,
    }


def _rows(df: Optional[pd.DataFrame]) -> list:
    if df is None or len(df) == 0:
        return []
    return [row for _i, row in df.iterrows()]


def build_sections(delivered: Optional[pd.DataFrame],
                   watch: Optional[pd.DataFrame]) -> list:
    """パターン別のセクション（§6.1）。中を候補（成立）と監視（未抜け）に分ける。

    **同一銘柄が成立と監視の両方に該当したら、監視から外す**（§6.4）。
    既に抜けている銘柄を「抜けるのを待つ」側に出すと読み手が混乱する。
    記録は成立のみなので、これは表示だけの話である。

    監視は**全体で上位 `WATCH_MAX` 件**に絞ってからセクションに割る —— パターンごとに
    10 件ずつにすると 1 枚目が収まらない。絞る基準は上値抵抗線に近い順で、
    「明日抜けるかもしれない順」であって優劣ではない。
    """
    done_rows = _rows(delivered)
    done_tickers = {str(r.get("ticker") or "") for r in done_rows}
    watch_rows = [r for r in _rows(watch)
                  if str(r.get("ticker") or "") not in done_tickers][:WATCH_MAX]

    by_pattern: dict = {}
    for r in done_rows:
        by_pattern.setdefault(str(r.get("pattern") or ""), {"done": [], "watch": []})
        by_pattern[str(r.get("pattern") or "")]["done"].append(build_card(r))
    for r in watch_rows:
        by_pattern.setdefault(str(r.get("pattern") or ""), {"done": [], "watch": []})
        by_pattern[str(r.get("pattern") or "")]["watch"].append(build_card(r, watch=True))

    order = [p for p in PATTERN_ORDER if p in by_pattern]
    order += [p for p in by_pattern if p not in PATTERN_ORDER]   # 未知名も落とさない
    return [{"pattern": p, "label": PATTERN_LABELS.get(p, p),
             "done": by_pattern[p]["done"], "watch": by_pattern[p]["watch"],
             "n_done": len(by_pattern[p]["done"]),
             "n_watch": len(by_pattern[p]["watch"])}
            for p in order]


def build_breakdown(summary: dict) -> list:
    """2枚目のパターン別内訳（§6.3）。検出時の全件数で、1枚目の表示件数ではない。"""
    counts = summary.get("counts") or {}
    order = PATTERN_ORDER + [p for p in counts if p not in PATTERN_ORDER]
    out = []
    for p in order:
        done = int((counts.get(p) or {}).get("done") or 0)
        watch = int((counts.get(p) or {}).get("watch") or 0)
        if done or watch:
            out.append({"label": PATTERN_LABELS.get(p, p), "done": done, "watch": watch})
    return out


def build_context(delivered: Optional[pd.DataFrame], watch: Optional[pd.DataFrame],
                  summary: dict) -> dict:
    """テンプレートに渡す全体（§6）。

    delivered は成立の配信記録（0 行でもよい）。watch はその日の監視（上値抵抗線に
    近い順に並んでいること）。summary は日次スナップショットの dict。
    """
    sections = build_sections(delivered, watch)
    breakdown = build_breakdown(summary)
    n_done = sum(s["n_done"] for s in sections)
    n_watch_shown = sum(s["n_watch"] for s in sections)
    return {
        "delivered_on": str(summary.get("delivered_on") or ""),
        "asof": str(summary.get("asof") or ""),
        "n_evaluated": int(summary.get("n_evaluated") or 0),
        "n_done": n_done,
        "n_watch": int(summary.get("n_watch") or 0),     # 検出された監視の全件数
        "n_watch_shown": n_watch_shown,                  # 1枚目に載せた件数
        "sections": sections,
        # **成立 0 件の日も配信する**（§6.1）。何も送らないと、動いているのか壊れて
        # いるのか分からない。0 件であることを明記したうえで監視だけ出す
        "no_done_note": None if n_done else NO_DONE_NOTE,
        "cards": [c for s in sections for c in s["done"]],   # 2枚目の銘柄一覧
        "breakdown": breakdown,
        # **表の合計は表の行から出す。** 見出しの件数（配信記録の行数）と別々に作ると、
        # 食い違ったときにカードが嘘をつく
        "breakdown_done": sum(b["done"] for b in breakdown),
        "breakdown_watch": sum(b["watch"] for b in breakdown),
        "target_note": TARGET_NOTE,
        "watch_note": WATCH_NOTE,
        "sector_note": SECTOR_NOTE,
        "disclaimer": DISCLAIMER,
    }
