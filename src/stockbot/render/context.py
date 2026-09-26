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
# 利確の目安（1段目）の倍率（§6.2）。**表示だけ。** 判定・記録・並び順には使わない。
# 記録の reached_target は**測定目標**が基準で、こちらとは別物（§6.6）
FIRST_TAKE_ATR_MULT = 1.0
FIRST_TAKE_NOTE = ("利確の目安（1段目）は終値+ATR×1で出した運用上の目安です。"
                   "文献値でも検証済みでもありません。記録の判定には使っていません"
                   "（記録は測定目標が基準）。")
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


def _signed_yen(value) -> str:
    """円建ての差額。**符号を必ず付ける**（株数計算で向きを間違えないため。§6.2）。"""
    v = _f(value)
    return "—" if v is None else f"{v:+,.1f}"


def _atr_text(atr: Optional[float], close: Optional[float]) -> str:
    """ATR を円と終値比 % で出す（§6.2）。**表示だけ。判定には使わない。**"""
    if atr is None:
        return "—"
    if close is None or close <= 0:
        return f"{atr:,.1f}円"
    return f"{atr:,.1f}円 / {atr / close * 100:.1f}%"


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


def build_card(row, watch: bool = False, multi: bool = False) -> dict:
    """1 行ぶんの表示内容。値はすべて記録・検出の列から来る（何も計算し直さない）。

    `multi` は同じ銘柄が複数パターンで出ていることの印（§6.4）。**表示だけ**で、
    記録の数え方（行単位）も検出側も変えない。
    """
    pattern = str(row.get("pattern") or "")
    close = _f(row.get("close_t"))
    scatter = _f(row.get("upper_scatter"))
    # **既存の記録から計算するだけ**（§6.2）。新しいデータ源は足さない
    atr = _f(row.get("atr_t"))
    stop = _f(row.get("pattern_low"))
    first_take = (None if (atr is None or close is None)
                  else close + FIRST_TAKE_ATR_MULT * atr)
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
        # **円建ての差額**（1 株あたりの想定損失）。株数計算に使う（§6.2）
        "stop_yen": ("—" if (stop is None or close is None)
                     else _signed_yen(stop - close)),
        # ATR は円と終値比 % の両方。**判定にも並び順にも使わない**
        "atr": _atr_text(atr, close),
        # 利確の目安（1段目）。**測定目標とは別物**。記録の reached_target は測定目標が基準
        "first_take": _yen(first_take),
        "first_take_yen": ("—" if atr is None
                           else _signed_yen(FIRST_TAKE_ATR_MULT * atr)),
        "first_take_mult": f"ATR×{FIRST_TAKE_ATR_MULT:.0f}",
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
        "multi": bool(multi),
    }


def _rows(df: Optional[pd.DataFrame]) -> list:
    if df is None or len(df) == 0:
        return []
    return [row for _i, row in df.iterrows()]


def multi_pattern_tickers(delivered: Optional[pd.DataFrame]) -> set:
    """同じ日に複数パターンで成立した銘柄コード（§6.4）。

    **落とさない。** 判定式には包含関係があるので重複は通常起きる（§2.2）。
    記録は行単位（`ticker` + `asof` + `pattern`）のままで、ここは表示の印だけ。
    """
    counts: dict = {}
    for r in _rows(delivered):
        t = str(r.get("ticker") or "")
        counts[t] = counts.get(t, 0) + 1
    return {t for t, n in counts.items() if n > 1}


def build_compact_rows(delivered: Optional[pd.DataFrame]) -> list:
    """1 枚目の成立一覧（**全件**）。値は記録の列をそのまま整形するだけ。

    **配信記録の並びをそのまま使う**（§6.6 の売買代金の降順を引き継ぐ）。
    ソートを書き足さない —— 同じ銘柄の複数パターンは売買代金が同値なので、
    記録の時点で既に隣り合っている。並べ替えずに隣接表示になる。
    """
    multi = multi_pattern_tickers(delivered)
    out = []
    for r in _rows(delivered):
        ticker = str(r.get("ticker") or "")
        pattern = str(r.get("pattern") or "")
        out.append({
            "ticker": ticker,
            "name": str(r.get("name") or ""),
            "pattern_label": PATTERN_LABELS.get(pattern, pattern),
            "sector33": str(r.get("sector33") or ""),
            "close": _yen(r.get("close_t")),
            "stop": _yen(r.get("pattern_low")),
            "target": _yen(r.get("target")),
            "rr": _num(r.get("rr")),
            "multi": ticker in multi,
        })
    return out


def build_sector_breakdown(delivered: Optional[pd.DataFrame]) -> list:
    """業種別の件数（件数の降順）。**表示のみ**（§6.5）。

    **並び順にも判定にも使わない。** 同数のときは配信記録に出てきた順にする
    （名前順にすると、業種に順位があるように見えるため）。
    """
    counts: dict = {}
    order: list = []
    for r in _rows(delivered):
        label = str(r.get("sector33") or "") or "業種不明"
        if label not in counts:
            counts[label] = 0
            order.append(label)
        counts[label] += 1
    seen = {label: i for i, label in enumerate(order)}
    return [{"label": label, "n": counts[label]}
            for label in sorted(order, key=lambda s: (-counts[s], seen[s]))]


def build_headline(n_done: int, n_tickers: int, n_multi: int,
                   n_watch: int, n_watch_shown: int) -> str:
    """見出しの件数（§6.1）。**何の件数かを文として読めるようにする。**

    成立は**行数**が記録の数え方（`ticker` + `asof` + `pattern`）で、銘柄数は
    その補足である。監視の掲載数は「上位 N 件」とだけ書くと成立の話に読めるので、
    何を基準に何件載せたかまで書く。
    """
    done = f"成立 {n_done}件"
    if n_multi:
        done += f"（{n_tickers}銘柄・複数パターン{n_multi}銘柄）"
    watch = f"監視 {n_watch}件"
    if n_watch_shown:
        watch += f"（うち上値抵抗線に近い{n_watch_shown}件を掲載）"
    return f"{done} ／ {watch}"


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

    # **複数パターンの印**（§6.4）。監視側は**載せる 10 件の中**で数える ——
    # 切られた行まで数えると、1 件しか出ていないカードに「複数」と出てしまう
    done_multi = multi_pattern_tickers(delivered)
    watch_counts: dict = {}
    for r in watch_rows:
        t = str(r.get("ticker") or "")
        watch_counts[t] = watch_counts.get(t, 0) + 1
    watch_multi = {t for t, n in watch_counts.items() if n > 1}

    by_pattern: dict = {}
    for r in done_rows:
        by_pattern.setdefault(str(r.get("pattern") or ""), {"done": [], "watch": []})
        by_pattern[str(r.get("pattern") or "")]["done"].append(
            build_card(r, multi=str(r.get("ticker") or "") in done_multi))
    for r in watch_rows:
        by_pattern.setdefault(str(r.get("pattern") or ""), {"done": [], "watch": []})
        by_pattern[str(r.get("pattern") or "")]["watch"].append(
            build_card(r, watch=True,
                       multi=str(r.get("ticker") or "") in watch_multi))

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
    multi = multi_pattern_tickers(delivered)
    n_tickers = len({str(r.get("ticker") or "") for r in _rows(delivered)})
    n_watch_all = int(summary.get("n_watch") or 0)
    return {
        "delivered_on": str(summary.get("delivered_on") or ""),
        "asof": str(summary.get("asof") or ""),
        "n_evaluated": int(summary.get("n_evaluated") or 0),
        "n_done": n_done,
        "n_watch": n_watch_all,                          # 検出された監視の全件数
        "n_watch_shown": n_watch_shown,                  # 3枚目に載せた件数
        # 成立の**銘柄数**と**複数パターンの銘柄数**。記録の数え方は行単位のままで、
        # これは見出しの補足である（§6.4）
        "n_done_tickers": n_tickers,
        "n_multi": len(multi),
        "headline": build_headline(n_done, n_tickers, len(multi),
                                   n_watch_all, n_watch_shown),
        "sections": sections,
        # **成立 0 件の日も配信する**（§6.1）。何も送らないと、動いているのか壊れて
        # いるのか分からない。0 件であることを明記したうえで監視だけ出す
        "no_done_note": None if n_done else NO_DONE_NOTE,
        # 1枚目の成立一覧（全件）。配信記録の並びのままで、パターン別には束ねない
        # —— 束ねると同じ銘柄の複数パターンが離れる
        "compact": build_compact_rows(delivered),
        # 1枚目の業種別件数（§6.5 のとおり**表示のみ**）
        "sectors": build_sector_breakdown(delivered),
        "breakdown": breakdown,
        # **表の合計は表の行から出す。** 見出しの件数（配信記録の行数）と別々に作ると、
        # 食い違ったときにカードが嘘をつく
        "breakdown_done": sum(b["done"] for b in breakdown),
        "breakdown_watch": sum(b["watch"] for b in breakdown),
        "target_note": TARGET_NOTE,
        "first_take_note": FIRST_TAKE_NOTE,
        "watch_note": WATCH_NOTE,
        "sector_note": SECTOR_NOTE,
        "disclaimer": DISCLAIMER,
    }
