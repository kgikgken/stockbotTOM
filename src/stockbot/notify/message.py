"""配信本文の組み立て（docs/SCREENER.md §4.2）。

**入力は配信記録（`delivered_*.csv`）とその日の要約（`screen_summary_*.json`）だけ。**
株価も指標もここでは計算し直さない。配信した内容と台帳が食い違わないようにするため、
カードに出す値はすべて配信記録の列をそのまま整形したものにする（§4.3 のテスト）。

順位は付けない。並びは配信記録の順（売買代金の降順）をそのまま使い、本文にもその旨を
書く（§2.8）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..screener.conditions import CONDITION_LABELS
from ..screener.record import RECORD_MISMATCH_NOTE

MAX_TEXT = 4900  # Worker 側で切られる上限（src/worker.js）。ここで超えないようにする
TOP_FAILS = 3    # 候補0件の日に添える「落ちた条件」の件数

# 通常は画像カード2枚だけを送る（§4.5）。この本文が流れるのは描画か送信に失敗した日
# だけなので、受け取った側が「いつもと違う」と分かるように先頭で断る（§4.4）
FALLBACK_NOTE = "画像生成に失敗（テキストで配信）"

# 記録では SMA5 のような機械的な名前だが、配信では日本語で出す
MA_LABELS = {"SMA5": "5日線", "SMA25": "25日線", "SMA75": "75日線", "SMA200": "200日線"}


def _num(value, digits: int = 1) -> str:
    """欠損を「—」にして桁区切りで整形する。"""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):,.{digits}f}"


def _oku(adv_jpy) -> str:
    """売買代金を億円で。"""
    if adv_jpy is None or (isinstance(adv_jpy, float) and not np.isfinite(adv_jpy)):
        return "—"
    return f"{float(adv_jpy) / 1e8:,.1f}億円"


def _earnings(row) -> str:
    """決算までの営業日数。取れていない銘柄はその旨を明記する（§2.6）。"""
    if bool(row.get("a4_earnings_unknown")):
        return "決算日未取得"
    days = row.get("earnings_days")
    if days is None or (isinstance(days, float) and not np.isfinite(days)):
        return "決算日未取得"
    return f"決算まで{int(days)}営業日"


def _header(summary: dict, n_candidates: int, fallback: bool = False) -> list[str]:
    level = summary.get("regime_level") or "不明"
    score = summary.get("regime_score")
    gauge = f"{level}（{score}/6）" if score is not None else level
    lines = [FALLBACK_NOTE] if fallback else []
    lines += [
        f"順張り押し目 {summary.get('delivered_on', '')}"
        f"（判定 {summary.get('asof', '')} の引け）",
        f"地合い {gauge} ／ 候補 {n_candidates}件",
    ]
    # 台帳に書けなかった日（想定外の衝突）。数字がどれを指すのか読み手に分かるようにする
    if summary.get("delivered_written") is False:
        lines.append(f"※ {RECORD_MISMATCH_NOTE}")
    # 候補0件の日は E1 の注記を出さない。母集団も0なので「未適用」は正しいが、
    # 適用する対象がそもそも無い日にこれを出すと E1 が壊れているように読める
    if n_candidates and summary.get("e1_skipped"):
        lines.append(f"※ E1（相対力の上位10%除外）は母集団 {summary.get('n_pool', 0)}件"
                     "のため本日は未適用")
    return lines


def _streak(row) -> str:
    """連続点灯日数。初日は何も出さない（§3.2 の streak）。"""
    try:
        n = int(row.get("streak") or 1)
    except (TypeError, ValueError):
        return ""
    return f"（連続{n}日目）" if n > 1 else ""


def _sector(row, n_sectors: Optional[int]) -> str:
    """業種と 5 日順位（§2.9）。順位が取れなければ業種名だけ。"""
    name = str(row.get("sector33") or "")
    r = row.get("sector_rank_5d")
    try:
        rank = int(r)
    except (TypeError, ValueError):
        return name or "—"
    return f"{name} {rank}/{n_sectors}位" if n_sectors else name or "—"


def _card(i: int, row, n_sectors: Optional[int] = None) -> list[str]:
    ma = MA_LABELS.get(str(row.get("landing_ma") or ""), "—")
    return [
        f"{i}. {row['ticker']} {row.get('name') or ''}{_streak(row)}".rstrip(),
        f"   {_sector(row, n_sectors)} ／ {row.get('state') or '—'}",
        f"   終値 {_num(row['close_t'])}円 ／ {_oku(row.get('adv_jpy'))}",
        f"   止まった線 {ma}（{_num(row.get('landing_dist_atr'), 2)} ATR）",
        f"   撤退ライン（押し安値） {_num(row['lp'])}円",
        f"   目標の目安（直近高値） {_num(row['h0_high'])}円",
        f"   深さ {_num(float(row['depth_pct']) * 100, 2)}% ／ 押し目"
        f"{int(row['pullback_days'])}日 ／ {_earnings(row)}",
    ]


def _zero_day(summary: dict) -> list[str]:
    """候補0件の日。落ちた条件の上位を添える（§4.2）。"""
    lines = ["本日は19条件を全て満たす銘柄がありませんでした。"]
    fails = summary.get("fail_counts") or {}
    if fails:
        top = sorted(fails.items(), key=lambda kv: -kv[1])[:TOP_FAILS]
        # 1銘柄が複数の条件で落ちるので、合計は評価銘柄数を超える。「延べ」と明記しないと
        # 数字が矛盾しているように読める
        lines.append(f"落ちた条件（評価 {summary.get('n_evaluated', 0):,}銘柄・"
                     "延べ件数・多い順）:")
        for cid, n in top:
            lines.append(f"   {cid} {CONDITION_LABELS.get(cid, '')} … {n:,}件")
    return lines


DISCLAIMER = "この配信は監視候補の一覧です。売買の判断はご自身で行ってください。"
ORDER_NOTE = ("並びは業種の5日リターン順位（昇順）→20日平均売買代金（降順）です。"
              "優劣ではありません。")


def build_message(delivered: Optional[pd.DataFrame], summary: dict,
                  fallback: bool = False) -> str:
    """配信本文を組み立てる（§4.2）。

    delivered は screener.record.load_delivered() の出力（0行でもよい）。
    summary は screener.screen.build_summary() が書いた JSON を読んだ dict。

    fallback=True で先頭に「画像生成に失敗（テキストで配信）」を入れる。**通常の配信は
    画像カード2枚だけで、この本文は使わない**（§4.5）。テキストが届いた時点で異常なので、
    受け取った側がすぐ分かるようにする。

    返す文字列は MAX_TEXT 以内。超える場合は末尾のカードから落とし、何件省いたかを
    最終行に書く（黙って切らない）。見出しと免責は必ず残す。
    """
    n = 0 if delivered is None else len(delivered)
    head = _header(summary, n, fallback=fallback)
    foot = ["", DISCLAIMER]

    if n == 0:
        return "\n".join(head + [""] + _zero_day(summary) + foot)

    n_sectors = len(summary.get("sector_ranking") or []) or None
    cards = [_card(i, row, n_sectors)
             for i, (_idx, row) in enumerate(delivered.iterrows(), start=1)]

    def render(keep: int) -> str:
        lines = head + [ORDER_NOTE]
        for card in cards[:keep]:
            lines += [""] + card
        if keep < n:
            lines += ["", f"（残り {n - keep}件は文字数の都合で省略しました）"]
        return "\n".join(lines + foot)

    # 入る枚数を前から詰めていく（再帰にすると枚数に対して指数的に膨らむ）
    keep = n
    while keep > 0 and len(render(keep)) > MAX_TEXT:
        keep -= 1
    return render(keep) if keep > 0 else render(0)[:MAX_TEXT]
