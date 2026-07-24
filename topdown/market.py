"""STEP1: 地合い分析 — 米国指数・ドル円・VIX・日経をyfinanceで機械取得し、ルーブリック加減点で
地合いスコア(1〜5)と基本姿勢を判定する。ニュース(変動要因)は取得不可のため扱わない。
主要データが複数欠けた場合はスコアに「暫定」を明記して続行(チャット版のデータ欠落時ルールに準拠)。
日経VIは取得不安定のためN225実現ボラ20d(年率換算)で代理する。
"""

from __future__ import annotations

import numpy as np

from .data import fetch_market_indices


def compute_sentiment(cfg, dryrun: bool = False) -> dict:
    m = fetch_market_indices(dryrun=dryrun)
    missing = list(m.get("missing", []))

    score = 3
    reasons = []

    spx = m.get("SPX")
    if spx:
        if spx["chg1d_pct"] >= cfg.sentiment_spx_th:
            score += 1; reasons.append(f"S&P500 +{spx['chg1d_pct']:.1f}%(+1)")
        elif spx["chg1d_pct"] <= -cfg.sentiment_spx_th:
            score -= 1; reasons.append(f"S&P500 {spx['chg1d_pct']:.1f}%(-1)")
        else:
            reasons.append(f"S&P500 {spx['chg1d_pct']:+.1f}%(±0)")

    sox = m.get("SOX")
    sox_rebound = False
    sox_available = bool(sox)
    if sox:
        sox_rebound = sox["chg1d_pct"] >= cfg.sox_rebound_th
        if sox["chg1d_pct"] >= cfg.sentiment_sox_th:
            score += 1; reasons.append(f"SOX +{sox['chg1d_pct']:.1f}%(+1)")
        elif sox["chg1d_pct"] <= -cfg.sentiment_sox_th:
            score -= 1; reasons.append(f"SOX {sox['chg1d_pct']:.1f}%(-1)")
        else:
            # ★2026-07-24: ±0の分岐が無く、横ばいだとSOX行そのものが消えていた。
            # 「取得できなかった」のか「動かなかった」のか区別できないため必ず出す。
            reasons.append(f"SOX {sox['chg1d_pct']:+.1f}%(±0)")
    else:
        reasons.append("SOX 未取得")

    vix = m.get("VIX")
    if vix and vix["last"] >= cfg.sentiment_vix_high:
        score -= 1; reasons.append(f"VIX {vix['last']:.0f}(-1)")

    # N225の5日リターン(自前計算)
    n225_series = m.get("n225_close_series")
    vi_proxy = None
    if n225_series is not None and len(n225_series) >= 21:
        r5 = float((n225_series.iloc[-1] / n225_series.iloc[-6] - 1) * 100)
        if r5 >= cfg.sentiment_n225_5d_th:
            score += 1; reasons.append(f"N225 5日{r5:+.1f}%(+1)")
        elif r5 <= -cfg.sentiment_n225_5d_th:
            score -= 1; reasons.append(f"N225 5日{r5:+.1f}%(-1)")
        lr = np.log(n225_series / n225_series.shift(1)).dropna()
        vi_proxy = float(lr.iloc[-20:].std(ddof=0) * np.sqrt(252) * 100)
    else:
        missing.append("N225系列(VI代理算出不可)")

    score = max(1, min(5, score))
    provisional = len(missing) >= 2

    if score >= 4:
        stance = "買い優勢"
    elif score == 3:
        stance = "中立"
    else:
        stance = "様子見・守り"

    hivol_env = vi_proxy is not None and vi_proxy > cfg.vi_high_threshold
    # 高ボラ・半導体大型株ルール(改訂版): VI高でも前夜SOX明確反発なら高ボラタグ付きで採用可。
    # 反発が無い場合のみ半導体・値がさ大型を新規候補から除外する。
    semis_mode = "normal"; semis_reason = ""
    if hivol_env:
        if sox_rebound:
            semis_mode = "hivol_tag"
        else:
            semis_mode = "exclude"
            # SOXが取れていない場合、「反発しなかった」ではなく「判定できなかった」。
            # 保守的に除外するのは維持しつつ、理由を取り違えないよう明示する。
            semis_reason = ("SOX未取得のため判定不能 — 保守的に除外"
                            if not sox_available else "前夜SOX反発なしのため除外")

    return {
        "score": score, "provisional": provisional, "stance": stance,
        "reasons": reasons, "missing": missing,
        "indices": {k: v for k, v in m.items() if k in ("SPX", "DOW", "NASDAQ", "SOX", "VIX", "USDJPY", "N225")},
        "vi_proxy": vi_proxy, "hivol_env": hivol_env, "sox_rebound": sox_rebound,
        "sox_available": sox_available, "semis_reason": semis_reason,
        "semis_mode": semis_mode, "synthetic": m.get("synthetic", False),
    }


def closing_note(sentiment: dict) -> dict:
    """レポート末尾に置く「今日の地合い」念押しの中身を組む。

    候補の件数は地合いで絞らない方針(2026-07-24)にしたため、最後に地合いを
    もう一度示して、見送る/ロットを落とすという判断材料を残す役割を持たせている。
    """
    sc = int(sentiment.get("score", 3))
    stance = sentiment.get("stance", "")
    lines = []
    detail = " / ".join(sentiment.get("reasons", []))
    if sentiment.get("vi_proxy") is not None:
        detail += f" / VI代理{sentiment['vi_proxy']:.0f}"
    if detail:
        lines.append(detail)
    if sentiment.get("missing"):
        lines.append("欠落: " + ", ".join(sentiment["missing"]) + " — 判断材料が揃っていない")
    if sentiment.get("hivol_env"):
        lines.append("高ボラ環境(VI代理30超) — " + (
            "前夜SOX反発あり" if sentiment.get("sox_rebound")
            else sentiment.get("semis_reason", "値がさ大型は除外")))
    if sc >= 4:
        tail = "買い優勢寄りの日。"
    elif sc == 3:
        tail = "中立。方向感は乏しい。"
    else:
        tail = "リスクオフ寄りの日。候補が出ても、見送る・ロットを落とすのは自由。"
    lines.append(tail)
    return {"score": sc, "stance": stance, "lines": lines,
            "provisional": bool(sentiment.get("provisional"))}
