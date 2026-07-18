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
    if sox:
        sox_rebound = sox["chg1d_pct"] >= cfg.sox_rebound_th
        if sox["chg1d_pct"] >= cfg.sentiment_sox_th:
            score += 1; reasons.append(f"SOX +{sox['chg1d_pct']:.1f}%(+1)")
        elif sox["chg1d_pct"] <= -cfg.sentiment_sox_th:
            score -= 1; reasons.append(f"SOX {sox['chg1d_pct']:.1f}%(-1)")

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
    semis_mode = "normal"
    if hivol_env:
        semis_mode = "hivol_tag" if sox_rebound else "exclude"

    return {
        "score": score, "provisional": provisional, "stance": stance,
        "reasons": reasons, "missing": missing,
        "indices": {k: v for k, v in m.items() if k in ("SPX", "DOW", "NASDAQ", "SOX", "VIX", "USDJPY", "N225")},
        "vi_proxy": vi_proxy, "hivol_env": hivol_env, "sox_rebound": sox_rebound,
        "semis_mode": semis_mode, "synthetic": m.get("synthetic", False),
    }
