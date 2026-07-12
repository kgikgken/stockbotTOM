"""STEP3: 3状態の日次分類 + エントリー水準の算出.

状態A(すでに流入): trend_align かつ 20日線基準の非対称帯での押し目(精緻化基準) → 新規ロング候補
状態B(初動): VCP収縮 かつ 出来高を伴うドンチアン・ブレイク → 新規ロング候補
状態C(流出): 50日線割れ → 新規対象外(保有ポジションの手仕舞い判定にのみ使用)

ロングオンリー(空売りはしない)。モメンタムクラッシュの非対称リスクを踏まえ、
ショート側は仕様に含めない(研究レポートのロングオンリー推奨に準拠)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .indicators import chandelier_exit_long  # noqa: F401 (re-export for report use)


@dataclass
class Candidate:
    ticker: str
    code: str
    name: str
    sector: str
    market: str
    state: str            # "A" or "B"
    score: float
    feat: dict
    entry: float = 0.0
    stop: float = 0.0
    chandelier: float = 0.0
    risk_w: float = 0.0
    risk_pct: float = 0.0
    shares: int = 0
    flags: List[str] = field(default_factory=list)
    checks: List[str] = field(default_factory=list)


def _round_tick(p: float) -> float:
    if p < 3000: t = 1
    elif p < 5000: t = 5
    elif p < 30000: t = 10
    elif p < 50000: t = 50
    else: t = 100
    return round(p / t) * t


def state_a_gate_diagnostics(feat: dict, cfg) -> list[str]:
    """状態Aの各ゲートのうち、この銘柄がどこで落ちたかを診断用に返す(trend_align=True前提)。
    判定ロジック自体(classify_state)は変更しない。棄却台帳の精査用の補助関数。"""
    failed = []
    if not feat["trend_align"]:
        failed.append("trend_align")
        return failed  # これが落ちていれば他は無意味
    ratio20 = feat["close"] / feat["sma20"] - 1
    if not ((-cfg.pullback_lower_pct / 100) <= ratio20 <= (cfg.pullback_upper_pct / 100)):
        failed.append("in_zone")
    if not (feat["pullback_depth_atr"] <= cfg.pullback_depth_atr_mult):
        failed.append("depth_ok")
    if not (feat["days_since_swing_high"] <= cfg.pullback_max_duration_days):
        failed.append("duration_ok")
    if not feat.get("bounce_confirmed"):
        failed.append("bounce_confirmed")
    return failed


def classify_state(feat: dict, cfg) -> str | None:
    """A / B / C / None(いずれでもない)を返す。優先順位: C(流出) > B(初動) > A(継続)。

    状態Aは以下5ゲートのAND(2026-07-12凍結版・懐疑的レビューを受けた整理後):
    ①トレンド整列(終値>50日>150日>200日) ②非対称な押し目ゾーン(20日線の上+2.5%〜下-5%、Alajbeg et al. 2017)
    ③深さ上限(スイング高値からATR×3以内) ④継続期間上限(スイング高値から35営業日以内・仮置き)
    ⑤反発確認(CLV≥0.5相当+直近2日プラス転換)

    整理の記録(2026-07-12):
    - 旧「50日線-10%フロア」は削除。①が終値>50日線を要求するため数学的に常に真(死んだ条件)だった。
    - 旧「52週高値60%以上」ゲートは削除。スコア側(w_high52w=0.25)と二重取りな上、
      60%という閾値は候補数を見て緩めた経緯(75→60)があり実証的根拠を持たないため。
      52週高値近接度はランキング要素としてのみ使う(George & Hwang 2004の使い方に整合)。
    """
    if feat["breakdown"]:
        return "C"
    if feat["vcp_now"] and feat["close"] > feat["donchian_prev"] \
            and feat["vol_ratio_today"] >= cfg.vcp_breakout_vol_mult:
        return "B"

    # ②非対称な押し目ゾーン(20日線基準。上+2.5%〜下-5%)
    ratio20 = feat["close"] / feat["sma20"] - 1
    in_zone = (-cfg.pullback_lower_pct / 100) <= ratio20 <= (cfg.pullback_upper_pct / 100)
    # ③深さ上限(スイング高値からATR×3以内)
    depth_ok = feat["pullback_depth_atr"] <= cfg.pullback_depth_atr_mult
    # ④継続期間上限(スイング高値からの経過営業日)
    duration_ok = feat["days_since_swing_high"] <= cfg.pullback_max_duration_days

    if (feat["trend_align"] and in_zone and depth_ok and duration_ok
            and feat.get("bounce_confirmed")):
        return "A"
    return None


def build_candidate(row: dict, feat: dict, state: str, score: float, cfg) -> Candidate | None:
    """状態A/Bのみ新規ロング候補化。エントリー・初期ストップ・シャンデリア水準を算出。
    score はプール構築時にz-score母集団全体で算出済みの総合スコア(compute_pool_scores)を渡す。"""
    if state not in ("A", "B"):
        return None
    tkr = row["ticker"]
    code = tkr.replace(".T", "")
    entry = feat["close"]
    stop_raw = entry - cfg.initial_stop_atr_mult * feat["atr"]
    if stop_raw <= 0:
        return None
    risk_w = entry - stop_raw
    if risk_w <= 0 or risk_w / entry * 100 > cfg.max_risk_width_pct:
        return None

    c = Candidate(ticker=tkr, code=code, name=row.get("name", ""), sector=row.get("sector", ""),
                 market=row.get("market", "Prime"), state=state, score=score, feat=feat)
    c.entry = _round_tick(entry)
    c.stop = _round_tick(stop_raw)
    c.chandelier = _round_tick(feat["chandelier"]) if feat["chandelier"] == feat["chandelier"] else c.stop
    c.risk_w = risk_w

    if state == "A":
        c.flags.append(f"押し目買い(凍結版基準): 20日線の上+{cfg.pullback_upper_pct:.1f}%〜下-{cfg.pullback_lower_pct:.1f}%・"
                      f"深さATR×{feat['pullback_depth_atr']:.1f}(上限{cfg.pullback_depth_atr_mult:.1f})・"
                      f"経過{feat['days_since_swing_high']}営業日(上限{cfg.pullback_max_duration_days})・"
                      f"反発確認済み、を全て満たす(参考: 52週高値比{feat['high52w_proximity']*100:.0f}%)")
        c.flags.append("⚠寄り付きで大きく上に窓が開いていたら追撃しない(この水準からの反発という前提が崩れる)")
    else:
        c.flags.append("ブレイク当日: 寄り付き直後の値動きで再確認してから発注(ダマシ注意)")
    c.checks = [
        f"出来高が平均比{feat['vol_ratio_today']:.1f}倍の裏取り(板・歩み値で確認)",
        f"ADX({feat['adx']:.0f}) — トレンド強度の参考値(閾値{cfg.adx_trend_th:.0f}は目安・厳格なゲートではない)",
        "同業種で他に強い銘柄が無いか(セクター全体の動きか個別かの確認)",
        "公開買付(TOB)・M&A等のコーポレートアクションが出ていないか(適時開示で確認)",
    ]
    return c
