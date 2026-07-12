"""エンジンR(反転・低回転率母集団向け)/エンジンM(モメンタム入口・高回転率母集団向け)の
候補判定 + エントリー・初期ストップ・固定利確目標の算出。

出口はシャンデリア・トレーリングではなく固定利確+時間ストップのハイブリッド
(調査レポート2026-07-11の中核提案: 平均回帰的な値幅には固定目標が実証的に有効)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Candidate:
    ticker: str
    code: str
    name: str
    sector: str
    market: str
    engine: str            # "R"(反転) or "M"(モメンタム入口)
    trigger: str            # 人間可読のトリガー説明
    score: float
    feat: dict
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
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


def _finalize(row: dict, feat: dict, engine: str, trigger: str, score: float, cfg) -> Candidate | None:
    entry = feat["close"]
    stop_raw = entry - cfg.initial_stop_atr_mult * feat["atr"]
    if stop_raw <= 0:
        return None
    risk_w = entry - stop_raw
    if risk_w <= 0 or risk_w / entry * 100 > cfg.max_risk_width_pct:
        return None
    target_raw = entry + risk_w * cfg.profit_target_r

    tkr = row["ticker"]
    c = Candidate(ticker=tkr, code=tkr.replace(".T", ""), name=row.get("name", ""),
                 sector=row.get("sector", ""), market=row.get("market", "Prime"),
                 engine=engine, trigger=trigger, score=score, feat=feat)
    c.entry = _round_tick(entry)
    c.stop = _round_tick(stop_raw)
    c.target = _round_tick(target_raw)
    c.risk_w = risk_w
    c.checks = [
        f"出来高が平均比{feat['vol_ratio_today']:.1f}倍の裏取り(板・歩み値で確認)" if feat.get("vol_ratio_today") == feat.get("vol_ratio_today") else "出来高の裏取り",
        f"RSI({feat['rsi']:.0f}) — 補助確認値(主軸は業種内相対リターン)",
        "公開買付(TOB)・M&A等のコーポレートアクションが出ていないか(適時開示で確認)",
        f"時間ストップ: 建玉から{cfg.time_stop_days}営業日経過で機械的に手仕舞い(要カレンダー管理)",
    ]
    return c


def classify_engine_r(item: dict, sector_z: dict, cfg) -> Candidate | None:
    """低〜中回転率母集団向け: 業種内相対で売られ過ぎ(主軸・維持) + 深さレンジ(追加) +
    強化反発確認(追加) + RSI(補助確認のみ)。

    調査レポート(2026-07-11)を反映: z値基準(≤-1.2)はそのまま維持しつつ、
    ①深さが浅すぎない/深すぎない(50日線-3%〜-15% または ATR1.5〜5倍)、
    ②反発確認を強化(CLV≥0.5相当 または 前日高値を上抜け、いずれか一方)、を追加条件とした。
    """
    feat = item["feat"]
    ticker = item["row"]["ticker"]
    z = sector_z.get(ticker)
    if z is None or z > cfg.rel_oversold_z:
        return None

    # 深さレンジ(浅すぎ/深すぎを除外)。50日線基準とATR基準のどちらかを満たせばよい。
    ma50_ok = False
    if feat.get("ma50_ratio") == feat.get("ma50_ratio"):  # NaNでない
        ma50_ok = -cfg.r_depth_ma50_max_pct / 100 <= feat["ma50_ratio"] <= -cfg.r_depth_ma50_min_pct / 100
    atr_ok = cfg.r_depth_atr_min <= feat.get("pullback_depth_atr", 0) <= cfg.r_depth_atr_max
    if not (ma50_ok or atr_ok):
        return None

    # 強化反発確認: CLV相当(bounce_confirmed) または 前日高値を上抜け(bounce_confirmed_strong)。
    # ★両方AND必須は絞り込みすぎと判明、いずれか一方で十分とするOR条件に緩和
    if not (feat.get("bounce_confirmed") or feat.get("bounce_confirmed_strong")):
        return None

    c = _finalize(item["row"], feat, "R",
                 f"業種内相対z={z:.2f}(直近{cfg.rel_lookback_days}日)+ 深さATR×{feat['pullback_depth_atr']:.1f}"
                 f"+ 強化反発確認 + RSI({feat['rsi']:.0f})",
                 score=-z, cfg=cfg)  # zが低い(より売られている)ほどスコア高
    if c is not None:
        c.flags.append("エンジンR(反転・精緻化基準): 業種内で相対的に売られ過ぎ + 深さが浅すぎず深すぎず"
                      "(falling knife除外) + 前日高値を上抜ける強い反発を確認済み。深追い厳禁")
        c.flags.append("⚠寄り付きで大きく上に窓が開いていたら追撃しない(この水準からの反発という前提が崩れる)")
        if feat["rsi"] <= cfg.rsi_secondary_th:
            c.flags.append(f"RSI({feat['rsi']:.0f})も過熱感の解消を示唆(補助確認・主軸ではない)")
        if feat.get("low_volume_decline") is True:
            c.flags.append("(参考)下落局面の出来高は過去20日平均より少なめ — 健全な押し目の傍証"
                          "(学術的裏付けは弱いため参考情報に留める)")
    return c


def classify_engine_m(item: dict, cfg) -> Candidate | None:
    """高回転率母集団向け: ①単日ギャップ+出来高急増 または ②52週高値ブレイク+出来高確認。"""
    feat = item["feat"]
    if feat.get("gap_found"):
        c = _finalize(item["row"], feat, "M",
                     f"単日ギャップ+{feat['gap_ret']*100:.0f}%(出来高{feat['gap_vol_ratio']:.1f}倍・{feat['gap_days_since']}営業日前)",
                     score=feat["gap_ret"] * 100, cfg=cfg)
        if c is not None:
            c.flags.append("エンジンM(モメンタム入口): 好材料由来の値動きの初動のみを取る設計。"
                          "決算等のコーポレートアクションか要確認(価格データのみでは真因は判別不可)")
        return c
    if feat.get("breakout_found"):
        c = _finalize(item["row"], feat, "M",
                     f"52週高値ブレイク(出来高{feat['vol_ratio_today']:.1f}倍)",
                     score=feat.get("ret_lookback", 0) * 100, cfg=cfg)
        if c is not None:
            c.flags.append("エンジンM(モメンタム入口): 52週高値更新の初動。継続するかはこの先次第")
        return c
    return None
