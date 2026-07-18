"""topdown スクリーニング本体 — 地合い・セクター整合 → トリガー判定 → 本命最大5+次点.

トリガー優先順位(カタリスト第一の設計思想):
  GAP(カタリスト痕跡: 単日+4%ギャップ+出来高2倍・3営業日以内) > BREAK(20日高値ブレイク+出来高+トレンド)
  > PULL(momentum凍結5ゲートの押し目)
S高・急騰済み(+15%/1日 or +25%/3日)は本命に入れず監視(次点)へ格下げ+寄り天警告。
確信度はカタリスト出典と出来高の裏付けのみに基づく(チャット版の規律)。一次情報を確認できない
botでは上限「中」: GAP/BREAK(出来高裏付けあり)=中 / PULL(裏付けなし)=低 / 逆風セクター・高ボラで-1。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config
from .indicators import compute_topdown_features, compute_sector_rank, tob_suspect

TAG_SHORT = "短期スイング"   # 数日〜1週間(黄系)
TAG_SWING = "スイング"       # 1〜2週間(青系)

_CONF_ORDER = {"低": 0, "中": 1, "高": 2}


@dataclass
class Candidate:
    ticker: str
    code: str
    name: str
    sector: str
    trigger: str            # GAP / BREAK / PULL / SPIKE(監視のみ)
    trigger_text: str
    tag: str                # 保有期間タグ
    score: float
    feat: dict
    tailwind: bool = False  # セクター順風(上位3)
    headwind: bool = False  # セクター逆風(下位2)
    hivol: bool = False
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    risk_w: float = 0.0
    risk_pct: float = 0.0
    shares: int = 0
    time_stop: int = 0
    confidence: str = "低"
    conf_reason: str = ""
    risks: List[str] = field(default_factory=list)   # リスク要因・弱気シナリオ(2つ以上)
    flags: List[str] = field(default_factory=list)


def _round_tick(p: float) -> float:
    if p < 3000: t = 1
    elif p < 5000: t = 5
    elif p < 30000: t = 10
    elif p < 50000: t = 50
    else: t = 100
    return round(p / t) * t


def build_pool(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame], cfg: Config):
    eligible: List[dict] = []
    tob_rejects: List[dict] = []
    for _, row in universe.iterrows():
        tkr = str(row["ticker"]).strip()
        df = ohlcv.get(tkr)
        if df is None:
            continue
        feat = compute_topdown_features(df, cfg)
        if feat is None:
            continue
        if feat["close"] < cfg.min_price or feat["adv20_jpy"] < cfg.min_adv_jpy:
            continue
        is_tob, tob_reason = tob_suspect(df, cfg)
        if is_tob:
            tob_rejects.append({"code": tkr.replace(".T", ""), "name": row.get("name", ""),
                                "stage": "TOB疑い", "reason": tob_reason})
            continue
        eligible.append({"row": row.to_dict(), "feat": feat})
    return eligible, tob_rejects


def _decide_trigger(feat: dict) -> tuple[str, str, str] | None:
    """(trigger, trigger_text, tag) — 優先順位: GAP > BREAK > PULL"""
    if feat.get("gap_found"):
        return ("GAP",
                f"単日ギャップ+{feat['gap_ret']*100:.0f}%(出来高{feat['gap_vol_ratio']:.1f}倍・"
                f"{feat['gap_days_since']}営業日前)— カタリスト痕跡(真因は価格データでは判別不可)",
                TAG_SHORT)
    if feat.get("breakout_found"):
        return ("BREAK",
                f"20日高値ブレイク(出来高{feat['vol_ratio_today']:.1f}倍・上昇トレンド中)",
                TAG_SHORT)
    if feat.get("pullback_state_a"):
        return ("PULL", "上昇トレンド中の押し目+反発確認(凍結5ゲート通過)", TAG_SWING)
    return None


def _confidence(trigger: str, feat: dict, headwind: bool, hivol: bool) -> tuple[str, str]:
    bits = []
    if trigger == "GAP":
        level = 1; bits.append(f"出来高{feat['gap_vol_ratio']:.1f}倍の裏付けあり")
    elif trigger == "BREAK":
        level = 1; bits.append(f"出来高{feat['vol_ratio_today']:.1f}倍の裏付けあり")
    else:
        level = 0; bits.append("カタリスト・出来高の裏付けなし(テクニカルのみ)")
    if headwind:
        level -= 1; bits.append("逆風セクター(-1)")
    if hivol:
        level -= 1; bits.append("高ボラ(-1)")
    level = max(0, min(1, level))  # ★上限は常に「中」(一次情報を機械確認できないため)
    label = "中" if level == 1 else "低"
    bits.append("一次情報未確認のため上限は中")
    return label, " / ".join(bits)


def _risks_for(c: Candidate, sentiment: dict) -> List[str]:
    r = []
    if c.trigger == "GAP":
        r.append("ギャップの真因が不明(悪材料の可能性もある)— TDnet/iSPEEDで一次情報を確認するまで発注不可")
        r.append("ギャップ埋め(窓埋め)方向への反落が起きた場合、想定より早く損切りに到達する")
    elif c.trigger == "BREAK":
        r.append("ブレイクがダマシに終わり20日レンジ内へ回帰する場合(寄り付き直後の値動きで再確認)")
        r.append("出来高が続かない場合、ブレイク水準がそのまま天井になり得る")
    else:
        r.append("押し目がさらに深くなりトレンド自体が転換する場合(反発確認は前日時点のもの)")
        r.append("カタリスト不在のため、地合い悪化時に真っ先に売られやすい")
    if c.headwind:
        r.append(f"所属業種({c.sector})が直近5日で下位 — セクター逆風が続けば個別の形は無効化されやすい")
    if c.hivol:
        r.append("高ボラ銘柄: 通常より広い損切り幅(2.5ATR)を採用。ロットは通常より小さく")
    if sentiment.get("score", 3) <= 2:
        r.append("地合いスコア≤2(様子見・守り)— 候補が出ても必ず取引する必要はない")
    return r


def run_screen(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
               sentiment: dict, cfg: Config) -> dict:
    eligible, tob_rejects = build_pool(universe, ohlcv, cfg)
    sector_rank = compute_sector_rank(eligible, cfg)
    top_set = {s for s, _ in sector_rank["top"]}
    bottom_set = {s for s, _ in sector_rank["bottom"]}
    semis = set(cfg.semis_tickers)

    fired: List[Candidate] = []
    watch: List[Candidate] = []
    rejects: List[dict] = list(tob_rejects)

    for item in eligible:
        row, feat = item["row"], item["feat"]
        tkr = row["ticker"]
        sector = row.get("sector") or "不明"

        # 高ボラ環境での半導体・値がさ大型の扱い(改訂版ルール)
        is_semis = tkr in semis
        hivol = bool(sentiment.get("hivol_env")) and is_semis
        if is_semis and sentiment.get("semis_mode") == "exclude":
            rejects.append({"code": tkr.replace(".T", ""), "name": row.get("name", ""),
                            "stage": "高ボラ除外",
                            "reason": "VI代理>30かつ前夜SOX非反発のため値がさ大型を新規候補から除外"})
            continue

        # S高・急騰済み → 監視(次点)へ格下げ
        if feat.get("spiked"):
            trig = _decide_trigger(feat)
            c = Candidate(ticker=tkr, code=tkr.replace(".T", ""), name=row.get("name", ""),
                         sector=sector, trigger="SPIKE",
                         trigger_text=f"急騰済み(前日比+{feat['chg1d_pct']:.0f}%等)— 寄り天リスク",
                         tag=TAG_SHORT, score=0.0, feat=feat)
            c.flags.append("⚠寄り天(高値掴み)リスク: 原則監視。入るならギャップ後の値固め確認後")
            watch.append(c)
            continue

        trig = _decide_trigger(feat)
        if trig is None:
            continue
        trigger, ttext, tag = trig

        tailwind = sector in top_set
        headwind = sector in bottom_set

        base = {"GAP": 3.0, "BREAK": 2.0, "PULL": 1.0}[trigger]
        vol_backing = feat.get("gap_vol_ratio") if trigger == "GAP" else feat.get("vol_ratio_today")
        score = base + (2.0 if tailwind else 0.0) + min(float(vol_backing or 0), 5.0) * 0.2

        c = Candidate(ticker=tkr, code=tkr.replace(".T", ""), name=row.get("name", ""),
                     sector=sector, trigger=trigger, trigger_text=ttext, tag=tag,
                     score=score, feat=feat, tailwind=tailwind, headwind=headwind, hivol=hivol)

        # 価格計画(IN=前日終値基準・確定済み価格ベースの相対設計)
        entry = feat["close"]
        stop_mult = 2.5 if hivol else cfg.initial_stop_atr_mult
        stop_raw = entry - stop_mult * feat["atr"]
        if stop_raw <= 0:
            continue
        risk_w = entry - stop_raw
        if risk_w / entry * 100 > cfg.max_risk_width_pct:
            rejects.append({"code": c.code, "name": c.name, "stage": "リスク幅",
                            "reason": f"リスク幅{risk_w/entry*100:.1f}%が上限{cfg.max_risk_width_pct:.0f}%超"})
            continue
        c.entry = _round_tick(entry)
        c.stop = _round_tick(stop_raw)
        c.target = _round_tick(entry + risk_w * cfg.profit_target_r)
        c.risk_w = risk_w
        c.risk_pct = cfg.risk_pct_fixed
        c.time_stop = cfg.time_stop_short_swing if tag == TAG_SHORT else cfg.time_stop_swing
        c.confidence, c.conf_reason = _confidence(trigger, feat, headwind, hivol)
        if hivol:
            c.flags.append("⚠高ボラ銘柄: 広め損切り(2.5ATR)・小ロット推奨・確信度-1適用済み")
        if tailwind:
            c.flags.append(f"セクター順風({sector}=直近5日上位)")
        c.risks = _risks_for(c, sentiment)
        fired.append(c)

    # 地合い・セクター整合順に並べ、本命最大5(1業種2まで)
    fired.sort(key=lambda x: (-x.score, -_CONF_ORDER[x.confidence]))
    picked: List[Candidate] = []
    overflow: List[Candidate] = []
    sector_used: Dict[str, int] = {}
    for c in fired:
        if len(picked) >= cfg.max_candidates:
            overflow.append(c); continue
        if sector_used.get(c.sector, 0) >= cfg.max_per_sector:
            rejects.append({"code": c.code, "name": c.name, "stage": "セクター分散",
                            "reason": f"同一業種({c.sector})は{cfg.max_per_sector}銘柄まで → 次点"})
            overflow.append(c); continue
        if cfg.account_equity > 0:
            risk_amount = cfg.account_equity * c.risk_pct / 100
            c.shares = int(risk_amount / c.risk_w // 100 * 100)
            if c.shares * c.entry < cfg.min_exec_jpy:
                rejects.append({"code": c.code, "name": c.name, "stage": "サイジング",
                                "reason": f"サイズ過小(≈{c.shares*c.entry/1e4:.0f}万円)につき見送り"})
                continue
        picked.append(c)
        sector_used[c.sector] = sector_used.get(c.sector, 0) + 1

    runner_up = (overflow + watch)[: cfg.max_watch]

    stats = {
        "eligible": len(eligible), "tob_excluded": len(tob_rejects),
        "fired": len(fired), "spiked_watch": len(watch),
        "picked": len(picked), "rejected": len(rejects),
        "total_risk": sum(c.risk_pct for c in picked),
        "trigger_count": {t: sum(1 for c in fired if c.trigger == t) for t in ("GAP", "BREAK", "PULL")},
    }
    return {"picked": picked, "watch": runner_up, "rejects": rejects,
            "sector_rank": sector_rank, "stats": stats}
