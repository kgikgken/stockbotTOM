"""topdown スクリーニング本体 v2.0 — ゾーン入口・構造ストップ・トリガー別出口.

v1.0からの変更(2026-07-19・仕様書v2.0):
- トリガーを日本語名に改称し、日本市場の実証地形に沿って序列を付けた
    材料反応(主力) > 押し目(準主力) > 高値ブレイク(補助・セクター順風時のみ)
- エントリーを「前日終値」から「浅いゾーンへの指値」に変更(浅く買って構造が壊れたら切る)
- ストップをATR固定距離から構造(直近安値)ベースへ変更
- 固定2R利確を撤廃(2Rは参照点)。出口はトリガー別の時間ストップ+トレーリング
- 価格上限20,000円、相関集中の抑制、決算またぎ推定(警告のみ)を追加
※本ファイルの全パラメータは未検証。実証との整合であって有効性の証明ではない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from .config import Config
from .indicators import compute_topdown_features, compute_sector_rank, tob_suspect

TRIG_GAP, TRIG_PULL, TRIG_BREAK = "材料反応", "押し目", "高値ブレイク"
TAG_MONTH, TAG_2WEEK = "1ヶ月", "2週間"

@dataclass
class Candidate:
    ticker: str
    code: str
    name: str
    sector: str
    trigger: str
    trigger_text: str
    tag: str
    score: float
    feat: dict
    tailwind: bool = False
    headwind: bool = False
    hivol: bool = False
    # --- 入口ゾーンと構造ストップ ---
    zone_hi: float = 0.0
    zone_lo: float = 0.0
    stop: float = 0.0
    risk_shallow: float = 0.0      # ゾーン上端で入った場合の1R幅(円)
    risk_deep: float = 0.0         # ゾーン下端で入った場合の1R幅(円)
    risk_pct_shallow: float = 0.0
    risk_pct_deep: float = 0.0
    unit_cost: float = 0.0         # 1単元(100株)の金額
    # --- 出口 ---
    time_stop: int = 0
    expire_date: str = ""
    # --- 付帯情報 ---
    score_reason: str = ""
    gap_date: str | None = None
    earnings_est_days: int | None = None
    risks: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


def _round_tick(p: float) -> float:
    if p < 3000: t = 1
    elif p < 5000: t = 5
    elif p < 30000: t = 10
    elif p < 50000: t = 50
    else: t = 100
    return round(p / t) * t


def _bday_str(today: str, n: int) -> str:
    try:
        return (pd.Timestamp(today) + pd.tseries.offsets.BDay(n)).strftime("%m/%d")
    except Exception:
        return ""


def build_pool(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame], cfg: Config):
    """流動性・価格上限・TOB疑いを通過した母集団を作る。"""
    eligible: List[dict] = []
    rejects: List[dict] = []
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
        if feat["close"] > cfg.max_price:
            rejects.append({"code": tkr.replace(".T", ""), "name": row.get("name", ""),
                            "stage": "価格上限",
                            "reason": f"株価{feat['close']:,.0f}円が上限{cfg.max_price:,.0f}円超"
                                      f"(1単元{feat['close']*100/1e4:,.0f}万円)"})
            continue
        is_tob, tob_reason = tob_suspect(df, cfg)
        if is_tob:
            rejects.append({"code": tkr.replace(".T", ""), "name": row.get("name", ""),
                            "stage": "TOB疑い", "reason": tob_reason})
            continue
        eligible.append({"row": row.to_dict(), "feat": feat})
    return eligible, rejects


def build_zone(trigger: str, feat: dict, cfg: Config):
    """入口ゾーンと構造ストップを組む。

    - ゾーン上端は STOP + max_risk_atr_mult×ATR で切り落とす(合否判定には使わない)
    - 切った結果ゾーンが成立しない、または現値がゾーン下端未満なら見送り
    戻り値: ((zone_hi, zone_lo, stop, capped), None) または (None, 見送り理由)
    """
    atr = feat.get("atr")
    if not atr or atr <= 0:
        return None, "ATR算出不可"
    buf = cfg.stop_buffer_atr_mult * atr
    close_now = feat["close"]

    if trigger == TRIG_GAP:
        gh, gl = feat.get("gap_high"), feat.get("gap_low")
        if gh is None or gl is None or gh <= gl:
            return None, "ギャップ足の構造を取得できない"
        stop = gl - buf
        zone_hi = gh
        zone_lo = gh - (gh - gl) * cfg.zone_fib_retrace
    elif trigger == TRIG_BREAK:
        lvl, plow = feat.get("breakout_level"), feat.get("pre_breakout_low")
        if lvl is None or plow is None or lvl <= plow:
            return None, "ブレイク構造を取得できない"
        stop = plow - buf
        zone_hi = lvl
        zone_lo = lvl - cfg.zone_break_atr_mult * atr
    else:  # 押し目
        ph, dl = feat.get("prev_day_high"), feat.get("dip_low")
        if ph is None or dl is None or ph <= dl:
            return None, "押し目構造を取得できない"
        stop = dl - buf
        zone_hi = ph
        # ★2026-07-23: 下端を押し安値そのものから38.2%押しへ変更。
        # 上端の切り落としを廃止した結果、押し目のゾーンだけ「前日高値〜押し安値」の
        # 全幅となり青天井に広がった(押し3ATRで幅2.7ATR)。材料反応と同じく
        # 実証のある38.2%(Alajbeg 2017/Bulkowski「浅い押し>深い押し」)で浅い側に
        # 揃え、3トリガーすべてが有界かつ浅い帯になるようにした。
        # 押し安値そのものは STOP の根拠として引き続き使う(構造は不変)。
        zone_lo = ph - (ph - dl) * cfg.zone_fib_retrace

    if stop <= 0:
        return None, "ストップ価格が0以下"

    # ★下端の引き上げ: 構造ちょうどを拾うとリスク幅が緩衝分(0.2ATR)しか無くなるため、
    # 最低でも min_risk_atr_mult×ATR の損切り余地を残す位置まで下端を上げる。
    floor = stop + cfg.min_risk_atr_mult * atr
    floored = zone_lo < floor
    zone_lo = max(zone_lo, floor)
    if zone_hi <= zone_lo:
        return None, "構造が反転している(下端が上端を上回る)"

    # ★リスク上限は「深い端で入ってもなお広すぎるか」で判定する(2026-07-23改訂)。
    # 上端を切り落とす旧設計はゾーンそのものを潰したため、帯の形は構造どおり保ち、
    # 最良のケース(深い端)でも上限を超える銘柄だけを見送る。
    risk_deep_atr = (zone_lo - stop) / atr
    if risk_deep_atr > cfg.max_risk_atr_mult:
        return None, (f"深い端で入ってもリスク{risk_deep_atr:.1f}ATRと広すぎる"
                      f"(上限{cfg.max_risk_atr_mult:.1f}ATR)")

    # ★リスク幅の絶対上限%(ATR相対だけだと高ボラ銘柄が損切り幅20%超で通ってしまうため)
    risk_deep_pct = (zone_lo - stop) / zone_lo * 100
    if risk_deep_pct > cfg.max_risk_width_pct:
        return None, (f"深い端でもリスク幅{risk_deep_pct:.1f}%と大きい"
                      f"(上限{cfg.max_risk_width_pct:.0f}%)")

    # ★退化ゾーンの検出: 帯が細すぎると「ゾーンで待つ」ではなくピンポイント指値になる。
    if (zone_hi - zone_lo) / atr < cfg.min_zone_width_atr:
        return None, (f"ゾーン幅{(zone_hi - zone_lo)/atr:.2f}ATRが最低幅"
                      f"{cfg.min_zone_width_atr:.1f}ATR未満(帯として成立しない)")

    if close_now < zone_lo:
        return None, "現値がすでにゾーン下端を下回る(構造が否定されつつある)"
    return (zone_hi, zone_lo, stop, floored), None


def _decide_trigger(feat: dict, tailwind: bool):
    """(trigger, 説明, 保有タグ, 序列) — 序列: 材料反応3 > 押し目2 > 高値ブレイク1"""
    if feat.get("gap_found"):
        return (TRIG_GAP,
                f"単日ギャップ+{feat['gap_ret']*100:.0f}%(出来高{feat['gap_vol_ratio']:.1f}倍・"
                f"{feat['gap_days_since']}営業日前)— カタリスト痕跡(中身は要確認)",
                TAG_MONTH, 3)
    if feat.get("pullback_state_a"):
        return (TRIG_PULL, "上昇トレンド中の押し目+反発確認", TAG_2WEEK, 2)
    if feat.get("breakout_found"):
        # 高値ブレイクはセクター順風時のみ採用(日本のモメンタムは弱く単独では根拠が薄い)
        if not tailwind:
            return None
        return (TRIG_BREAK,
                f"20日高値ブレイク(出来高{feat['vol_ratio_today']:.1f}倍・セクター順風)",
                TAG_2WEEK, 1)
    return None


def _expectation_score(trigger: str, feat: dict, tailwind: bool, headwind: bool, hivol: bool):
    """期待度スコア(0〜10)。大きいほど上位に並べる。

    配点の軸は「日本市場でその現象にリターンの実証がどれだけあるか」に統一した。
    値動きの願望ではなく、調査(2026-07-19)で確認した実証の強さを点にしている。
    ★配点そのものは全て仮置き。各項目を台帳に個別記録し、数十件たまった時点で
      「どの項目が実現Rと相関したか」を検証して調整する前提。
    """
    parts = []
    if trigger == TRIG_GAP:
        pts = 4; parts.append("材料反応+4(日本でもPEAD確認)")
    elif trigger == TRIG_PULL:
        pts = 3; parts.append("押し目+3(日本は短期反転が強い)")
    else:
        pts = 1; parts.append("高値ブレイク+1(日本のモメンタムは弱い)")

    # 反応の強さ。出来高が仕掛けの定義に入っているトリガーのみ加点する
    # (押し目は下げ日の出来高が逆符号になりうるため加点せず、質は下の2項目で見る)
    if trigger in (TRIG_GAP, TRIG_BREAK):
        vr = feat.get("gap_vol_ratio") if trigger == TRIG_GAP else feat.get("vol_ratio_today")
        if vr:
            if vr >= 3.0: pts += 2; parts.append(f"出来高{vr:.1f}倍+2")
            elif vr >= 2.0: pts += 1; parts.append(f"出来高{vr:.1f}倍+1")

    # 鮮度: PEADは発表直後が最も強い
    if trigger == TRIG_GAP and feat.get("gap_days_since") == 0:
        pts += 1; parts.append("当日ギャップ+1")

    if feat.get("trend_align"):
        pts += 1; parts.append("トレンド整合+1")

    cp = feat.get("close_pos")
    if cp is not None and cp >= 0.7:
        pts += 1; parts.append(f"高値引け({cp*100:.0f}%)+1")

    if tailwind: pts += 1; parts.append("順風セクター+1")
    if headwind: pts -= 1; parts.append("逆風セクター-1")
    if hivol: pts -= 1; parts.append("高ボラ-1")

    return max(0, min(10, pts)), " / ".join(parts)


def _risks_for(c: Candidate, sentiment: dict) -> List[str]:
    r = []
    if c.trigger == TRIG_GAP:
        r.append("ギャップの真因が不明(悪材料の可能性もある)— TDnet/iSPEEDで確認するまで発注不可")
        r.append("ギャップ埋め方向へ反落した場合、ゾーン下端割れで即失効となる")
    elif c.trigger == TRIG_BREAK:
        r.append("ブレイクがダマシに終わり20日レンジ内へ回帰する場合")
        r.append("日本株のモメンタムは弱く、ブレイク追随の優位は米国より小さいとされる")
    else:
        r.append("押し目がさらに深くなりトレンド自体が転換する場合(反発確認は前日時点のもの)")
        r.append("カタリスト不在のため、地合い悪化時に真っ先に売られやすい")
    if c.headwind:
        r.append(f"所属業種({c.sector})が直近5日で下位 — セクター逆風")
    if c.hivol:
        r.append("高ボラ銘柄: ロットは通常より小さく")
    if sentiment.get("score", 3) <= 2:
        r.append("地合いスコア≤2 — 候補が出ても必ず取引する必要はない")
    return r


def run_screen(universe: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame],
               sentiment: dict, cfg: Config, today: str) -> dict:
    eligible, rejects = build_pool(universe, ohlcv, cfg)
    sector_rank = compute_sector_rank(eligible, cfg)
    top_set = {s for s, _ in sector_rank["top"]}
    bottom_set = {s for s, _ in sector_rank["bottom"]}
    semis = set(cfg.semis_tickers)

    fired: List[Candidate] = []
    watch: List[Candidate] = []

    for item in eligible:
        row, feat = item["row"], item["feat"]
        tkr = row["ticker"]
        sector = row.get("sector") or "不明"
        tailwind = sector in top_set
        headwind = sector in bottom_set

        is_semis = tkr in semis
        hivol = bool(sentiment.get("hivol_env")) and is_semis
        if is_semis and sentiment.get("semis_mode") == "exclude":
            rejects.append({"code": tkr.replace(".T", ""), "name": row.get("name", ""),
                            "stage": "高ボラ除外",
                            "reason": "VI代理>30かつ前夜SOX非反発のため値がさ大型を除外"})
            continue

        if feat.get("spiked"):
            c = Candidate(ticker=tkr, code=tkr.replace(".T", ""), name=row.get("name", ""),
                         sector=sector, trigger="急騰済み",
                         trigger_text=f"前日比+{feat['chg1d_pct']:.0f}%等の急騰済み",
                         tag=TAG_2WEEK, score=0.0, feat=feat)
            c.flags.append("⚠寄り天リスク: 原則監視。入るなら値固め確認後")
            watch.append(c)
            continue

        hit = _decide_trigger(feat, tailwind)
        if hit is None:
            continue
        trigger, ttext, tag, rank = hit

        zone, why = build_zone(trigger, feat, cfg)
        if zone is None:
            rejects.append({"code": tkr.replace(".T", ""), "name": row.get("name", ""),
                            "stage": "ゾーン不成立", "reason": why})
            continue
        zone_hi, zone_lo, stop, floored = zone

        c = Candidate(ticker=tkr, code=tkr.replace(".T", ""), name=row.get("name", ""),
                     sector=sector, trigger=trigger, trigger_text=ttext, tag=tag,
                     score=0.0, feat=feat,
                     tailwind=tailwind, headwind=headwind, hivol=hivol)
        c.zone_hi = _round_tick(zone_hi)
        c.zone_lo = _round_tick(zone_lo)
        c.stop = _round_tick(stop)
        c.risk_shallow = c.zone_hi - c.stop
        c.risk_deep = c.zone_lo - c.stop
        c.risk_pct_shallow = c.risk_shallow / c.zone_hi * 100 if c.zone_hi else 0
        c.risk_pct_deep = c.risk_deep / c.zone_lo * 100 if c.zone_lo else 0
        c.unit_cost = feat["close"] * 100
        c.time_stop = {TRIG_GAP: cfg.time_stop_gap, TRIG_BREAK: cfg.time_stop_break,
                       TRIG_PULL: cfg.time_stop_pull}[trigger]
        c.expire_date = _bday_str(today, cfg.zone_expire_days)
        c.gap_date = feat.get("gap_date")
        c.earnings_est_days = feat.get("earnings_est_days")
        c.score, c.score_reason = _expectation_score(trigger, feat, tailwind, headwind, hivol)
        c.risks = _risks_for(c, sentiment)

        if floored:
            c.flags.append(f"ゾーン下端をリスク下限{cfg.min_risk_atr_mult:.1f}ATRまで引き上げ済み"
                          "(底値ちょうどは狙わない)")
        if c.earnings_est_days is not None and c.earnings_est_days <= c.time_stop:
            c.flags.append(f"⚠推定: 約{c.earnings_est_days}営業日後に決算の可能性"
                          f"(保有{c.time_stop}日に重なる・要iSPEED確認)")
        if c.unit_cost > 1.5e6:
            c.flags.append(f"⚠1単元{c.unit_cost/1e4:,.0f}万円 — 1単元のみなら半分利確は使えず"
                          "トレーリング+時間ストップ一本")
        fired.append(c)

    # --- 絞り込み: 序列 → セクター分散 → 相関集中の抑制 ---
    fired.sort(key=lambda x: -x.score)

    # 地合いに応じて採用枠を絞る(元プロンプトの「基本姿勢」を出力に反映させる)
    max_slots = cfg.max_candidates
    slot_note = ""
    if cfg.slots_by_sentiment:
        sc = int(sentiment.get("score", 3))
        max_slots = {5: cfg.max_candidates, 4: cfg.max_candidates,
                     3: max(1, cfg.max_candidates - 1),
                     2: max(1, cfg.max_candidates - 2),
                     1: max(1, cfg.max_candidates - 3)}.get(sc, cfg.max_candidates)
        if max_slots < cfg.max_candidates:
            slot_note = (f"地合い{sc}/5のため採用枠を{cfg.max_candidates}→{max_slots}件に縮小"
                         f"({sentiment.get('stance','')})")

    # 材料反応が点灯していれば最低1枠を確保する(PEAD検証のサンプルを絶やさないため)
    reserved: Candidate | None = None
    if cfg.reserve_gap_slot and max_slots > 0:
        gaps = [c for c in fired if c.trigger == TRIG_GAP]
        if gaps and not any(c.trigger == TRIG_GAP for c in fired[:max_slots]):
            reserved = gaps[0]

    picked: List[Candidate] = []
    overflow: List[Candidate] = []
    sector_used: Dict[str, int] = {}
    gap_count = 0
    gapday_used: Dict[str, int] = {}

    for c in fired:
        if len(picked) >= max_slots:
            overflow.append(c); continue
        if sector_used.get(c.sector, 0) >= cfg.max_per_sector:
            rejects.append({"code": c.code, "name": c.name, "stage": "セクター分散",
                            "reason": f"同一業種({c.sector})は{cfg.max_per_sector}件まで"})
            overflow.append(c); continue
        if c.trigger == TRIG_GAP:
            if gap_count >= cfg.max_gap_candidates:
                rejects.append({"code": c.code, "name": c.name, "stage": "相関集中",
                                "reason": f"材料反応は{cfg.max_gap_candidates}件まで"})
                overflow.append(c); continue
            if c.gap_date and gapday_used.get(c.gap_date, 0) >= cfg.max_same_gapday:
                rejects.append({"code": c.code, "name": c.name, "stage": "相関集中",
                                "reason": f"同一ギャップ日({c.gap_date})は{cfg.max_same_gapday}件まで"})
                overflow.append(c); continue
        picked.append(c)
        sector_used[c.sector] = sector_used.get(c.sector, 0) + 1
        if c.trigger == TRIG_GAP:
            gap_count += 1
            if c.gap_date:
                gapday_used[c.gap_date] = gapday_used.get(c.gap_date, 0) + 1

    if reserved is not None and reserved not in picked:
        if len(picked) >= max_slots and picked:
            dropped = picked.pop()
            overflow.append(dropped)
        picked.append(reserved)
        reserved.flags.append("材料反応の予約枠で採用(PEAD検証のサンプル確保)")

    concentration = ""
    picked_gapdays: Dict[str, int] = {}
    for c in picked:
        if c.trigger == TRIG_GAP and c.gap_date:
            picked_gapdays[c.gap_date] = picked_gapdays.get(c.gap_date, 0) + 1
    if picked_gapdays:
        top_day, top_n = max(picked_gapdays.items(), key=lambda kv: kv[1])
        if top_n >= 2:
            concentration = f"本日{len(picked)}件中{top_n}件が同一カタリスト日({top_day})— 相関に注意"

    runner_up = (overflow + watch)[: cfg.max_watch]
    stats = {
        "eligible": len(eligible), "fired": len(fired), "spiked_watch": len(watch),
        "picked": len(picked), "rejected": len(rejects),
        "trigger_count": {t: sum(1 for c in fired if c.trigger == t)
                          for t in (TRIG_GAP, TRIG_PULL, TRIG_BREAK)},
        "concentration": concentration, "slot_note": slot_note, "max_slots": max_slots,
    }
    return {"picked": picked, "watch": runner_up, "rejects": rejects,
            "sector_rank": sector_rank, "stats": stats}
