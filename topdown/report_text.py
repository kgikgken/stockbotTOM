"""topdown — LINEテキストレポート v2.0(ゾーン入口・構造ストップ・トリガー別出口)."""

from __future__ import annotations

from pathlib import Path

from .market import closing_note

import pandas as pd


def load_week_events(cfg, today: str) -> list:
    p = Path(cfg.events_path)
    if not p.exists():
        return []
    try:
        ev = pd.read_csv(p, dtype=str).fillna("")
        ev["date_p"] = pd.to_datetime(ev["date"], errors="coerce")
        t0 = pd.Timestamp(today)
        t1 = t0 + pd.Timedelta(days=cfg.events_horizon_days)
        ev = ev[(ev["date_p"] >= t0) & (ev["date_p"] <= t1)].sort_values("date_p")
        return [{"date": r["date"], "label": r["label"]} for _, r in ev.iterrows()]
    except Exception:
        return []


def build_text(today: str, meta: dict, sentiment: dict, res: dict,
               position_alerts: list, pending_summary: dict, pending_events: list,
               events: list, cfg) -> str:
    L = []
    ap = L.append
    st = res["stats"]
    sr = res["sector_rank"]
    tc = st["trigger_count"]

    ap(f"◆stockbotTOM 新スクリーニング {today}")
    ap("")

    ap(f"【データ】{meta.get('data_ok','?')}/{meta.get('data_total','?')}"
       f"({meta.get('data_coverage',0)*100:.0f}%) {meta.get('source','')}")
    if sentiment.get("missing"):
        ap(f"　欠落: {', '.join(sentiment['missing'])}")
    ap("　※カタリストの中身・需給は取得不可 — 発注前にiSPEED/TDnetで要確認")
    ap("")

    prov = "(暫定)" if sentiment.get("provisional") else ""
    ap(f"【地合い】{sentiment['score']}/5{prov} — {sentiment['stance']}")
    ap("　" + " / ".join(sentiment.get("reasons", [])[:4]))
    if sentiment.get("vi_proxy") is not None:
        note = f"VI代理={sentiment['vi_proxy']:.1f}"
        if sentiment.get("hivol_env"):
            note += " — 高ボラ環境。" + (
                "前夜SOX反発あり→値がさ大型は高ボラタグ付きで対象"
                if sentiment.get("sox_rebound") else sentiment.get("semis_reason", "値がさ大型は除外"))
        ap("　" + note)
    n_pos = len([a for a in (position_alerts or [])])
    ap(f"　保有中 {n_pos}件 / ゾーン待ち {pending_summary.get('pending',0)}件")
    ap("")

    if pending_events:
        ap("【ゾーン待ち候補の動き】")
        for e in pending_events:
            mark = {"reached": "✅到達", "expired": "⏱失効", "broken": "✕下端割れ"}.get(e["event"], "")
            ap(f"・{mark} {e['code']} {e['name']}: {e['note']}")
        ap("")

    notable = [a for a in (position_alerts or []) if a.get("hit") or a.get("hit") is None]
    if notable:
        ap("【保有銘柄】")
        for a in notable:
            tag = {"stop": "⚠ストップ割れ", "partial": "📈+1R到達",
                   "time": "⏱時間ストップ"}.get(a.get("hit"), "要確認")
            prog = ""
            if a.get("days_held") is not None and a.get("time_stop"):
                prog = f" [{a['days_held']}/{a['time_stop']}営業日]"
            ap(f"・{tag} {a['code']} {a['name']}{prog}")
            ap(f"　{a['note']}")
        ap("")

    ap("【セクター(等ウェイト代理・直近5日)】")
    if sr["top"]:
        ap("　↑上位: " + " / ".join(f"{s}({r:+.1f}%)" for s, r in sr["top"]))
    if sr["bottom"]:
        ap("　↓回避: " + " / ".join(f"{s}({r:+.1f}%)" for s, r in sr["bottom"]))
    ap("")

    ap(f"【本日の候補】母集団{st['eligible']} / 点灯 "
       f"材料反応{tc.get('材料反応',0)}・押し目{tc.get('押し目',0)}・高値ブレイク{tc.get('高値ブレイク',0)}")
    if st.get("slot_note"):
        ap(f"　▼{st['slot_note']}")
    if st.get("concentration"):
        ap(f"　⚠{st['concentration']}")
    if res["picked"]:
        ap("★[要確認] iSPEED/TDnetでカタリストの中身を確認するまで発注不可")
    else:
        ap("該当なし — ゼロ件はゼロ件。無理に格下げ採用しない。")
    for i, c in enumerate(res["picked"], 1):
        ap("")
        ap(f"◆{i}. {c.code} {c.name} [{c.trigger}/めやす{c.tag}] {c.sector}"
           + ("(順風)" if c.tailwind else "(逆風)" if c.headwind else ""))
        ap(f"   {c.trigger_text}")
        ap(f"   INゾーン(指値) {c.zone_hi:,.0f} 〜 {c.zone_lo:,.0f} 円")
        ap(f"   STOP {c.stop:,.0f} 円(構造・どこで入っても同じ)")
        ap(f"   1Rの幅: 浅く{c.risk_shallow:,.0f}円({c.risk_pct_shallow:.1f}%) / "
           f"深く{c.risk_deep:,.0f}円({c.risk_pct_deep:.1f}%)")
        ap(f"   失効{c.expire_date}({cfg.zone_expire_days}営業日) / 時間ストップ{c.time_stop}営業日 / "
           f"1単元{c.unit_cost/1e4:,.0f}万円")
        ap(f"   期待度 {c.score:.0f}/10 — {c.score_reason}")
        ap("   出口: +1Rで半分利確(2単元以上) → 残玉は構造まで引上げ+トレーリング / 固定利確なし")
        for r in c.risks[:2]:
            ap(f"   ⚠{r}")
        for fl in c.flags:
            ap(f"   {fl}")
    ap("")

    if res["watch"]:
        ap("【次点(監視)】")
        for c in res["watch"]:
            ap(f"・{c.code} {c.name} [{c.trigger}] {c.trigger_text[:44]}")
            for fl in c.flags:
                ap(f"　{fl}")
        ap("")

    ap("【今週の重要イベント(events.csv・手動管理)】")
    if events:
        for e in events:
            ap(f"・{e['date']} {e['label']}")
    else:
        ap("・登録なし(経済指標カレンダーは自動取得不可 — 手動確認)")
    ap("")
    cn = closing_note(sentiment)
    ap(f"【最後に: 今日の地合い】{cn['score']}/5{'(暫定)' if cn['provisional'] else ''} — {cn['stance']}")
    for ln in cn["lines"]:
        ap("　" + ln)
    ap("")
    ap("【免責】AIによる候補提示で投資助言ではない。数値は単一ソースにつき要再確認。"
       "候補が出た日に必ず取引する必要はない。最終判断と結果責任はユーザーにある。")
    return "\n".join(L)
