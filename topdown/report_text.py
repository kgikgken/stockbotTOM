"""topdown — LINEテキストレポート(一新プロンプトの【出力】仕様に準拠)."""

from __future__ import annotations

from pathlib import Path

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
        return [{"date": r["date"], "label": r["label"], "kind": r.get("kind", "")} for _, r in ev.iterrows()]
    except Exception:
        return []


def build_text(today: str, meta: dict, sentiment: dict, res: dict,
               position_alerts: list, pos_note: str, events: list, cfg) -> str:
    L = []
    ap = L.append
    st = res["stats"]
    sr = res["sector_rank"]

    ap(f"◆stockbotTOM 新スクリーニング(地合い×セクター×カタリスト痕跡) {today}")
    ap("")

    # --- データ取得状況(冒頭・欠落列挙) ---
    ap(f"【データ取得状況】株価 {meta.get('data_ok','?')}/{meta.get('data_total','?')}"
       f"({meta.get('data_coverage',0)*100:.0f}%) {meta.get('source','')}")
    if sentiment.get("missing"):
        ap(f"　欠落指数: {', '.join(sentiment['missing'])}")
    ap("　※単一ソース(yfinance)。カタリストの中身・需給(信用残)・イベント詳細は取得不可 — iSPEED/TDnetで要確認")
    ap("")

    # --- 保有銘柄アラート ---
    notable = [a for a in (position_alerts or []) if a.get("hit") or a.get("hit") is None]
    if notable:
        ap("【保有銘柄アラート】")
        for a in notable:
            tag = {"target": "✅利確到達", "stop": "⚠ストップ到達", "time": "⏱時間ストップ"}.get(a.get("hit"), "要確認")
            ap(f"・{a['code']} {a['name']}: {tag} — {a['note']}")
        ap("")

    # --- 地合いサマリー ---
    prov = "(暫定)" if sentiment.get("provisional") else ""
    ap(f"【地合い】スコア {sentiment['score']}/5{prov} — 基本姿勢: {sentiment['stance']}")
    ap("　" + " / ".join(sentiment.get("reasons", [])[:4]))
    if sentiment.get("vi_proxy") is not None:
        note = f"VI代理(N225実現ボラ20d)={sentiment['vi_proxy']:.1f}"
        if sentiment.get("hivol_env"):
            note += " — 高ボラ環境。" + ("前夜SOX反発あり→値がさ大型は高ボラタグ付きで対象" if sentiment.get("sox_rebound")
                                        else "前夜SOX反発なし→値がさ大型は新規候補から除外")
        ap("　" + note)
    ap("")

    # --- セクター見立て ---
    ap("【セクター見立て(構成銘柄等ウェイト代理・直近5日)】")
    if sr["top"]:
        ap("　資金が向かいやすい上位: " + " / ".join(f"{s}({r:+.1f}%)" for s, r in sr["top"]))
    if sr["bottom"]:
        ap("　避けたい下位: " + " / ".join(f"{s}({r:+.1f}%)" for s, r in sr["bottom"]))
    ap("")

    # --- 本命候補 ---
    ap(f"【本日の候補(最大{cfg.max_candidates}・母集団{st['eligible']}銘柄・点灯 GAP{st['trigger_count']['GAP']}/BREAK{st['trigger_count']['BREAK']}/PULL{st['trigger_count']['PULL']})】")
    if res["picked"]:
        ap("★[要確認] 発注前にiSPEED/TDnetで適時開示(カタリストの中身)を必ず確認★")
    else:
        ap("該当なし — ゼロ件はゼロ件。無理に格下げ採用しない。")
    for i, c in enumerate(res["picked"], 1):
        ap(f"◆{i}. {c.code} {c.name} [{c.tag}] {c.sector}" + ("(順風)" if c.tailwind else "(逆風)" if c.headwind else ""))
        ap(f"   トリガー: {c.trigger_text}")
        ap(f"   IN {c.entry:,.0f}円(前日終値基準) / STOP {c.stop:,.0f}円 / 利確目安 {c.target:,.0f}円({cfg.profit_target_r:.1f}R)")
        ap(f"   リスク{c.risk_pct:.2f}%" + (f"(≈{c.shares}株)" if c.shares else "") + f" / 時間ストップ{c.time_stop}営業日")
        ap(f"   確信度: {c.confidence} — {c.conf_reason}")
        for r in c.risks[:3]:
            ap(f"   ⚠{r}")
        for fl in c.flags:
            ap(f"   {fl}")
    ap("")

    # --- 次点 ---
    if res["watch"]:
        ap("【次点(監視)】")
        for c in res["watch"]:
            ap(f"・{c.code} {c.name} [{c.trigger}] {c.trigger_text[:40]}")
            for fl in c.flags:
                ap(f"　{fl}")
        ap("")

    # --- 共通リスク ---
    ap("【本日全体のメモ(共通リスク)】")
    ap("・全候補ともカタリストの中身は未確認(価格痕跡のみ)。悪材料由来の急変動の可能性を常に残す")
    ap(f"・総リスク: 新規計 {st['total_risk']:.2f}% | {pos_note}")
    ap("")

    # --- 今週の重要イベント ---
    ap("【今週の重要イベント(events.csv・手動管理)】")
    if events:
        for e in events:
            ap(f"・{e['date']} {e['label']}")
    else:
        ap("・登録なし(経済指標カレンダーは自動取得不可 — 手動で確認すること)")
    ap("")

    ap("【免責】AIによる候補提示で投資助言ではない。数値は単一ソースにつき要再確認。"
       "候補が出た日に必ず取引する必要はない。最終判断と結果責任はユーザーにある。")
    return "\n".join(L)
