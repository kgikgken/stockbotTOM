"""swing2w — レポートのテキストセクション生成(momentum側の統合レポートに追記される前提)."""

from __future__ import annotations


def build_text_section(today: str, res: dict, position_alerts: list, pos_note: str, cfg) -> list:
    L = []
    ap = L.append
    st = res["stats"]

    notable_pos = [a for a in (position_alerts or []) if a.get("hit") or a.get("hit") is None]
    if notable_pos:
        ap("【2週間スイング・保有銘柄アラート】")
        for a in notable_pos:
            tag = {"target": "✅利確到達", "stop": "⚠ストップ到達", "time": "⏱時間ストップ"}.get(a.get("hit"), "要確認")
            ap(f"・{a['code']} {a['name']}: {tag} — {a['note']}")
        ap("")

    ap("═" * 20)
    ap(f"【2週間スイング(回転率二分・固定利確+時間ストップ)】")
    ap(f"母集団{st.get('universe_considered',0)}銘柄(TOB疑い{st.get('tob_excluded',0)}件除外済) → "
       f"低回転率{st.get('low_turnover_n',0)}銘柄(エンジンR対象) / "
       f"高回転率{st.get('high_turnover_n',0)}銘柄(エンジンM対象)")
    ap(f"点灯 R{st.get('fired_r',0)}件 / M{st.get('fired_m',0)}件 → "
       f"実保有上限{cfg.max_positions}銘柄(モメンタムとは別枠)・セクター1業種までの制約でアクション候補{st.get('picked',0)}件")
    ap("")

    if res["picked"] or res.get("watch"):
        ap("★[要確認] iSPEEDで適時開示(TOB/M&A/大量保有報告等)を確認するまで発注不可★")
    if not res["picked"]:
        ap("該当なし")
    for i, c in enumerate(res["picked"], 1):
        eng_label = "エンジンR(反転)" if c.engine == "R" else "エンジンM(モメンタム入口)"
        ap(f"◆{i}. {c.code} {c.name} [{eng_label}] {c.sector}")
        ap(f"   トリガー: {c.trigger}")
        ap(f"   エントリー{c.entry:,.0f}円 / 初期ストップ{c.stop:,.0f}円 / 固定利確目標{c.target:,.0f}円"
           f"(約{cfg.profit_target_r:.1f}R)")
        ap(f"   リスク{c.risk_pct:.2f}%" + (f"(≈{c.shares}株)" if c.shares else "")
           + f" / リスク幅{c.risk_w/c.entry*100:.1f}% / 時間ストップ{cfg.time_stop_days}営業日")
        for fl in c.flags:
            ap(f"   ⚠{fl}")
        ap(f"   iSPEED確認: " + " / ".join(c.checks[:2]))
    ap("")

    if res.get("watch"):
        ap("【2週間スイング・参考層(本命の次点)】")
        for c in res["watch"]:
            eng_label = "R" if c.engine == "R" else "M"
            ap(f"・{c.code} {c.name}[{eng_label}] {c.sector} — 監視のみ・直接エントリーは規律違反")
        ap("")

    ap(f"【2週間スイング総リスク】新規計 {st.get('total_risk',0):.2f}% / 上限 {st.get('risk_cap',0):.1f}% | {pos_note}")
    ap("出口: 固定利確(約{:.1f}R)または初期ストップ、いずれか先に到達で手仕舞い。".format(cfg.profit_target_r)
       + f"{cfg.time_stop_days}営業日経過でも未決着なら時間ストップで機械的に手仕舞い(シャンデリア・トレーリングは使わない)。")
    ap("行動注意: エントリー時にpositions_swing2w.csvへentry_date(必須)・stop_price・target_priceを記録すること。"
       "記録が無いと時間ストップ判定ができない。")

    return L
