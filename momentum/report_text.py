"""モメンタム・スクリーニング — LINEテキストレポート."""

from __future__ import annotations


def build_text(today: str, meta: dict, regime: dict, res: dict, pos_note: str, cfg,
               position_alerts: list[dict] | None = None) -> str:
    L = []
    ap = L.append

    ap(f"◆stockbotTOM モメンタム・スクリーニング {today}")
    ap("")

    all_alerts = position_alerts or []
    notable = [a for a in all_alerts if a.get("state_c") or a.get("score_drop") or a.get("tob_jump") or a.get("state_c") is None]
    if notable:
        ap("【保有銘柄アラート】")
        for a in notable:
            if a.get("state_c") or (a.get("tob_jump") and a.get("tob_stage") == "confirmed"):
                tag = "⚠状態C" if a.get("state_c") else "⚠TOB疑い(要確認)"
            elif a.get("tob_jump") and a.get("tob_stage") == "day0":
                tag = "△急騰検知(参考)"
            elif a.get("score_drop"):
                tag = "△スコア劣化"
            else:
                tag = "⚠要確認"
            ap(f"・{a['code']} {a['name']}: {tag} — {a['note']}")
        ap("")

    ap(f"【レジーム】{regime.get('mode','-')}")
    ap(f"　{regime.get('detail', regime.get('reason',''))}")
    ap("")

    cov = meta.get("data_coverage", 0.0)
    ap(f"【データ】{meta.get('data_ok',0)}/{meta.get('data_total',0)} ({cov*100:.0f}%) "
       f"出典:{meta.get('source','?')} {meta.get('fetched_at','')}")
    ap("※単一ソース(yfinance)。本命は全件仮点灯、確定はiSPEED照合+チャット側確認後")
    if meta.get("data_warn"):
        ap(f"⚠データ被覆率<{cfg.data_coverage_min*100:.0f}% → プール規模・ランキングの精度に影響の可能性")
    ap("")

    st = res["stats"]
    ap(f"【候補プール】{st.get('universe_considered',0)}銘柄を検討 → 上位{st.get('pool_size',0)}銘柄をプール化"
       + (f"(TOB/コーポレートアクション疑い{st['tob_excluded']}件を事前除外)" if st.get('tob_excluded') else ""))
    sc = st.get("state_count", {})
    ap(f"状態内訳: A(継続中の押し目){sc.get('A',0)}件 / B(初動ブレイク){sc.get('B',0)}件 / "
       f"C(流出・新規対象外){sc.get('C',0)}件")
    ap(f"点灯{st.get('fired',0)}件 → 実保有上限{cfg.max_positions}銘柄・セクター1業種まで の制約で"
       f"アクション候補{st.get('picked',0)}件")
    if st.get("top_sectors"):
        sec_txt = " / ".join(f"{s}{n}銘柄" for s, n in st["top_sectors"])
        ap(f"(参考・指示⑨診断) プール業種上位{cfg.sector_diag_top_n}: {sec_txt}")
    ap("")

    ap(f"【アクション候補(仮点灯・要iSPEED確認)】")
    if res["picked"] or res.get("watch"):
        ap("★[要確認] iSPEEDで適時開示(TOB/M&A/大量保有報告等)を確認するまで発注不可★")
    if st.get("regime_caution") and res["picked"]:
        ap(f"⚠相場全体が防御モード: {regime.get('detail','')}")
        ap("　個別シグナルの期待値はレジーム条件付き(Hanauer 2014等)。銘柄選定は通常どおりだが、通常より慎重に。")
    if not res["picked"]:
        reason = "該当なし"
        ap(f"該当なし — {reason}")
    for i, c in enumerate(res["picked"], 1):
        ap(f"◆{i}. {c.code} {c.name} [状態{c.state}] {c.sector}")
        ap(f"   エントリー{c.entry:,.0f}円 / 初期ストップ{c.stop:,.0f}円 / "
           f"シャンデリア水準{c.chandelier:,.0f}円")
        ap(f"   リスク{c.risk_pct:.2f}%" + (f"(≈{c.shares}株)" if c.shares else "")
           + f" / リスク幅{c.risk_w/c.entry*100:.1f}%")
        for fl in c.flags:
            ap(f"   ⚠{fl}")
        ap(f"   iSPEED確認: " + " / ".join(c.checks[:2]))
    ap("")

    if res.get("watch"):
        ap("【参考層(本命の次点・直接エントリーは規律違反)】")
        for c in res["watch"]:
            ap(f"・{c.code} {c.name}[状態{c.state}] {c.sector} スコア{c.score:.1f} "
               f"— 昇格には3銘柄枠が空くか同業種枠が空くことが必要")
        ap("")

    ap(f"【総リスク】新規計 {st['total_risk']:.2f}% / 上限 {st['risk_cap']:.1f}% | {pos_note}")
    ap("")

    ap("【出口方針】固定利確なし。シャンデリア水準(直近22日高値-3×ATR)を割ったら手仕舞い。"
       "初期ストップは初日のみのR定義。トレンドが崩れるまで持ち続ける設計。")
    ap("【行動注意】候補が出た日に必ず取引する必要はない。レジームが防御モードの日は無理をしない。")
    ap("【免責】AIの候補提示で投資助言ではない。モメンタムクラッシュ(急な逆転)のリスクは"
       "レジームフィルターとロングオンリー運用でも完全には消えない。"
       "最終判断と結果責任はユーザーにある。")
    return "\n".join(L)
