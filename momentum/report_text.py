"""モメンタム・スクリーニング — LINEテキストレポート."""

from __future__ import annotations


def build_text(today: str, meta: dict, regime: dict, res: dict, pos_note: str, cfg) -> str:
    L = []
    ap = L.append

    ap(f"◆stockbotTOM モメンタム・スクリーニング {today}")
    ap("")

    ap(f"【レジーム】{regime.get('mode','-')}")
    ap(f"　{regime.get('detail', regime.get('reason',''))}")
    if not regime.get("attack", False):
        ap("　→ 新規エントリー全停止(例外なし)。既存はトレールのみで管理。")
    ap("")

    cov = meta.get("data_coverage", 0.0)
    ap(f"【データ】{meta.get('data_ok',0)}/{meta.get('data_total',0)} ({cov*100:.0f}%) "
       f"出典:{meta.get('source','?')} {meta.get('fetched_at','')}")
    ap("※単一ソース(yfinance)。本命は全件仮点灯、確定はiSPEED照合+チャット側確認後")
    ap("")

    st = res["stats"]
    ap(f"【候補プール】{st.get('universe_considered',0)}銘柄を検討 → 上位{st.get('pool_size',0)}銘柄をプール化")
    sc = st.get("state_count", {})
    ap(f"状態内訳: A(継続中の押し目){sc.get('A',0)}件 / B(初動ブレイク){sc.get('B',0)}件 / "
       f"C(流出・新規対象外){sc.get('C',0)}件")
    ap(f"点灯{st.get('fired',0)}件 → 実保有上限{cfg.max_positions}銘柄・セクター1業種まで の制約で"
       f"アクション候補{st.get('picked',0)}件")
    ap("")

    ap(f"【アクション候補(仮点灯・要iSPEED確認)】")
    if not res["picked"]:
        reason = st.get("blocked_reason") or "該当なし"
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

    ap(f"【総リスク】新規計 {st['total_risk']:.2f}% / 上限 {st['risk_cap']:.1f}% | {pos_note}")
    ap("")

    ap("【出口方針】固定利確なし。シャンデリア水準(直近22日高値-3×ATR)を割ったら手仕舞い。"
       "初期ストップは初日のみのR定義。トレンドが崩れるまで持ち続ける設計。")
    ap("【行動注意】候補が出た日に必ず取引する必要はない。レジームが防御モードの日は無理をしない。")
    ap("【免責】AIの候補提示で投資助言ではない。モメンタムクラッシュ(急な逆転)のリスクは"
       "レジームフィルターとロングオンリー運用でも完全には消えない。"
       "最終判断と結果責任はユーザーにある。")
    return "\n".join(L)
