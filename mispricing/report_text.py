"""LINEテキストレポート(PNG失敗時のフォールバック兼ログ) — v4.1出力順序準拠."""

from __future__ import annotations


def build_text(today: str, meta: dict, macro: dict, res: dict,
               pos_note: str, events: list[str], backfilled: int, cfg) -> str:
    L = []
    ap = L.append

    ap(f"◆stockbotTOM 歪みスクリーニング v4.1-bot {today}")
    ap("")

    # 1. データ取得状況
    cov = meta.get("data_coverage", 0.0)
    ap(f"【データ】{meta.get('data_ok',0)}/{meta.get('data_total',0)} ({cov*100:.0f}%) "
       f"出典:{meta.get('source','?')} {meta.get('fetched_at','')}")
    ap("※全指標は単一ソース算出=未確認扱い。確定にはiSPEED照合(独立2ソース化)が必要")
    if meta.get("data_warn"):
        ap(f"⚠データ被覆率<{cfg.data_coverage_min*100:.0f}% → 本日は新規見送り推奨")
    if backfilled:
        ap(f"棄却台帳: {backfilled}件にN営業日後リターンを自動追記")
    ap("")

    # 2. 地合い
    vi_str = "" if macro["vi"] is None else "={:.1f}".format(macro["vi"])
    ap(f"【地合い】スコア {macro['score']}/5 | {macro['vi_label']}{vi_str}")
    for p in macro["parts"]:
        ap(f"・{p}")
    ap(f"→ ロット指示: {macro['lot_text']} / 総オープンリスク上限 {macro['risk_cap']:.1f}%")
    if macro["provisional"]:
        ap(f"(暫定: {','.join(macro['provisional'])} 欠落)")
    ap("")

    # 4-5. 概況と棄却率
    st = res["stats"]
    ap(f"【概況】検討{st['considered']}→棄却{st['rejected']}→仮点灯{st['picked']}"
       f"(次点{len(res['runners'])})")
    if st["by_stage"]:
        ap("棄却内訳: " + " / ".join(f"{k}:{v}" for k, v in st["by_stage"].items()))
    ap("")

    # 6'. 仮点灯候補(本日の候補には未昇格)
    ap("【仮点灯候補(要ユーザー確認)】※本日の確定候補は0件(パス1のため)")
    if not res["picked"]:
        ap("該当なし — ゼロ件はゼロ件。無理に格下げ採用しない")
    for i, c in enumerate(res["picked"], 1):
        ap(f"{i}. {c.code} {c.name} [{c.direction}/{c.mtype}/{c.conf}] {c.sector}")
        ap(f"   {c.trigger_text}")
        ap(f"   IN {c.entry:,.0f} / STOP {c.stop:,.0f} / TP1 {c.tp1:,.0f} / 2R参照 {c.ref2r:,.0f}円")
        ap(f"   リスク{c.risk_pct:.2f}%"
           + (f"(≈{c.shares}株)" if c.shares else "")
           + f" 保有≤{c.hold_days}日 失効{c.expiry_days}日 到達≈{c.reach_days:.0f}ATR日")
        for fl in c.flags:
            ap(f"   ⚠{fl}")
        ap(f"   iSPEED確認: " + " / ".join(c.checks[:3]))
    ap("")

    # 7. ポートフォリオ
    ap(f"【総リスク】新規計 {st['total_risk']:.2f}% / 上限 {st['risk_cap']:.1f}% | {pos_note}")
    ap("")

    # 9. 次点
    if res["runners"]:
        ap("【次点】" + " / ".join(
            f"{c.code}{c.name}({c.direction[0]}·{c.conf})" for c in res["runners"]))
        ap("")

    # 10. 今週イベント
    if events:
        ap("【今週】" + " / ".join(events))
        ap("")

    # 11-12. 行動注意・免責
    ap("【注意】候補が出た日に必ず取引する必要はない。ゼロ件の日に無理な格下げ採用をしない。")
    ap("ゲート0(スティールマン)/ゲート3(反証・プレモータム)は未実施 → チャット側パス2で実施のこと。")
    ap("IN/STOP等は前日確定値ベースの目安。寄り後に必ず再計算。逆指値は発注と同時にセット(IFD-OCO)。")
    ap("寄りが計画INからリスク幅50%以上乖離 → その計画は失効。")
    ap("免責: AIによる候補提示であり投資助言ではない。数値誤りの可能性あり。市場が正しい可能性が常にある。"
       "利益には約20%課税。最終判断と結果責任はユーザーにある。")
    return "\n".join(L)
