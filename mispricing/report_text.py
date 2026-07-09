"""LINEテキストレポート(PNG失敗時のフォールバック兼ログ) — v5.0出力順序準拠(13セクション)."""

from __future__ import annotations


def _flow_line(rows) -> str:
    return " / ".join(f"{r['sector']}({r['ret5']:+.1f}%,5d)" for r in rows)


def _position_lines(positions: list[dict]) -> list[str]:
    if not positions:
        return []
    L = ["【保有ポジション評価(日次再評価)】"]
    for p in positions:
        if p.get("error"):
            L.append(f"・{p['code']} {p['name']}: ⚠{p['error']}")
            continue
        pnl = f"{p['pnl_pct']:+.1f}%" if "pnl_pct" in p else "建値未登録"
        L.append(f"・{p['code']} {p['name']}[{p['direction']}] 現在値{p['close']:,.0f}円"
                 + (f"(建値{p['entry_price']:,.0f}円 含み損益{pnl})" if "entry_price" in p else ""))
        L.append(f"   本日の構造的ストップ参考値: {p['today_stop']:,.0f}円"
                 f"(25日線{p['today_sma25']:,.0f}円 / RSI{p['rsi14']:.0f} / 25日乖離{p['rel_dev_pct']:+.1f}%)")
        c = p.get("candidate")
        if c:
            L.append(f"   {p['status_note']} → 新規評価ならIN{c.entry:,.0f}/STOP{c.stop:,.0f}/"
                     f"TP1{c.tp1:,.0f}/2R{c.ref2r:,.0f}円")
        else:
            L.append(f"   {p['status_note']}")
        if p.get("liquidity_warn"):
            L.append(f"   ⚠{p['liquidity_warn']}")
    L.append("(価格・%ベースの参考表示。R倍数は当初stop_price未登録のため非表示)")
    L.append("")
    return L


def build_text(today: str, meta: dict, macro: dict, flow: dict, res: dict,
               pos_note: str, events: list[str], backfilled: int, cfg,
               positions: list[dict] | None = None) -> str:
    L = []
    ap = L.append

    ap(f"◆stockbotTOM 歪み×資金循環スクリーニング v5.0-bot {today}")
    ap("")

    L.extend(_position_lines(positions or []))

    # 1. データ取得状況
    cov = meta.get("data_coverage", 0.0)
    ap(f"【1.データ】{meta.get('data_ok',0)}/{meta.get('data_total',0)} ({cov*100:.0f}%) "
       f"出典:{meta.get('source','?')} {meta.get('fetched_at','')}")
    ap("※算出指標は単一ソース(yfinance)を許容(確信度減点なし)。ただし「本命」はいずれも仮点灯(未確認)"
       "であり、確定候補への昇格はiSPEED照合+チャット側ゲート0/3経由")
    if meta.get("data_warn"):
        ap(f"⚠データ被覆率<{cfg.data_coverage_min*100:.0f}% → 本日は新規見送り推奨")
    if backfilled:
        ap(f"棄却台帳: {backfilled}件にN営業日後リターンを自動追記")
    ap("")

    # 2. 地合い
    vi_str = "" if macro["vi"] is None else "={:.1f}".format(macro["vi"])
    ap(f"【2.地合い】スコア {macro['score']}/5 | {macro['vi_label']}{vi_str}")
    for p in macro["parts"]:
        ap(f"・{p}")
    ap(f"→ ロット指示: {macro['lot_text']} / 総オープンリスク上限 {macro['risk_cap']:.1f}%")
    if macro["provisional"]:
        ap(f"(暫定: {','.join(macro['provisional'])} 欠落)")
    ap("")

    # 3. 資金循環マップ(本命ゼロの日も必ず出す)
    ap(f"【3.資金循環マップ】レジーム: {flow.get('regime','不明')}"
       + (f"(値上がり業種比率{flow['up_share']:.0f}%)" if flow.get("ok") else ""))
    if flow.get("ok"):
        ap("流入上位3: " + " / ".join(
            f"{r['sector']}({r['ret5']:+.1f}%,5d/段階{flow['sector_stage'].get(r['sector'],'不明')})"
            for r in flow["inflow"]))
        ap("流出上位3: " + " / ".join(
            f"{r['sector']}({r['ret5']:+.1f}%,5d/段階{flow['sector_stage'].get(r['sector'],'不明')})"
            for r in flow["outflow"]))
        ap(f"({flow.get('note','')})")
    else:
        ap(f"作成不可: {flow.get('note','データ不足')}")
    ap("")

    # 4. ユーザー確認依頼リスト
    st = res["stats"]
    ap(f"【4.ユーザー確認依頼リスト】仮点灯 本命候補{len(res['picked'])}件・参考層{len(res['watch'])}件"
       " — 発注前に必ずiSPEEDで照合")
    for c in (res["picked"] + res["watch"]):
        ap(f"・{c.code} {c.name}: " + " / ".join(c.checks[:2]))
    ap("")

    # 5. 歪みハンティング概況(エンジン別)
    eng = st["by_engine"]
    ap(f"【5.エンジン別概況】A(業種内リバーサル):{eng.get('A',0)}件 / "
       f"S(逆流戻り売り):{eng.get('S',0)}件 / B疑い(PEAD):{eng.get('B疑い',0)}件")
    ap("")

    # 6. 棄却率サマリー
    ap(f"【6.棄却率】検討{st['considered']}→棄却{st['rejected']}→本命{st['picked']}(参考層{st['watch']})")
    if st["by_stage"]:
        ap("内訳: " + " / ".join(f"{k}:{v}" for k, v in st["by_stage"].items()))
    ap("")

    # 7. 本命候補(4点セット構造)
    ap(f"【7.本命候補(仮点灯・要iSPEED確認)】※確定候補は0件(パス1のため)")
    if not res["picked"]:
        ap("該当なし — ゼロ件はゼロ件。無理に格下げ採用しない")
    for i, c in enumerate(res["picked"], 1):
        ap(f"◆{i}. {c.code} {c.name} [{c.direction}/エンジン{c.engine}/確信度{c.conf}] {c.sector}")
        ap(f"①歪みの構造: {c.nonfund}")
        ap(f"   {c.trigger_text}")
        ap(f"②資金フロー: {c.ftag}タグ / セクター段階{c.sec_stage} / レジーム{flow.get('regime','不明')}")
        ap(f"③IN {c.entry:,.0f} / STOP {c.stop:,.0f} / TP1(+1R) {c.tp1:,.0f} / 2R参照 {c.ref2r:,.0f}円")
        ap(f"   計画R=2.0(ブレンド参考+0.5R/+1.5R) 概算ネットR≈{c.net2r:.2f} "
           f"リスク{c.risk_pct:.2f}%" + (f"(≈{c.shares}株)" if c.shares else ""))
        ap(f"④保有≤{c.hold_days}営業日(時間ストップ) 失効{c.expiry_days}日 到達≈{c.reach_days:.0f}ATR日 "
           "寄りギャップ±リスク幅50%で計画失効")
        for fl in c.flags:
            ap(f"   ⚠{fl}")
        ap(f"   iSPEED確認: " + " / ".join(c.checks[:3]))
    ap("")

    # 8. 参考監視層
    if res["watch"]:
        ap("【8.参考監視層(本命関連・直接エントリー規律違反)】")
        for c in res["watch"]:
            ap(f"・{c.code} {c.name}({c.direction}/{c.engine}/{c.watch_reason}) "
               f"順流逆流={c.ftag} — 昇格には後日エンジン点灯+4ゲート通過が必要")
        ap("")

    # 9. ポートフォリオ
    ap(f"【9.総リスク】新規計 {st['total_risk']:.2f}% / 上限 {st['risk_cap']:.1f}% | {pos_note}")
    if res["picked"]:
        engines_used = {c.engine for c in res["picked"]}
        dirs_used = {c.direction for c in res["picked"]}
        ap(f"エンジン分布: {engines_used} / 方向分布: {dirs_used}(偏りは目視確認)")
    ap("")

    # 10. 構造化ログ
    ap("【10.構造化ログ】plan_log/result_log_template/reject_ledger をCSV出力済み(添付/リポジトリ参照)")
    ap("")

    # 11. 今週イベント
    if events:
        ap("【11.今週の重要イベント】" + " / ".join(events))
        ap("")

    # 12-13. 行動注意・免責
    ap("【12.行動注意】候補が出た日に必ず取引する必要はない。ゼロ件の日に無理な格下げ採用をしない。")
    ap("ゲート0(スティールマン)/ゲート3(反証・プレモータム)は未実施 → チャット側パス2で実施のこと。")
    ap("IN/STOP等は前日確定値ベースの目安。寄り後に必ず再計算。逆指値は発注と同時にセット(IFD-OCO)。")
    ap("【13.免責】AIによる候補提示であり投資助言ではない。数値誤りの可能性あり。市場が正しい可能性が常にある。"
       "利益には約20%課税。最終判断と結果責任はユーザーにある。")
    return "\n".join(L)
