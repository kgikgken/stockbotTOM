"""v6 通知フォーマット — 発注可能な状態まで計算を済ませて渡す。

v6.0最大の変更: リスク金額まで通知に含める。人間側に残す判断は
「ニュース確認」と「最終的なGo/No-Go」のみ(三澤の10分制約)。
"""

from __future__ import annotations

from .filters import STAGE_D_NAMES

STAGE_W_NAMES = {1: "底固め", 2: "上昇期", 3: "天井圏", 4: "下降期"}


def build_notification(res: dict, exits: list, today: str, cfg) -> str:
    L = []
    ap = L.append
    picked = res["picked"]
    c = res["counts"]

    ap(f"◆stockbotTOM v6 {today}")
    ap("")

    # ---- ファネル(各レイヤーの通過数) ----
    ap(f"【絞り込み】{c['universe']} → 流動性{c['layer0']} → 週足St2 {c['layer1']}"
       f" → TT+RS {c['layer2']} → 日足St {c['layer3']}"
       f" → 点灯{c['layer4']} → 通知{c['layer5']}")

    # ★点灯したのに通知に至らなかった銘柄の理由を集計して出す。
    # これが無いと「点灯13→通知3」の間で何が起きたのか運用者から見えず、
    # 設計上の不具合(例: リスク上限で特定セットアップが全滅)に気づけない。
    bs = c.get("by_setup") or {}
    if bs:
        ap("　点灯内訳: " + " / ".join(f"{k}{v}" for k, v in bs.items()))
    drops = res.get("drops") or []
    if drops:
        by_reason = {}
        for dr in drops:
            key = dr.get("layer", "不明")
            by_reason[key] = by_reason.get(key, 0) + 1
        ap("　脱落: " + " / ".join(f"{k} {v}件" for k, v in sorted(by_reason.items())))

    # ---- 資金状態 ----
    dd = res["dd"]
    ap(f"【資金】ヒート{res['heat']*100:.1f}%(上限{cfg.portfolio_heat_max*100:.0f}%)"
       f" / DD {dd['dd']*100:.1f}%")
    if dd.get("note"):
        ap(f"　⚠{dd['note']}")
    if not res["can_new"]:
        ap(f"　🛑 新規停止: {res['gate_reason']}")
    ap("")

    # ---- 新規候補 ----
    if not picked:
        ap("【本日の新規候補】なし")
        ap("　条件を満たす銘柄がなかった。0件は正常な結果であり、無理に基準を緩めない。")
    else:
        ap(f"【本日の新規候補】{len(picked)}件")
        for i, x in enumerate(picked, 1):
            ap("")
            ap(f"─────────────")
            ap(f"【{x.code} {x.name}】{x.setup}  score {x.score}")
            ap(f"週足: St{x.stage_w}({STAGE_W_NAMES.get(x.stage_w,'')}) "
               f"30週線{x.slope_w*100:+.1f}%   RS: {x.rs_rating:.0f}")
            dtl = ""
            if x.setup == "VCP":
                d = x.detail
                dtl = f"   収縮: {d['n_contractions']}回 → {d['final_depth']*100:.1f}%"
            elif x.setup == "押し目":
                dtl = f"   押し: {x.detail['depth']*100:.1f}% / {x.detail['days_since_high']}日"
            elif x.setup == "ピボット":
                dtl = f"   上抜け: +{x.detail['extension']*100:.1f}%"
            ap(f"日足: St{x.stage_d}(滞在{x.stage_days}日){dtl}")
            ap("")
            ap(f"{x.order_type:6} {x.entry:,.0f}円")
            ap(f"ストップ {x.stop:,.0f}円（{-x.stop_pct*100:.1f}% / {x.stop_kind}）")
            ap(f"株数    {x.shares:,}株   リスク {x.risk_amount:,.0f}円"
               f"（口座の{x.risk_pct*100:.1f}%）")
            ap(f"利確目安 +2R = {x.target_2r:,.0f}円")
            ap(f"注文期限 {x.expire_date}まで（{cfg.order_expire_days}営業日）")
            for f in x.flags:
                ap(f"⚠ {f}")
            if not x.tradable:
                ap("🚫 発注不可（上記の理由による）")
            ap("⚠ ニュース内容は未確認。発注前に自分の目で確認すること")

    # ---- 保有銘柄シグナル(件数上限なし) ----
    ap("")
    if exits:
        ap(f"【保有銘柄】{len(exits)}件")
        for e in exits:
            top = e.get("top")
            if top:
                ap(f"・{e['code']} {e['name']}: [優先{top[0]}] {top[1]} — {top[2]}")
            else:
                ap(f"・{e['code']} {e['name']}: 該当なし"
                   f"（{e['metrics'].get('r', 0):+.2f}R 保有{e['metrics'].get('days_held','?')}日）")
    else:
        ap("【保有銘柄】なし")

    ap("")
    ap("【免責】テクニカル分析に基づく抽出であり投資助言ではない。"
       "通知は「調べる価値があるかもしれない」という意味に留まる。"
       "発注可否の最終判断と結果の責任は運用者にある。")
    return "\n".join(L)
