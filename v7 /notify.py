"""v7.1 第9部 — 出力。

本ロジックの出力は「銘柄と資格の内訳のみ」(§9.4)。
IN/STOP/TP価格はエントリー層で算出する。スクリーニング仕様に価格を含めない
ことで責務分離を保つ。

★v6との最大の違い: 価格・株数・リスク金額を一切出さない。
"""

from __future__ import annotations

from .pipeline import DIM_NAMES
from .gates import RISK_ON, NEUTRAL, RISK_OFF


def build_notification(res: dict, today: str, cfg) -> str:
    L = []
    ap = L.append
    reg = res["regime"]
    counts = res["counts"]
    asof = res.get("asof", {})

    ap(f"◆stockbotTOM v7.1 次元抽出型スクリーニング {today}")
    ap("")

    # ---- レジーム ----
    mark = {RISK_ON: "🟢", NEUTRAL: "🟡", RISK_OFF: "🔴"}.get(reg["state"], "")
    ap(f"【レジーム】{mark} {reg['state']}")
    ap(f"　{reg['reason']}")
    if res["dd"].get("note"):
        ap(f"　L0-B: {res['dd']['note']}")

    if res.get("halted"):
        ap("")
        ap("【候補】なし")
        ap("　RISK_OFFのためL1以降を実行していない。")
        ap("　候補ゼロは規律が機能している証拠であり、バグではない(§5.1.6)。")
        _footer(ap, asof, cfg)
        return "\n".join(L)

    # ---- 絞り込み ----
    dims = res.get("enabled_dims", [])
    ap("")
    ap(f"【絞り込み】{counts['universe']} → L1流動性{counts['l1']} → 採点{counts['scored']}"
       f" → 本命{counts['main']} / 監視{counts['watch']} / 除外{counts['excluded']}")
    ap(f"　有効次元 {len(dims)}個: " + " ".join(DIM_NAMES[d] for d in dims))
    ap(f"　θ={cfg.theta:.2f} / 本命≧{cfg.coverage_main:.0%} / 監視≧{cfg.coverage_watch:.0%}")

    # ---- 本命 ----
    ap("")
    if not res["main"]:
        ap("【本命】なし")
    else:
        ap(f"【本命】{len(res['main'])}件")
        for i, c in enumerate(res["main"], 1):
            ap("")
            ap(f"─────────────")
            ap(f"{i}. {c.code} {c.name}  DCS {c.dcs:.1f}  カバレッジ {c.coverage:.0%}")
            ap(f"   {c.sector}")
            ap("   " + _dim_line(c, dims))
            if c.missing:
                ap(f"   欠落: {' '.join(DIM_NAMES[d] for d in c.missing)}")
            for n in c.notes[:2]:
                ap(f"   ※{n}")

    # ---- 予備軍(§9.3) ----
    ap("")
    if res["watch"]:
        ap(f"【予備軍(監視)】{len(res['watch'])}件 — 発注対象ではない")
        for c in res["watch"]:
            miss = " ".join(DIM_NAMES[d] for d in c.missing)
            extra = ""
            if c.maturity_weeks is not None:
                extra = f" / 成熟まで推定{c.maturity_weeks}週"
            ap(f"・{c.code} {c.name} DCS{c.dcs:.0f} 欠落[{miss}]{extra}")
    else:
        ap("【予備軍(監視)】なし")

    _footer(ap, asof, cfg)
    return "\n".join(L)


def _dim_line(c, dims) -> str:
    out = []
    for d in dims:
        v = c.dims.get(d)
        s = "—" if v is None else f"{v:.2f}"
        mark = "✓" if d in c.satisfied else " "
        out.append(f"{DIM_NAMES[d]}{s}{mark}")
    return " ".join(out)


def _footer(ap, asof: dict, cfg) -> None:
    # §6.2 規則5: 基準時点を明記する
    ap("")
    ap("【基準時点】")
    ap(f"　DCS基準日 {asof.get('dcs') or '—'}（t−{cfg.align_lag_days}営業日に整合）")
    ap(f"　価格基準日 {asof.get('price') or '—'}")
    ap(f"　レジーム基準日 {asof.get('regime') or '—'}")

    # データ制約の恒久表示(§10.2.3)
    ap("")
    ap("【データ制約】yfinance運用のため以下は満たせていない")
    for lim in cfg.data_limitations:
        ap(f"　・{lim}")

    ap("")
    ap("【留意】本仕様は検証0%の設計案である。候補に挙がったことは「今買え」を意味しない。")
    ap("　エントリータイミング・価格・サイズは本ロジックの範囲外(§0.1)。")
    ap("　投資助言ではなく、最終判断と結果の責任は運用者にある。")
