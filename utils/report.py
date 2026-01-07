# ============================================
# utils/report.py
# LINE出力フォーマット（日本語で分かりやすく）
# ============================================

from __future__ import annotations

from typing import List, Tuple
import math

from utils.util import round_price
from utils.screener import ScreeningResult


def _fmt_sector_list(sectors: List[Tuple[str, float]]) -> str:
    if not sectors:
        return "（データなし）"
    lines = []
    for i, (name, ret) in enumerate(sectors, 1):
        lines.append(f"{i}. {name} ({ret:+.2f}%)")
    return "\n".join(lines)


def build_report_text(r: ScreeningResult) -> str:
    # ヘッダ
    trade_line = "✅ 新規可（条件クリア）" if not r.no_trade else f"🚫 新規見送り（{r.no_trade_reason}）"
    macro_on = "ON" if r.macro_risk else "OFF"
    lev_text = f"{r.leverage:.1f}倍"
    max_pos = int(round(r.max_exposure_jpy, -4))

    # 候補まとめ
    n = len(r.picks)
    if n > 0:
        avg_rr = sum([x["rr"] for x in r.picks]) / n
        avg_ev = sum([x["ev"] for x in r.picks]) / n
        avg_adj = sum([x["adj_ev"] for x in r.picks]) / n
        avg_rday = sum([x["r_day"] for x in r.picks]) / n
    else:
        avg_rr = avg_ev = avg_adj = avg_rday = 0.0

    lines = []
    lines.append(f"📅 {r.today} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")
    lines.append(trade_line)
    lines.append(f"- 地合い: {r.market_score:.0f}点（{('強い' if r.market_score>=70 else 'やや強い' if r.market_score>=60 else '中立' if r.market_score>=45 else '弱い')}）")
    lines.append(f"- 地合い変化: Δ{r.delta_3d:+.0f}")
    lines.append(f"- 相場判断: {r.regime}")
    lines.append(f"- 週次新規回数: {r.weekly_new_count} / 3")
    lines.append(f"- マクロ警戒: {macro_on}")
    lines.append(f"- 推奨レバレッジ: {lev_text}（{'守り（新規禁止）' if r.no_trade else '運用'}）")
    lines.append(f"- 最大建玉目安: 約{max_pos:,}円")
    lines.append("")
    lines.append("📈 セクター動向（直近5日）")
    lines.append(_fmt_sector_list(r.sectors_top5))
    lines.append("")
    lines.append("⚠ 重要イベント")
    lines.append(r.nearest_event_text)
    lines.append("")

    lines.append("🏆 Swing候補（順張りのみ / 追いかけ禁止 / 速度重視）")
    lines.append(f"  候補数:{n}銘柄  平均RR:{avg_rr:.2f} / 平均EV:{avg_ev:.2f} / 平均補正EV:{avg_adj:.2f} / 平均R/日:{avg_rday:.2f}")
    if r.macro_risk:
        lines.append("※ イベント接近のため、候補は最大2銘柄までに制限")
    lines.append("")

    # 本命
    for x in r.picks:
        ticker = x["ticker"]
        sector = x["sector"]
        setup_type = x["setup_type"]
        rr = x["rr"]
        adj = x["adj_ev"]
        rday = x["r_day"]
        entry = x["entry"]
        lo = x["band_low"]
        hi = x["band_high"]
        close = x["close"]
        atr = x["atr"]
        gu = "あり" if x["gu"] else "なし"
        stop = x["stop"]
        tp1 = x["tp1"]
        tp2 = x["tp2"]
        days = x["days"]
        action = x["action"]

        lines.append(f"- {ticker} [{sector}]")
        lines.append(f"  型:{setup_type}  RR:{rr:.2f}  補正EV:{adj:.2f}  R/日:{rday:.2f}")
        lines.append(
            f"  エントリー:{round_price(entry,1)}（範囲:{round_price(lo,1)}〜{round_price(hi,1)}） 現在値:{round_price(close,1)}  ATR:{round_price(atr,1)}  GU:{gu}"
        )
        lines.append(
            f"  損切:{round_price(stop,1)}  利確1:{round_price(tp1,1)}  利確2:{round_price(tp2,1)}  想定日数:{days:.1f}"
        )
        lines.append(f"  行動:{'今日は監視' if r.no_trade else action}")
        lines.append("")

    # 監視（必要なら残す：ユーザー方針でcで詰めた後a）
    if r.watch:
        lines.append("🧠 監視リスト（今日は入らない）")
        for w in r.watch[:10]:
            t = w.get("ticker", "")
            sec = w.get("sector", "不明")
            reason = w.get("drop_reason", "理由なし")
            lines.append(f"- {t} [{sec}] 理由:{reason}")
        lines.append("")

    # ロット事故警告
    if r.lot_risk_text:
        lines.append(r.lot_risk_text)
        lines.append("")

    lines.append("📊 保有ポジション")
    lines.append(r.positions_text or "n/a")

    return "\n".join(lines)