from __future__ import annotations

from datetime import date
from typing import Dict, List, Tuple
import numpy as np


def _fmt_money(yen: int) -> str:
    return f"{yen:,}円"


def _fmt_float(x: float, nd: int = 2) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def build_daily_report(
    today_str: str,
    today_date: date,
    market: Dict,
    macro_caution: bool,
    weekly_new: int,
    total_asset: float,
    sector_rank: List[Tuple[str, float]],
    event_lines: List[str],
    swing: Dict,
    pos_text: str,
) -> str:
    m = int(market.get("score", 50))
    d3 = int(market.get("delta3d", 0))
    comment = str(market.get("comment", "中立"))
    phase = str(market.get("phase", "不安定"))

    no_trade = bool(swing.get("no_trade", False))
    reasons = swing.get("no_trade_reasons", []) or []
    lev = float(swing.get("lev", 1.0))
    lev_reason = str(swing.get("lev_reason", ""))

    max_pos = int(round(total_asset * lev)) if (np.isfinite(total_asset) and total_asset > 0 and lev > 0) else 0

    lines: List[str] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")

    if no_trade:
        if reasons:
            lines.append(f"🚫 新規見送り（{'・'.join(reasons)}）")
        else:
            lines.append("🚫 新規見送り（条件該当）")
    else:
        lines.append("✅ 新規可（条件クリア）")

    lines.append(f"- 地合い: {m}点（{comment}）")
    lines.append(f"- 地合い変化: Δ{d3:+d}")
    lines.append(f"- 相場判断: {phase}")
    lines.append(f"- 週次新規回数: {weekly_new} / 3")
    lines.append(f"- マクロ警戒: {'ON' if macro_caution else 'OFF'}")
    lines.append(f"- 推奨レバレッジ: {lev:.1f}倍（{lev_reason}）")
    lines.append(f"- 最大建玉目安: 約{_fmt_money(max_pos)}")
    lines.append("")

    lines.append("📈 セクター動向（直近5日）")
    if sector_rank:
        for i, (s, pct) in enumerate(sector_rank[:5]):
            lines.append(f"{i+1}. {s} ({pct:+.2f}%)")
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ 重要イベント")
    for ev in event_lines:
        lines.append(ev)
    lines.append("")

    # Swing
    lines.append("🏆 Swing候補（順張りのみ / 追いかけ禁止 / 速度重視）")
    summ = swing.get("summary", {}) or {}
    cnt = int(summ.get("count", 0))
    lines.append(
        f"  候補数:{cnt}銘柄"
        f"  平均RR:{_fmt_float(float(summ.get('avg_rr', 0.0)),2)}"
        f" / 平均EV:{_fmt_float(float(summ.get('avg_ev', 0.0)),2)}"
        f" / 平均補正EV:{_fmt_float(float(summ.get('avg_adjev', 0.0)),2)}"
        f" / 平均R/日:{_fmt_float(float(summ.get('avg_rday', 0.0)),2)}"
    )

    if bool(macro_caution):
        lines.append("※ イベント接近のため、候補は最大2銘柄までに制限")

    lines.append("")

    cands = swing.get("candidates", []) or []
    if not cands:
        lines.append("- 該当なし")
        lines.append("")
    else:
        for c in cands:
            gu = "あり" if c.get("gu") else "なし"
            lines.append(f"- {c['ticker']} {c['name']}［{c['sector']}］")
            lines.append(
                f"  型:{c['setup']}  RR:{_fmt_float(c['rr'],2)}  補正EV:{_fmt_float(c['adjev'],2)}  R/日:{_fmt_float(c['rday'],2)}  特性:{c.get('macro','other')}"
            )
            lines.append(
                f"  エントリー:{_fmt_float(c['entry'],1)}（範囲:{_fmt_float(c['entry_low'],1)}〜{_fmt_float(c['entry_high'],1)}） 現在値:{_fmt_float(c['price_now'],1)}  ATR:{_fmt_float(c['atr'],1)}  GU:{gu}"
            )
            lines.append(
                f"  損切:{_fmt_float(c['stop'],1)}  利確1:{_fmt_float(c['tp1'],1)}  利確2:{_fmt_float(c['tp2'],1)}  想定日数:{_fmt_float(c['exp_days'],1)}"
            )
            lines.append(f"  行動:{c['action']}")
            lines.append("")

    # Watchlist
    lines.append("🧠 監視リスト（今日は入らない）")
    watch = swing.get("watchlist", []) or []
    if not watch:
        lines.append("- なし")
    else:
        for w in watch[:10]:
            t = w.get("ticker", "")
            n = w.get("name", "")
            sec = w.get("sector", "")
            rsn = w.get("reason", "")
            if n:
                lines.append(f"- {t} {n}［{sec}］ 理由:{rsn}")
            else:
                lines.append(f"- {t} 理由:{rsn}")
    lines.append("")

    # ロット事故警告（position側で出す想定だが、ここは空出力しない）
    # → analyze_positions が警告文を含む場合だけ出す
    if pos_text and "⚠ ロット事故警告" in pos_text:
        # pos_text内に警告があるなら reportにもそのまま残す（末尾のポジションに含まれる）
        pass

    lines.append("📊 保有ポジション")
    lines.append(pos_text.strip() if pos_text else "ノーポジション")

    return "\n".join(lines)