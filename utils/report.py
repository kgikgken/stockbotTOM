from __future__ import annotations

from datetime import date
from typing import Dict, List, Tuple

from utils.util import safe_float


def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def build_report(
    today_str: str,
    today_date: date,
    market: Dict,
    screening: Dict,
    pos_text: str,
    total_asset: float,
) -> str:
    mkt_score = int(screening.get("meta", {}).get("market_score", market.get("score", 50)))
    delta3d = int(screening.get("meta", {}).get("delta3d", market.get("delta3d", 0)))
    lev = safe_float(screening.get("leverage", 1.0), 1.0)
    max_pos = int(screening.get("max_position", 0))

    notrade = bool(screening.get("notrade", False))
    notrade_reason = str(screening.get("notrade_reason", "")).strip()

    sectors = screening.get("sectors", []) or []
    events = screening.get("events", []) or []
    picks = screening.get("picks", []) or []
    watch = screening.get("watch", []) or []
    stats = screening.get("stats", {}) or {}

    header = []
    header.append(f"📅 {today_str} stockbotTOM 日報")
    header.append("")
    header.append("◆ 今日の結論（Swing専用 / 1〜7日）")
    if notrade:
        reason = f"（{notrade_reason}）" if notrade_reason else ""
        header.append(f"🚫 新規見送り {reason}".rstrip())
    else:
        header.append("✅ 新規可（条件クリア）")
    header.append(f"- 地合い: {mkt_score}点 ({market.get('comment','')})")
    header.append(f"- ΔMarketScore_3d: {delta3d:+d}")
    header.append(f"- レバ: {lev:.1f}倍（{screening.get('leverage_comment','')}）")
    header.append(f"- MAX建玉: 約{max_pos:,}円")
    header.append("")
    header.append("📈 セクター（5日）")
    if sectors:
        for i, (s, v) in enumerate(sectors, start=1):
            header.append(f"{i}. {s} ({v:+.2f}%)")
    else:
        header.append("- 取得失敗")
    header.append("")
    header.append("⚠ イベント")
    header.extend(events)
    header.append("")
    header.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    header.append(f"  候補数:{len(picks)}銘柄 / 平均RR:{stats.get('avg_rr',0):.2f} / 平均EV:{stats.get('avg_ev',0):.2f} / 平均AdjEV:{stats.get('avg_adj_ev',0):.2f} / 平均R/day:{stats.get('avg_r_per_day',0):.2f}")
    header.append("")

    body = []
    for c in picks:
        body.append(f"- {c['ticker']} {c.get('name','')} [{c.get('sector','')}]")
        body.append(f"  形:{c.get('setup','-')}  RR:{c.get('rr',0):.2f}  AdjEV:{c.get('adj_ev',0):.2f}  R/day:{c.get('r_per_day',0):.2f}")
        body.append(f"  IN:{c.get('in_center',0):.1f}（帯:{c.get('in_low',0):.1f}〜{c.get('in_high',0):.1f}） 現在:{c.get('price_now',0):.1f}  ATR:{c.get('atr',0):.1f}  GU:{c.get('gu','N')}")
        body.append(f"  STOP:{c.get('stop',0):.1f}  TP1:{c.get('tp1',0):.1f}  TP2:{c.get('tp2',0):.1f}  ExpectedDays:{c.get('expected_days',0):.1f}  行動:{'即IN可' if c.get('action')=='EXEC_NOW' else '指値待ち'}")
        body.append("")

    if not body:
        body.append("- 該当なし")
        body.append("")

    watch_lines = []
    watch_lines.append("🧠 監視リスト（今日は入らない）")
    if watch:
        for w in watch:
            t = w.get("ticker","")
            nm = w.get("name","")
            sec = w.get("sector","")
            rsn = w.get("reason","")
            if "rr" in w:
                watch_lines.append(f"- {t} {nm} [{sec}] 形:{w.get('setup','-')} RR:{w.get('rr',0):.2f} R/day:{w.get('r_per_day',0):.2f} 理由:{rsn} 行動:{'即IN可' if w.get('action')=='EXEC_NOW' else '指値待ち'} GU:{w.get('gu','N')}")
            else:
                watch_lines.append(f"- {t} {nm} [{sec}] 理由:{rsn}")
    else:
        watch_lines.append("- なし")

    risk_warn = bool(stats.get("risk_warn", False))
    if risk_warn:
        wl = float(stats.get("worst_loss_pct", 0.0))
        yen = float(stats.get("assumed_risk_yen", 0.0))
        watch_lines.append("")
        watch_lines.append(f"⚠ ロット事故警告：想定最大損失 ≈ {yen:,.0f}円（資産比 {wl*100:.2f}%）")

    footer = []
    footer.append("")
    footer.append("📊 ポジション")
    footer.append(pos_text.strip() if pos_text.strip() else "- なし")

    return "\n".join(header + body + watch_lines + footer).strip() + "\n"