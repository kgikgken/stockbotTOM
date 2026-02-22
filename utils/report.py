from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional
import numpy as np


from utils.util import safe_float

def _fmt_yen(x: float) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "-"


def _env_truthy(name: str, default: bool = False) -> bool:
    """Parse a boolean-like environment variable.

    - If the variable is not set, returns `default`.
    - Accepts common truthy/falsy strings.
    """

    raw = os.getenv(name)
    if raw is None:
        return default
    v = str(raw).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _strip_icons(s: str) -> str:
    """Best-effort removal of emoji markers for image/table rendering."""

    if not s:
        return ""
    for ch in ("🟢", "🔴", "⚠", "✅", "🚫"):
        s = s.replace(ch, "")
    return " ".join(s.split())


def _symbol_cell(head: str) -> str:
    """Format the '銘柄' cell for the table image.

    To improve readability in LINE, split ticker and name/tags into 2 lines.
    Example: "7599.T IDOM [A1/...]" -> "7599.T\nIDOM [A1/...]"
    """
    s = _strip_icons(head)
    if " " in s:
        a, b = s.split(" ", 1)
        return f"{a}\n{b}"
    return s



def _fmt_oku(yen: float) -> str:
    """Format yen value as Japanese "億" unit (1億=1e8円).

    Used as a liquidity proxy (ADV20 / median traded value).
    """
    try:
        y = float(yen)
    except Exception:
        return "-"
    if not (y > 0) or not (y == y):
        return "-"
    oku = y / 1e8
    if oku < 10:
        return f"{oku:.1f}億"
    return f"{oku:.0f}億"

def build_report(
    today_str: str,
    market: Dict,
    delta3: float,
    futures_chg: float,
    risk_on: bool,
    macro_on: bool,
    events_lines: List[str],
    no_trade: bool,
    weekly_used: int,
    weekly_max: int,
    leverage: float,
    policy_lines: List[str],
    cands: List[Dict],
    pos_text: str,
    no_trade_reason: str | None = None,
    saucers: Dict[str, List[Dict]] | List[Dict] | None = None,
) -> str:
    mkt_score = int(market.get("score", 50))
    mkt_comment = str(market.get("comment", "中立"))

    lines: List[str] = []

    # Optional: build a compact, structured summary for exporting as an image.
    # The LINE text output stays beginner-friendly; the image is for at-a-glance sharing.
    table_headers = ["区分", "#", "銘柄", "注文", "SL", "TP1/リム", "Risk", "メモ"]
    table_rows: List[List[str]] = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")

    # Macro day preface (keep strict; do not promote market-in on event days)
    if macro_on:
        lines.append("⚠ 本日は重要イベント警戒日")
        if risk_on:
            lines.append("※ 先物Risk-ONにつき、警戒しつつ最大5まで表示")
        lines.append("")
        if events_lines:
            lines.append("対象イベント：")
            for ev in events_lines:
                if ev.startswith("⚠ "):
                    lines.append("・" + ev.replace("⚠ ", "").split("（")[0])
            lines.append("")
    # Header
    if no_trade:
        reason = no_trade_reason or ("重要イベント警戒" if macro_on else "地合い条件")
        lines.append(f"新規：🛑 NO（{reason}）")
    else:
        lines.append("新規：✅ OK（指値 / 現値INは銘柄別）")
    lines.append("")

    fut_txt = f"  先物:{futures_chg:+.2f}%(NKD=F) {'Risk-ON' if risk_on else ''}".rstrip()
    lines.append(f"地合い：{mkt_score}（{mkt_comment}）  ΔMarketScore_3d:{delta3:.1f}{fut_txt}")
    lines.append(f"Macro警戒：{'ON' if macro_on else 'OFF'}")
    lines.append(f"週次新規：{weekly_used} / {weekly_max}")
    lines.append(f"推奨レバ：{leverage:.1f}x")
    lines.append("")

    # Policy (explicit; was previously computed but not rendered)
    if policy_lines:
        lines.append("🧭 運用ルール（本日）")
        for p in policy_lines:
            if str(p).strip():
                lines.append("・" + str(p).strip())
        lines.append("")

    # Candidates (beginner-first)
    if cands:
        lines.append("👀 監視リスト（新規は見送り / 最大5）" if no_trade else "🏆 狙える形（ランキング / 最大5）")

        # Beginner-first output: show *what to do* (order type & price) and hide the rest.
        # Avoid confusing "buy limit above market" situations: if price is below the band, we do NOT suggest a limit order.
        band_tol = 0.0005  # 0.05% 表示/判定のズレを吸収

        def _risk_mid(entry_p: float, sl_p: float) -> float:
            """Risk% for a long entry (distance to SL)."""
            if entry_p > 0 and sl_p > 0 and entry_p > sl_p:
                return float((entry_p - sl_p) / entry_p * 100.0)
            return float("nan")

        def _risk_txt_for_entries(entries: List[float], sl_p: float) -> str:
            """Format risk for ladder entries.

            - N>=2: returns a range like "1.8〜2.4%".
            - N==1: returns a single value like "2.1%".
            """

            vals: List[float] = []
            for e in entries:
                r = _risk_mid(float(e), sl_p)
                if np.isfinite(r):
                    vals.append(float(r))
            if not vals:
                return "-"
            if len(vals) == 1:
                return f"{vals[0]:.1f}%"
            lo = float(min(vals))
            hi = float(max(vals))
            if abs(hi - lo) < 0.05:
                return f"{hi:.1f}%"
            return f"{lo:.1f}〜{hi:.1f}%"

        # Keep original ranking (idx) but renumber within each bucket for readability.
        # For readability, split order and risk into separate lines.
        order_items: List[Tuple[int, str, str, str]] = []
        watch_items: List[Tuple[int, str, str]] = []
        skip_items: List[Tuple[int, str]] = []

        for idx, c in enumerate(cands, 1):
            ticker = str(c.get("ticker", ""))
            name = str(c.get("name", ticker))
            setup = str(c.get("setup", "")).strip()

            entry_low = safe_float(c.get("entry_low"), 0.0)
            entry_high = safe_float(c.get("entry_high"), 0.0)
            entry_price = safe_float(c.get("entry_price"), (entry_low + entry_high) / 2.0)
            sl = safe_float(c.get("sl"), 0.0)
            tp1 = safe_float(c.get("tp1"), 0.0)
            close_last = safe_float(c.get("close_last"), 0.0)

            # Market-in (現値IN)
            entry_mode = str(c.get("entry_mode", "LIMIT_ONLY"))
            in_band = (
                (close_last > 0)
                and (entry_low > 0)
                and (entry_high > 0)
                and (close_last >= entry_low * (1.0 - band_tol))
                and (close_last <= entry_high * (1.0 + band_tol))
            )
            # Market-in is only allowed when the tape is supportive.
            # - Require Risk-ON day to avoid catching knives on weak index days.
            # - Optionally require futures not too negative (env: MARKET_IN_MIN_FUTURES).
            min_fut = safe_float(os.getenv("MARKET_IN_MIN_FUTURES", "-0.50"), -0.50)
            fut_ok = True if futures_chg is None else (float(futures_chg) >= float(min_fut))
            market_in_ok = bool(
                entry_mode == "MARKET_OK"
                and in_band
                and (not macro_on)
                and (not no_trade)
                and bool(risk_on)
                and fut_ok
            )

            # Liquidity summary tags (beginner-first): keep only categorical info.
            # We intentionally hide numeric liquidity metrics (ADV/Impact) to avoid clutter.
            liq_grade = int(safe_float(c.get("liq_grade"), 0.0)) if c.get("liq_grade") is not None else 0
            weekly_ok = c.get("weekly_ok", None)

            tags: List[str] = []
            if setup:
                tags.append(setup)
            if liq_grade == 2:
                tags.append("板厚◎")
            elif liq_grade == 1:
                tags.append("板厚○")
            if weekly_ok is True:
                tags.append("週足OK")
            elif weekly_ok is False:
                tags.append("週足NG")
            tag_txt = f" [{'/'.join(tags)}]" if tags else ""

            # Compute concise reason
            ns = safe_float(c.get("noise_score"), float("nan"))
            q = safe_float(c.get("quality"), float("nan"))
            vr = safe_float(c.get("vol_ratio"), float("nan"))
            gf = safe_float(c.get("gap_freq"), float("nan"))
            gu = bool(c.get("gu", False))

            # Where is price vs band?
            above_band = bool(close_last > 0 and entry_high > 0 and close_last > entry_high * (1.0 + band_tol))
            below_band = bool(close_last > 0 and entry_low > 0 and close_last < entry_low * (1.0 - band_tol))

            # Beginner action classification
            # - ORDER: either market-in (rare) or safe pullback limit below current
            # - WATCH: below the band (do not suggest a limit order above market)
            # - SKIP: event day / macro / GU / quality-noise issues
            if no_trade:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（停止）"))
                continue
            if macro_on:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（イベント）"))
                continue

            blackout = str(c.get("blackout_reason", "") or "").strip()
            if blackout:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（イベント:{blackout}）"))
                continue
            if gu:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（GU）"))
                continue
            if weekly_ok is False and setup in ("A1-Strong", "A1"):
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（週足NG）"))
                continue
            noise_skip = safe_float(os.getenv("REPORT_NOISE_SKIP_SCORE", "2"), 2.0)
            if in_band and np.isfinite(ns) and ns >= float(noise_skip):
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（ノイズ{int(ns)}）"))
                continue
            if in_band and np.isfinite(vr) and vr > 1.35:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（出来高↑）"))
                continue
            if in_band and np.isfinite(gf) and gf > 0.25:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（Gap多）"))
                continue
            if in_band and np.isfinite(q) and q < -0.05:
                skip_items.append((idx, f"🔴 {ticker} {name}{tag_txt} 見送り（品質低）"))
                continue

            # Order suggestion
            if market_in_ok:
                r_mid = _risk_mid(close_last, sl)
                risk_txt = f"{r_mid:.1f}%" if np.isfinite(r_mid) else "-"
                order_items.append(
                    (
                        idx,
                        f"🟢 {ticker} {name}{tag_txt}",
                        f"成行（寄り後）{_fmt_yen(close_last)}",
                        f"SL {_fmt_yen(sl)} / TP1 {_fmt_yen(tp1)} / Risk {risk_txt}",
                    )
                )
                continue


            # If price is far above the pullback band, auto-placing limit orders often
            # results in low fill-rate. In that case, downgrade to WATCH (no order).
            if above_band and close_last > 0 and entry_high > 0:
                max_wait_pct = safe_float(os.getenv("PULLBACK_WAIT_MAX_PCT", "4.0"), 4.0)
                max_wait_atr = safe_float(os.getenv("PULLBACK_WAIT_MAX_ATR", "1.8"), 1.8)
                dist_pct = (close_last / entry_high - 1.0) * 100.0
                atrp = safe_float(c.get("atrp"), 0.0)
                atr_yen = (close_last * atrp / 100.0) if (close_last > 0 and atrp > 0) else 0.0
                dist_atr = ((close_last - entry_high) / atr_yen) if atr_yen > 0 else float("nan")
                if dist_pct > float(max_wait_pct) or (np.isfinite(dist_atr) and dist_atr > float(max_wait_atr)):
                    watch_items.append(
                        (
                            idx,
                            f"🟡 {ticker} {name}{tag_txt}",
                            f"監視（伸びすぎ：帯まで +{dist_pct:.1f}%）",
                        )
                    )
                    continue

            if above_band and entry_price > 0 and close_last > 0 and entry_price < close_last:
                # Pullback limit ladder to improve fill-rate.
                # Default: 3-step ladder (上→中→下) within the pullback band.
                levels_raw: List[float] = []
                if entry_high > 0:
                    levels_raw.append(entry_high)
                if entry_price > 0:
                    levels_raw.append(entry_price)
                if entry_low > 0:
                    levels_raw.append(entry_low)

                # Normalize order: shallow(high) -> deep(low)
                levels_raw = [float(x) for x in levels_raw if np.isfinite(x) and x > 0]
                levels_raw = sorted(levels_raw, reverse=True)

                # De-duplicate (prices are tick-rounded; int round is safe)
                uniq: List[float] = []
                for lv in levels_raw:
                    if not uniq:
                        uniq.append(lv)
                    else:
                        if int(round(lv)) != int(round(uniq[-1])):
                            uniq.append(lv)

                # Risk-cap filter (max 8%)
                levels: List[float] = []
                for lv in uniq:
                    r = _risk_mid(lv, sl)
                    if np.isfinite(r) and r <= 8.0 + 1e-6:
                        levels.append(lv)

                if not levels:
                    # Fallback
                    levels = [entry_price]

                # Decide ladder display
                if len(levels) >= 3 and (levels[0] / levels[-1] - 1.0) >= 0.002:
                    order_line = (
                        f"指値（押し待ち3段）浅 {_fmt_yen(levels[0])} / 中 {_fmt_yen(levels[1])} / 深 {_fmt_yen(levels[2])}"
                    )
                elif len(levels) >= 2 and (levels[0] / levels[-1] - 1.0) >= 0.002:
                    order_line = f"指値（押し待ち2段）浅 {_fmt_yen(levels[0])} / 深 {_fmt_yen(levels[-1])}"
                else:
                    order_line = f"指値（押し待ち）{_fmt_yen(levels[0])}"

                risk_txt = _risk_txt_for_entries(levels, sl)

                order_items.append(
                    (
                        idx,
                        f"🟢 {ticker} {name}{tag_txt}",
                        order_line,
                        f"SL {_fmt_yen(sl)} / TP1 {_fmt_yen(tp1)} / Risk {risk_txt}",
                    )
                )
                continue

            if in_band and entry_price > 0 and close_last > 0:
                if entry_price <= close_last:
                    # In-zone: avoid limit above market.
                    # Use 2-step ladder: mid(entry_price) -> deep(entry_low).
                    levels: List[float] = [float(entry_price)]
                    if entry_low > 0 and np.isfinite(entry_low) and entry_low < entry_price:
                        # Risk-cap filter
                        r_deep = _risk_mid(entry_low, sl)
                        if np.isfinite(r_deep) and r_deep <= 8.0 + 1e-6:
                            levels.append(float(entry_low))

                    if len(levels) >= 2 and (levels[0] / levels[1] - 1.0) >= 0.002:
                        order_line = f"指値（帯内2段）中 {_fmt_yen(levels[0])} / 深 {_fmt_yen(levels[1])}"
                    else:
                        order_line = f"指値（帯内）{_fmt_yen(levels[0])}"
                    risk_txt = _risk_txt_for_entries(levels, sl)

                    order_items.append(
                        (
                            idx,
                            f"🟢 {ticker} {name}{tag_txt}",
                            order_line,
                            f"SL {_fmt_yen(sl)} / TP1 {_fmt_yen(tp1)} / Risk {risk_txt}",
                        )
                    )
                else:
                    watch_items.append(
                        (
                            idx,
                            f"🟡 {ticker} {name}{tag_txt}",
                            "監視（帯内だが指値が上：注文は様子見）",
                        )
                    )
                continue

            if below_band:
                # Price is *below* the pullback band.
                # Instead of just "watch", we can propose a *re-entry buy stop* at the band low,
                # so you don't miss it when it snaps back into the zone.
                dist_to_band = float("nan")
                if close_last > 0 and entry_low > 0:
                    dist_to_band = (entry_low / close_last - 1.0) * 100.0

                reentry_max = safe_float(os.getenv("TREND_REENTRY_STOP_MAX_DIST_PCT"), 2.5)
                reentry_ok = (
                    np.isfinite(dist_to_band)
                    and (dist_to_band <= reentry_max)
                    and (entry_low > 0)
                    and (entry_high > 0)
                    and (sl > 0)
                )

                if reentry_ok:
                    r_low = _risk_mid(entry_low, sl)
                    r_high = _risk_mid(entry_high, sl)
                    if np.isfinite(r_low) and np.isfinite(r_high):
                        r_lo, r_hi = sorted([r_low, r_high])
                        if r_hi <= 8.0 + 1e-9:
                            risk_txt = (
                                f"{r_lo:.1f}〜{r_hi:.1f}%" if abs(r_hi - r_lo) >= 0.15 else f"{r_hi:.1f}%"
                            )
                            order_items.append(
                                (
                                    idx,
                                    f"🟢 {ticker} {name}{tag_txt}",
                                    f"逆指値（戻り） Trg {_fmt_yen(entry_low)} / 上限 {_fmt_yen(entry_high)}",
                                    f"SL {_fmt_yen(sl)} / TP1 {_fmt_yen(tp1)} / Risk {risk_txt}",
                                )
                            )
                        else:
                            watch_items.append(
                                (
                                    idx,
                                    f"🟡 {ticker} {name}{tag_txt}",
                                    f"監視（帯下）距離+{dist_to_band:.1f}% / Risk {r_hi:.1f}%",
                                )
                            )
                    else:
                        watch_items.append((idx, f"🟡 {ticker} {name}{tag_txt}", "監視（戻り待ち：帯まで距離あり）"))
                else:
                    watch_items.append((idx, f"🟡 {ticker} {name}{tag_txt}", "監視（戻り待ち：帯まで距離あり）"))
                continue

            watch_items.append((idx, f"🟡 {ticker} {name}{tag_txt}", "監視"))

        if order_items:
            lines.append("✅ 今日やること：注文（上から優先）")
            _orders = sorted(order_items, key=lambda x: x[0])
            for n, (_rank, head, order_line, risk_line) in enumerate(_orders, 1):
                lines.append(f"{n}. {head}")
                lines.append(f"   注文：{order_line}")
                lines.append(f"   {risk_line}")

                # Table row for optional image export
                sl_txt, tp1_txt, risk_txt = "-", "-", "-"
                try:
                    parts = [p.strip() for p in str(risk_line).split("/")]
                    if len(parts) >= 1 and parts[0].startswith("SL"):
                        sl_txt = parts[0].replace("SL", "", 1).strip()
                    if len(parts) >= 2 and parts[1].startswith("TP1"):
                        tp1_txt = parts[1].replace("TP1", "", 1).strip()
                    if len(parts) >= 3 and "Risk" in parts[2]:
                        risk_txt = parts[2].replace("Risk", "", 1).strip()
                except Exception:
                    pass
                table_rows.append(
                    [
                        "狙える",
                        str(n),
                        _symbol_cell(head),
                        _strip_icons(order_line),
                        sl_txt,
                        tp1_txt,
                        risk_txt,
                        "",
                    ]
                )

                if n != len(_orders):
                    lines.append("")
        else:
            lines.append("✅ 今日やること：注文")
            lines.append("・該当なし")

        if watch_items:
            lines.append("")
            lines.append("👀 監視（まだ入らない）")
            _watch = sorted(watch_items, key=lambda x: x[0])
            for n, (_rank, head, detail) in enumerate(_watch, 1):
                lines.append(f"{n}. {head}")
                if detail:
                    lines.append(f"   {detail}")
                if n != len(_watch):
                    lines.append("")

        if skip_items:
            lines.append("")
            lines.append("🚫 見送り")
            _skips = sorted(skip_items, key=lambda x: x[0])
            for n, (_rank, txt) in enumerate(_skips, 1):
                # Split into 2 lines to improve readability.
                # Example: "🔴 2986.T ... 見送り（ノイズ2）"
                head = txt
                reason = ""
                if " 見送り" in txt:
                    head, tail = txt.split(" 見送り", 1)
                    head = head.strip()
                    reason = ("見送り" + tail).strip()
                lines.append(f"{n}. {head}")
                if reason:
                    lines.append(f"   {reason}")

                # Table row
                table_rows.append(
                    [
                        "見送り",
                        "-",
                        _symbol_cell(head),
                        "見送り",
                        "-",
                        "-",
                        "-",
                        _strip_icons(reason) if reason else "",
                    ]
                )
                if n != len(_skips):
                    lines.append("")

        lines.append("")
    else:
        lines.append("🏆 狙える形（1〜7営業日 / 最大5）")
        lines.append("・該当なし")
        lines.append("")

    # Positions (beginner-first: compact one line per position)
    if pos_text.strip():
        import re

        def _pick_num(line: str) -> str:
            """Pick the *price-like* number from a segment.

            Important: segments like "TP1 26,205" contain a digit in the label ("TP1").
            A naive regex would incorrectly return "1".
            """
            s = str(line or "")
            # Remove TP/TP1/TP2... tokens to avoid picking the digit in the label.
            s = re.sub(r"\bTP[0-9]+\b", "TP", s)

            # Prefer comma-grouped numbers (most yen prices are formatted like 12,345).
            m = re.search(r"([0-9]{1,3}(?:,[0-9]{3})+)", s)
            if m:
                return m.group(1)

            # Fallback: any 2+ digit number.
            m = re.search(r"([0-9]{2,})", s)
            if m:
                return m.group(1)

            # Last resort: single digit, but only if preceded by a separator.
            m = re.search(r"(?:\s|：)([0-9])\b", s)
            return m.group(1) if m else ""

        def _cut_tail(s: str) -> str:
            for sep in ("（", " / "):
                if sep in s:
                    s = s.split(sep, 1)[0]
            return s.strip()

        raw_lines = [ln.strip() for ln in pos_text.splitlines() if ln.strip()]
        blocks: List[List[str]] = []
        cur: List[str] = []
        for ln in raw_lines:
            if ln.startswith("■ ") and cur:
                blocks.append(cur)
                cur = []
            cur.append(ln)
        if cur:
            blocks.append(cur)

        lines.append("📊 ポジション（やること）")
        for b in blocks:
            head = b[0].replace("■", "").strip()
            status = ""
            next_act = ""
            entry = ""
            now = ""
            pnl = ""
            sl = ""
            tp1 = ""
            setup_used = ""
            for ln in b[1:]:
                # New compact position format (utils/position.py): one bullet line with '/' separated segments.
                ln_clean = str(ln or "").strip()
                if ln_clean.startswith("・"):
                    ln_clean = ln_clean.lstrip("・").strip()
                    segs = [s.strip() for s in ln_clean.split("/") if s.strip()]
                    for seg in segs:
                        if seg.startswith("Entry ") and not entry:
                            entry = _pick_num(seg)
                            continue
                        if seg.startswith("Now ") and not now:
                            now = _pick_num(seg)
                            continue
                        if seg.startswith("PnL ") and not pnl:
                            pnl = seg.replace("PnL", "", 1).strip()
                            continue
                        if seg.startswith("SL ") and not sl:
                            sl = _pick_num(seg)
                            continue
                        if seg.startswith("TP1 ") and not tp1:
                            tp1 = _pick_num(seg)
                            continue
                        if seg.startswith("Setup ") and not setup_used:
                            setup_used = seg.replace("Setup", "", 1).strip()
                            continue
                        if seg.startswith("次:") and not next_act:
                            next_act = seg.split("次:", 1)[1].strip()
                            continue

                if "状態：" in ln and not status:
                    status = _cut_tail(ln.split("状態：", 1)[1])
                if "次アクション：" in ln and not next_act:
                    next_act = _cut_tail(ln.split("次アクション：", 1)[1])
                if "次:" in ln and not next_act:
                    next_act = _cut_tail(ln.split("次:", 1)[1])
                if "取得単価：" in ln and not entry:
                    m1 = re.search(r"取得単価：\s*([0-9,]+)\s*円", ln)
                    if m1:
                        entry = m1.group(1)
                    m2 = re.search(r"現値：\s*([0-9,]+)", ln)
                    if m2:
                        now = m2.group(1)
                if "Entry" in ln and not entry:
                    entry = _pick_num(ln)
                if ("現値：" in ln or "Now" in ln) and not now:
                    now = _pick_num(ln)
                if "損益：" in ln and not pnl:
                    pnl = _cut_tail(ln.split("損益：", 1)[1])
                if "PnL" in ln and not pnl:
                    pnl = _cut_tail(ln.split("PnL", 1)[1])
                if ("想定SL：" in ln or "SL：" in ln or "SL " in ln) and not sl:
                    sl = _pick_num(ln)
                if ("想定TP1：" in ln or "TP1：" in ln or "TP1 " in ln) and not tp1:
                    tp1 = _pick_num(ln)
                if "Setup" in ln and not setup_used:
                    setup_used = _cut_tail(ln.split("Setup", 1)[1])

            act = next_act or status or "保有"

            # Beginner-first: multi-line, no ambiguity about what to do.
            lines.append(f"■ {head}：{act}")

            p1_parts: List[str] = []
            if entry or now:
                e = entry or "-"
                n = now or "-"
                p1_parts.append(f"Entry {e} → Now {n}")
            if pnl:
                p1_parts.append(f"PnL {pnl}")
            if p1_parts:
                lines.append("   " + " / ".join(p1_parts))

            p2_parts: List[str] = []
            if sl:
                p2_parts.append(f"SL {sl}")
            if tp1:
                p2_parts.append(f"TP1 {tp1}")
            if setup_used:
                p2_parts.append(f"Setup {setup_used}")
            if p2_parts:
                lines.append("   " + " / ".join(p2_parts))

            # Table row
            memo_parts: List[str] = []
            if entry or now:
                e = entry or "-"
                n = now or "-"
                memo_parts.append(f"Entry {e}→{n}")
            if setup_used:
                memo_parts.append(f"Setup {setup_used}")
            table_rows.append(
                [
                    "ポジ",
                    "-",
                    _symbol_cell(head),
                    _strip_icons(act),
                    sl or "-",
                    tp1 or "-",
                    pnl or "-",
                    " / ".join(memo_parts),
                ]
            )

            # Visual separator between multiple positions
            lines.append("")
        # (blank line already appended after each block)

    # Summary: removed (beginner-first mode). The actionable list above is the summary.

    # Saucer bucket (beginner-first)
    # Expected format: dict {"D":[...], "W":[...], "M":[...]}
    if saucers:
        def _iter_tf(key: str):
            if isinstance(saucers, dict):
                return list(saucers.get(key, []) or [])
            # backward-compat: legacy list with 'timeframe' = 'W'/'M'
            if isinstance(saucers, list):
                if key == "W":
                    return [x for x in saucers if str(x.get("timeframe", "W")) == "W"]
                if key == "M":
                    return [x for x in saucers if str(x.get("timeframe", "W")) == "M"]
                return []
            return []

        def _tf_title(key: str) -> str:
            return {"D": "日足", "W": "週足", "M": "月足"}.get(key, key)

        def _len_label(tf_key: str, n: int) -> str:
            if n <= 0:
                return "-"
            if tf_key == "W":
                return f"{n}週"
            if tf_key == "M":
                return f"{n}ヶ月"
            return f"{n}本"

        for key in ("D", "W", "M"):
            items = _iter_tf(key)[:5]
            if lines and key != "D":
                lines.append("")
            lines.append(f"🥣 ソーサー枠（{_tf_title(key)}）ランキング（最大5）")
            if not items:
                lines.append("・該当なし")
                continue

            for idx, s in enumerate(items, 1):
                ticker = str(s.get("ticker", ""))
                name = str(s.get("name", ticker))
                tier = str(s.get("tier", "A") or "A")
                tier_tag = "（準候補）" if tier.upper() == "B" else ""

                rim_f = safe_float(s.get("rim"), 0.0)
                last_f = safe_float(s.get("last"), 0.0)
                atrp_f = safe_float(s.get("atrp"), 0.0)
                cup_len = int(s.get("cup_len", 0) or 0)
                progress = float(s.get("progress", 0.0))
                prog_pct = int(round(min(1.5, max(0.0, progress)) * 100))

                # Zone/SL
                entry_low = safe_float(s.get("entry_low"), float("nan"))
                entry_high = safe_float(s.get("entry_high"), float("nan"))
                sl_s = safe_float(s.get("sl"), float("nan"))
                hvol_ratio = safe_float(s.get("handle_vol_ratio"), float("nan"))
                warn = " ⚠" if (np.isfinite(hvol_ratio) and hvol_ratio >= 1.25) else ""

                # If scan provides an explicit zone, use it; otherwise fallback to a rim-buffer zone.
                if rim_f > 0 and np.isfinite(entry_low) and np.isfinite(entry_high) and entry_low > 0 and entry_high > 0:
                    zone_low = float(min(entry_low, entry_high))
                    zone_high = float(max(entry_low, entry_high))
                else:
                    base_pre = {"D": 0.6, "W": 0.9, "M": 1.2}.get(key, 0.8)  # percent
                    max_pre = {"D": 2.0, "W": 3.0, "M": 4.0}.get(key, 2.5)   # percent
                    atr_pre = (atrp_f * 0.35) if atrp_f > 0 else 0.0
                    pre_buf_pct = max(base_pre, atr_pre)
                    pre_buf_pct = min(pre_buf_pct, max_pre)
                    zone_low = rim_f * (1.0 - pre_buf_pct / 100.0) if rim_f > 0 else 0.0
                    zone_high = rim_f * (1.0 - base_pre / 100.0) if rim_f > 0 else 0.0
                    zone_high = max(zone_low, zone_high)

                if not (zone_low > 0 and zone_high > 0 and np.isfinite(sl_s) and sl_s > 0):
                    lines.append(f"{idx}. 🟡 {ticker} {name}{tier_tag} 監視（ゾーン計算失敗）")
                    continue

                # Risk range inside the zone
                r1 = (zone_low - sl_s) / zone_low * 100.0
                r2 = (zone_high - sl_s) / zone_high * 100.0
                r_lo = min(r1, r2)
                r_hi = max(r1, r2)
                risk_txt = f"{r_lo:.1f}〜{r_hi:.1f}%" if abs(r_hi - r_lo) >= 0.15 else f"{r_hi:.1f}%"

                # Order type hint
                order_tag = "指値"
                tol_zone = 0.0010
                if last_f > 0:
                    if last_f < zone_low * (1.0 - tol_zone):
                        order_tag = "逆指値"
                    elif last_f > zone_high * (1.0 + tol_zone):
                        order_tag = "押し待ち指値"

                # Where is price now? (one short note; avoid nested parentheses)
                now_note = ""
                if last_f > 0:
                    if last_f < zone_low * (1.0 - tol_zone):
                        to_zone = (zone_low / last_f - 1.0) * 100.0
                        # Beginner-first: make the action explicit.
                        now_note = f"状態：下 / ゾーンまで +{to_zone:.1f}% / 逆指値待ち"
                    elif last_f > zone_high * (1.0 + tol_zone):
                        over = (last_f / zone_high - 1.0) * 100.0
                        risk_last = (last_f - sl_s) / last_f * 100.0
                        # Above zone: this is a pullback-limit idea. Explicitly ban market chasing.
                        now_note = f"状態：上 / ゾーン上 +{over:.1f}%"
                        if np.isfinite(risk_last):
                            now_note += f" / 成行Risk {risk_last:.1f}%"
                            if risk_last > 8.0:
                                now_note += "（8%超）"
                        now_note += " / 成行禁止（指値待ち）"
                    else:
                        now_note = "状態：ゾーン内（注文有効）"

                # Print one line per symbol
                if order_tag == "逆指値":
                    if abs(zone_high / zone_low - 1.0) <= 0.001:
                        ord_txt = f"逆指値 Trg {_fmt_yen(zone_low)}"
                    else:
                        ord_txt = f"逆指値 Trg {_fmt_yen(zone_low)} / 上限 {_fmt_yen(zone_high)}"
                else:
                    if abs(zone_high / zone_low - 1.0) <= 0.001:
                        ord_txt = f"指値 {_fmt_yen(zone_low)}"
                    else:
                        ord_txt = f"指値 {_fmt_yen(zone_low)}〜{_fmt_yen(zone_high)}"
                    if order_tag == "押し待ち指値":
                        ord_txt = "指値（押し待ち）" + ord_txt.replace("指値 ", "")

                # Beginner-first: use 2 lines per symbol.
                # (Progress/length are kept in data but hidden to reduce noise.)
                lines.append(f"{idx}. 🟢 {ticker} {name}{tier_tag}{warn}")
                lines.append("   " + f"注文：{ord_txt}")
                lines.append("   " + f"SL {_fmt_yen(sl_s)} / Risk {risk_txt}")
                if now_note:
                    lines.append("   " + now_note)

                # Table row
                memo_parts: List[str] = []
                if tier_tag:
                    memo_parts.append("準候補")
                if warn:
                    memo_parts.append("出来高⚠")
                if now_note:
                    # keep it short (just the first clause)
                    memo_parts.append(now_note.replace("状態：", "").split("/")[0].strip())
                table_rows.append(
                    [
                        f"ソーサー（{_tf_title(key)}）",
                        str(idx),
                        _strip_icons(f"{ticker} {name}{tier_tag}"),
                        _strip_icons(ord_txt),
                        _fmt_yen(sl_s),
                        _fmt_yen(rim_f) if rim_f > 0 else "-",
                        risk_txt,
                        " / ".join(memo_parts),
                    ]
                )
                if idx != len(items):
                    lines.append("")
    # Optional: export a shareable PNG/CSV table.
    #
    # NOTE:
    #   Users often expect "it changed" without setting env vars, so we default
    #   this ON when table_rows exist.
    #
    # Disable explicitly with:
    #   REPORT_TABLE_IMAGE=0   (or REPORT_IMAGE=0)
    img_enabled = _env_truthy("REPORT_TABLE_IMAGE", default=_env_truthy("REPORT_IMAGE", default=True))
    note_enabled = _env_truthy("REPORT_IMAGE_NOTE", default=True)

    if table_rows and img_enabled:
        outdir = os.getenv("REPORT_OUTDIR", "out")
        os.makedirs(outdir, exist_ok=True)

        base = os.path.join(outdir, f"report_table_{today_str}")
        png_main = base + ".png"
        png_d = base + "_d.png"
        png_w = base + "_w.png"
        png_m = base + "_m.png"
        csv_path = base + ".csv"

        new_str = "OK" if (not no_trade) else "NG"
        macro_str = "ON" if macro_on else "OFF"
        fut_str = f"{futures_chg:+.2f}%" if futures_chg is not None else "--"

        subtitle = (
            f"新規:{new_str}  地合い:{mkt_score}  先物:{fut_str}  Macro:{macro_str}  "
            f"週次:{weekly_used}/{weekly_max}  レバ:{leverage:.1f}x"
        )

        try:
            from utils.table_image import TableImageStyle, render_table_csv, render_table_png
        except Exception as e:
            if note_enabled:
                lines.append("")
                lines.append(f"🖼 表画像: 生成不可（table_image import失敗: {e}）")
        else:
            # CSV is lightweight and always useful for debugging/backtesting.
            try:
                render_table_csv(f"stockbotTOM {today_str} 注文サマリ", table_headers, table_rows, csv_path)
                if note_enabled:
                    lines.append("")
                    lines.append(f"🗒 注文サマリCSV: {csv_path}")
            except Exception as e:
                if note_enabled:
                    lines.append("")
                    lines.append(f"🗒 注文サマリCSV: 生成失敗（{e}）")

            # --- Image rendering (mobile-first) ---
            import re

            def _shorten_order_text(s: str) -> str:
                s = (s or "").strip()
                s = s.replace("成行（現値）", "成行(現)")
                s = s.replace("成行（現）", "成行(現)")
                s = s.replace("成行（寄り後）", "成行(寄)")
                s = s.replace("指値（押し待ち3段）", "指値(押)")
                s = s.replace("指値（押し待ち2段）", "指値(押)")
                s = s.replace("指値（押し待ち）", "指値(押)")
                s = s.replace("指値（押）", "指値(押)")
                s = s.replace("指値（帯内2段）", "指値(帯)")
                s = s.replace("指値（帯内）", "指値(帯)")
                s = s.replace("指値（帯）", "指値(帯)")
                return s.strip()

            def _format_order_cell(s: str) -> str:
                """注文セルをLINE向けに整形。

                - 意味の区切りで改行して、文字の途中で不自然に割れないようにする
                - stop(逆指値)の "Trg" が "T / rg" に割れるのを抑える
                - レンジ(〜)は区切りで改行して価格が崩れないようにする
                """

                s = _shorten_order_text(s)
                if not s:
                    return ""

                # slash 区切りは必ず改行（2段指値/上限など）
                s = s.replace(" / ", "\n")

                # stop注文: "逆指値 Trg 4,253" → "逆指値\nTrg\n4,253"
                s = s.replace("逆指値(戻) Trg", "逆指値(戻)\nTrg")
                s = s.replace("逆指値 Trg", "逆指値\nTrg")
                s = re.sub(r"\bTrg\s*([0-9,]+)", r"Trg\n\1", s)
                s = re.sub(r"\b上限\s*([0-9,]+)", r"上限\n\1", s)
                s = re.sub(r"\b下限\s*([0-9,]+)", r"下限\n\1", s)

                # レンジ注文: "指値 4,935〜5,003" → "指値\n4,935〜\n5,003"
                if s.startswith("指値 ") and "〜" in s:
                    s = s.replace("指値 ", "指値\n", 1)
                    s = s.replace("〜", "〜\n", 1)
                elif "〜" in s:
                    s = s.replace("〜", "〜\n", 1)

                # Order-type と価格は必ず改行して、数値が途中で割れないようにする。
                # 例: "成行(現) 1,506" -> "成行(現)\n1,506"
                s = re.sub(r"^(成行\(現\)|成行\(寄\)|指値\(押\)|指値\(帯\)|指値)\s*", r"\1\n", s)

                # Many order strings are concatenated like "指値(押)1,489".
                # If the above rule didn't trigger, insert a single newline before the first digit.
                if "\n" not in s:
                    s = re.sub(r"([^\s])([0-9])", r"\1\n\2", s, count=1)

                return s.strip()

            def _split_symbol_cell(cell: str) -> tuple[str, str]:
                cell = (cell or "").strip()
                if not cell:
                    return "", ""
                m = re.search(r"(\[[^\]]+\])\s*$", cell)
                if m:
                    tags = m.group(1).strip("[]").strip()
                    main = cell[: m.start()].strip()
                    return main, tags
                return cell, ""

            def _format_tags(tags: str) -> str:
                tags = _strip_icons(tags)
                if not tags:
                    return ""
                tags = tags.replace("・", "/")
                parts = [p.strip() for p in tags.split("/") if p.strip()]

                short: list[str] = []
                for p in parts:
                    # Compact tags for mobile (reduce column width; prevents awkward line breaks)
                    if p == "A1-Strong":
                        p = "A1S"
                    elif p == "週足OK":
                        p = "週OK"
                    elif p == "週足NG":
                        p = "週NG"
                    elif p == "板厚◎":
                        p = "厚◎"
                    elif p == "板厚○":
                        p = "厚○"
                    short.append(p)

                return "/".join(short)

            def _format_risk_block(sl: str, tp1: str, risk: str) -> str:
                sl = (sl or "").strip()
                tp1 = (tp1 or "").strip()
                risk = (risk or "").strip()
                out: list[str] = []
                if sl:
                    out.append(f"SL {sl}")
                if tp1:
                    out.append(f"TP {tp1}")
                if risk:
                    # Keep '%' so the renderer can tint by risk.
                    if risk.startswith("R"):
                        out.append(risk)
                    else:
                        out.append(f"R {risk}")
                return "\n".join([x for x in out if x])

            def _pretty_group_label(g: str) -> str:
                if g == "狙える":
                    return "☑ 狙える（今日やること）"
                if g == "見送り":
                    return "✕ 見送り"
                if g == "ポジ":
                    return "☑ ポジション"
                if g.startswith("ソーサー"):
                    return f"☑ {g}"
                return f"☑ {g}"

            def _build_img_rows(rows_src: list[list[str]]) -> list[list[str]]:
                img_rows: list[list[str]] = []
                current_group: str | None = None

                for r in rows_src:
                    group = str(r[0]) if r and len(r) > 0 else ""
                    if group != current_group:
                        current_group = group
                        img_rows.append([_pretty_group_label(group)])

                    idx = str(r[1]) if len(r) > 1 else ""
                    sym_main, sym_tags = _split_symbol_cell(str(r[2]) if len(r) > 2 else "")
                    sym_tags = _format_tags(sym_tags)
                    sym_cell = sym_main
                    if sym_tags:
                        sym_cell = f"{sym_main}\n{sym_tags}"

                    order_txt = _format_order_cell(str(r[3]) if len(r) > 3 else "")

                    sl = str(r[4]) if len(r) > 4 else ""
                    tp1 = str(r[5]) if len(r) > 5 else ""
                    risk = str(r[6]) if len(r) > 6 else ""
                    risk_block = _format_risk_block(sl, tp1, risk)

                    memo = _strip_icons(str(r[7]) if len(r) > 7 else "")
                    # Saucer pages: keep status compact (LINE wraps aggressively)
                    if group.startswith("ソーサー"):
                        memo = memo.replace("（注文有効）", "")
                        memo = memo.replace("注文有効", "")
                        memo = " ".join(memo.split())

                    img_rows.append([idx, sym_cell, order_txt, risk_block, memo])

                return img_rows

            # Split rows for multi-page PNG
            rows_orders = [r for r in table_rows if r and str(r[0]) in ("狙える", "見送り", "ポジ")]
            rows_saucer_d = [r for r in table_rows if r and str(r[0]) == "ソーサー（日足）"]
            rows_saucer_w = [r for r in table_rows if r and str(r[0]) == "ソーサー（週足）"]
            rows_saucer_m = [r for r in table_rows if r and str(r[0]) == "ソーサー（月足）"]

            style = TableImageStyle(
                max_total_px=1080,
                max_col_px=520,
                margin=24,
                pad_x=16,
                pad_y=14,
                font_size=30,
                title_font_size=42,
                section_font_size=34,
                line_width=2,
                line_spacing=4,
                header_bg="#F3F4F6",
                zebra_bg="#FAFAFA",
                section_bg="#DBEAFE",
                text_color="#111827",
                grid_color="#D1D5DB",
                wrap_cells=True,
                max_lines=6,
            )

            png_paths: list[str] = []

            # 1) Orders + Position (main)
            if rows_orders:
                img_headers = ["#", "銘柄", "注文", "SL/TP\nR", "メモ"]
                img_rows = _build_img_rows(rows_orders)
                title_orders = f"stockbotTOM {today_str} 注文サマリ"
                try:
                    # NOTE: `render_table_png` auto-detects risk columns from the header names
                    # (e.g. columns containing "Risk" or "SL/TP" + "R").
                    # Older drafts passed `risk_cols=...`, but the current `render_table_png`
                    # implementation does not accept that argument.
                    render_table_png(title_orders, img_headers, img_rows, png_main, style=style)
                    png_paths.append(png_main)
                except Exception as e:
                    if note_enabled:
                        lines.append("")
                        lines.append(f"🖼 表画像: 生成失敗（{e}）")

            # 2) Saucer (daily)
            if rows_saucer_d:
                img_headers = ["#", "銘柄", "注文", "SL/TP\nR", "状態"]
                img_rows = _build_img_rows(rows_saucer_d)
                title_d = f"stockbotTOM {today_str} ソーサー（日足）"
                try:
                    render_table_png(title_d, img_headers, img_rows, png_d, style=style)
                    png_paths.append(png_d)
                except Exception as e:
                    if note_enabled:
                        lines.append("")
                        lines.append(f"🖼 日足画像: 生成失敗（{e}）")

            # 3) Saucer (weekly)
            if rows_saucer_w:
                img_headers = ["#", "銘柄", "注文", "SL/TP\nR", "状態"]
                img_rows = _build_img_rows(rows_saucer_w)
                title_w = f"stockbotTOM {today_str} ソーサー（週足）"
                try:
                    render_table_png(title_w, img_headers, img_rows, png_w, style=style)
                    png_paths.append(png_w)
                except Exception as e:
                    if note_enabled:
                        lines.append("")
                        lines.append(f"🖼 週足画像: 生成失敗（{e}）")

            # 4) Saucer (monthly)
            if rows_saucer_m:
                img_headers = ["#", "銘柄", "注文", "SL/TP\nR", "状態"]
                img_rows = _build_img_rows(rows_saucer_m)
                title_m = f"stockbotTOM {today_str} ソーサー（月足）"
                try:
                    render_table_png(title_m, img_headers, img_rows, png_m, style=style)
                    png_paths.append(png_m)
                except Exception as e:
                    if note_enabled:
                        lines.append("")
                        lines.append(f"🖼 月足画像: 生成失敗（{e}）")

            if png_paths and note_enabled:
                lines.append("")
                for p in png_paths:
                    lines.append(f"🖼 表画像: {p}")

    # Debug footer (helps confirm which commit/version actually ran in GitHub Actions)
    if _env_truthy("REPORT_DEBUG", default=False):
        try:
            import sys
            build = os.environ.get("GITHUB_SHA", "")
            build = build[:7] if build else "-"
            pyver = sys.version.split()[0]
            try:
                import PIL  # type: ignore
                pillow = getattr(PIL, "__version__", "?")
            except Exception:
                pillow = "-"
            lines.append("")
            lines.append(f"🔧 build:{build} / py:{pyver} / pillow:{pillow}")
        except Exception:
            pass

    return "\n".join(lines).rstrip() + "\n"
