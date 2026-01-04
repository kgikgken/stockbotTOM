# stockbotTOM - Swing Screener (1-7 days)
from __future__ import annotations

import os

from utils.setup import Config, load_config
from utils.util import safe_print
from utils.events import load_events
from utils.screener import run_screening
from utils.report import build_line_report
from utils.line import send_line_via_worker


def main() -> int:
    cfg: Config = load_config()

    events = load_events(cfg.EVENTS_PATH)

    result = run_screening(
        universe_path=cfg.UNIVERSE_PATH,
        positions_path=cfg.POSITIONS_PATH,
        events=events,
        cfg=cfg,
    )

    report = build_line_report(result, cfg=cfg)
    safe_print(report)

    worker_url = os.getenv("WORKER_URL", "").strip()
    if worker_url:
        ok, msg = send_line_via_worker(worker_url, report, timeout=cfg.WORKER_TIMEOUT_SEC)
        safe_print(f"[LINE] ok={ok} msg={msg}")
    else:
        safe_print("[LINE] WORKER_URL not set. (Skipped sending)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())