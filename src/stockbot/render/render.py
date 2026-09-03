"""HTML → PNG（docs/SCREENER.md §4.5）。

Jinja2 でテンプレートを HTML にし、Playwright（headless Chromium）で 1枚目・2枚目を
それぞれ PNG にする。SPEC.md §5 の方式（HTML/CSS を Chromium に描かせる）に従う。
Pillow で座標を手計算するより、日本語の折返しとカード配置を CSS に任せられる。

**描画は失敗しうる前提で書く。** Chromium が無い・フォントが無い・起動に失敗する、の
いずれでも例外を投げるだけにして、テキスト配信への切り替えは呼び出し側（cli.step_notify）
が行う（§4.4）。ここで握り潰すと、画像が出ていない理由が分からなくなる。

jinja2 と playwright は関数内で遅延 import する（テストと DRYRUN では不要）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .context import build_context

TEMPLATE_NAME = "template.html"
PAGE_IDS = ("page1", "page2")
VIEWPORT = {"width": 1000, "height": 1400}
SCALE = 2          # Retina 相当。LINE で読める解像度にする
TIMEOUT_MS = 60000


def chromium_executable() -> Optional[str]:
    """Playwright が既定で使う Chromium が無い環境向けの明示パス。

    `CHROMIUM_PATH` があればそれを使う。無ければ `PLAYWRIGHT_BROWSERS_PATH` の直下の
    `chromium` を見る。Playwright は自分のバージョンに紐づくリビジョンを探すため、
    ブラウザが先に入っている環境（バージョンがずれる）では既定の起動が失敗する。
    CI では `playwright install chromium` が一致するリビジョンを入れるので None でよい。
    """
    explicit = os.getenv("CHROMIUM_PATH", "").strip()
    if explicit:
        return explicit if Path(explicit).exists() else None
    base = os.getenv("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if base:
        candidate = Path(base) / "chromium"
        if candidate.exists():
            return str(candidate)
    return None


def _asset_url(path: Path) -> Optional[str]:
    """ローカル資産を file:// URL にする（無ければ None でテンプレート側が省く）。"""
    path = Path(path)
    return path.resolve().as_uri() if path.exists() else None


def render_html(delivered: Optional[pd.DataFrame], summary: dict,
                assets_dir: Optional[Path] = None,
                template_dir: Optional[Path] = None) -> str:
    """配信記録と要約から HTML を作る（PNG 化はしない）。

    ブラウザを使わないので、表示内容のテストはここまでで完結する（§4.3）。
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    template_dir = Path(template_dir or Path(__file__).parent)
    assets_dir = Path(assets_dir or Path(__file__).resolve().parents[3])
    env = Environment(loader=FileSystemLoader(str(template_dir)),
                      autoescape=select_autoescape(["html"]))
    template = env.get_template(TEMPLATE_NAME)
    return template.render(
        ctx=build_context(delivered, summary),
        hero=_asset_url(assets_dir / "hero.jpeg"),
        mascot=_asset_url(assets_dir / "mascot.png"),
    )


def html_to_pngs(html: str, out_dir: Path, stem: str) -> list[Path]:
    """HTML の #page1 / #page2 をそれぞれ PNG にする。

    要素単位のスクリーンショットなので、カード枚数でページの高さが変わっても
    切れずに収まる。
    """
    from playwright.sync_api import sync_playwright

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = out_dir / f"{stem}.html"
    source.write_text(html, encoding="utf-8")

    paths: list[Path] = []
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:
            # 既定のリビジョンが無い環境では、入っている Chromium を明示して再試行する
            executable = chromium_executable()
            if not executable:
                raise
            browser = p.chromium.launch(executable_path=executable)
        try:
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=SCALE)
            page.set_default_timeout(TIMEOUT_MS)
            page.goto(source.resolve().as_uri(), wait_until="load")
            for i, page_id in enumerate(PAGE_IDS, start=1):
                target = out_dir / f"{stem}_{i}.png"
                page.locator(f"#{page_id}").screenshot(path=str(target))
                paths.append(target)
        finally:
            browser.close()
    return paths


def render_images(delivered: Optional[pd.DataFrame], summary: dict, out_dir: Path,
                  stem: str = "screen", assets_dir: Optional[Path] = None,
                  template_dir: Optional[Path] = None) -> list[Path]:
    """配信記録と要約から PNG 2枚を作る。失敗は例外のまま呼び出し側へ返す。"""
    html = render_html(delivered, summary, assets_dir=assets_dir, template_dir=template_dir)
    return html_to_pngs(html, out_dir, stem)
