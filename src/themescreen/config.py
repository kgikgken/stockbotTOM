"""Paths of the theme version. Everything it writes lives under data/theme/.

It does not import src/stockbot/ and never writes to data/daily/ (CLAUDE.md,
"このリポジトリには2つのプロジェクトがある"; docs/theme/README.md).
"""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
THEME_DIR: Path = ROOT / "data" / "theme"
UNIVERSE_DIR: Path = THEME_DIR / "universe"
RAW_DIR: Path = THEME_DIR / "raw"
OUT_DIR: Path = THEME_DIR / "out"
EDINET_DIR: Path = THEME_DIR / "edinet"
