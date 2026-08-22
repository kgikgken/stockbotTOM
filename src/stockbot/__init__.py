"""stockbotTOM v8 — 日本株 順張り押し目スクリーナー。

モジュール構成（段階的に追加）:
  M1 data/ universe/ config cli   … データ基盤（本モジュール）
  M2 features/                     … 特徴量
  M3 scoring/                      … スコアリング
  M4 validation/                   … 検証 L1/L2
  M5 render/ notify/               … 描画・LINE 配信
"""
__version__ = "0.2.0"
