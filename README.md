# stockbotTOM v8 — 日本株 順張り押し目スクリーナー

仕様は `docs/SPEC.md`、調査は `docs/RESEARCH.md`。本リポジトリはモジュール単位で段階的に構築する。

| モジュール | 内容 | 状態 |
|---|---|---|
| M1 | データ基盤（取得・整合性検査・保存・スナップショット・ユニバース） | 完了 |
| M2 | 特徴量（トレンド・押し目・出来高・収縮・相対力・週足） | 未着手 |
| M3 | スコアリング | 未着手 |
| M4 | 検証 L1 / L2 | 未着手 |
| M5 | 描画（既存デザイン踏襲）・LINE 配信 | 未着手 |

## セットアップ

```bash
python -m pip install -r requirements.txt
pip install -e .
python -m unittest discover -s tests -t .        # ネットワーク不要
