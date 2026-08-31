# stockbotTOM v8 — 日本株 順張り押し目スクリーナー

仕様は `docs/SCREENER.md`（運用ツール）と `docs/SPEC.md`（決定事項）、調査は `docs/RESEARCH.md`。

検証プロジェクト（v8 順張り押し目）は 2026-08-30 に終了した（`docs/CLOSING.md`）。
現在は運用ツールを作っている。`docs/DESIGN.md` と `docs/TASKS.md` は終了済みの記録で、編集しない。

| 段 | 内容 | 状態 |
|---|---|---|
| データ基盤 | 取得・整合性検査・保存・スナップショット・ユニバース（`data-daily` が毎営業日実行） | 完了 |
| 記録 | 配信記録と 5 営業日後の結果付け（`docs/SCREENER.md` §3） | 完了 |
| 条件 | 19 条件のブール判定（`docs/SCREENER.md` §2） | 未着手（一覧待ち・§6 Q-1） |
| 描画・配信 | 既存デザイン踏襲・LINE 配信 | 未着手 |

```bash
python -m stockbot.cli daily      # 取得 → 保存 → スナップショット → ユニバース → 日次特徴量
python -m stockbot.cli resolve    # 配信記録に 5 営業日後の結果を付ける
```

## セットアップ

```bash
python -m pip install -r requirements.txt
pip install -e .
python -m unittest discover -s tests -t .        # ネットワーク不要
