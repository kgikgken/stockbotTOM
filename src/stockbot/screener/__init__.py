"""運用ツール（順張り押し目スクリーナー）の記録まわり（docs/SCREENER.md）。

終了した検証プロジェクト（docs/DESIGN.md・docs/TASKS.md・docs/CLOSING.md）とは
別立ての運用ツールである。t 値も撤退基準も持たず、代わりに配信の実績を貯める。

- record.py   … 配信記録の作成と保存。T（判定日）の引けまでのデータしか見ない
- resolver.py … 5 営業日後の結果付け。T+1 以降のデータしか見ない
"""
