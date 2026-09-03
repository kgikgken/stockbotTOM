"""画像カードの描画（docs/SCREENER.md §4.5）。

- context.py  … 配信記録とその日の要約から、表示に必要な値だけを組み立てる（純粋関数）
- template.html … Jinja2 テンプレート。1枚目=詳細、2枚目=サマリー
- render.py   … テンプレートを HTML にし、Playwright で PNG 2枚にする

**入力は `delivered_*.csv` と `screen_summary_*.json` だけ。** 株価も指標もここでは
計算し直さない（§4.3）。描画に失敗したらテキスト配信に落ちる（§4.4）。
"""
