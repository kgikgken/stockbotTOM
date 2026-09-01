"""LINE 配信（docs/SCREENER.md §4）。

- message.py   … 配信本文の組み立て。入力は配信記録（delivered_*.csv）と
                 その日の要約（screen_summary_*.json）だけ
- line_send.py … Cloudflare Worker 経由で LINE に push する
"""
