# v10では main.py が rr.py を使わない構成（scoringのTP/SLからRR算出）
# 既存 import が残っても落ちないように置いておく互換ファイル

def compute_rr(*args, **kwargs):
    return {"rr": 0.0, "entry": 0.0, "tp_pct": 0.0, "sl_pct": 0.0}