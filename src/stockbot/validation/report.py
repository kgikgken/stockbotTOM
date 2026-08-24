"""L1 レポート（DESIGN.md §10.2/§10.3 / TASKS.md T-405）。

validation.layer1.run_layer1() が output_dir に書き出した表（csv）を1つの
markdown に束ねる。冒頭に「検定数」「ホールドアウト未使用」を明記する。
DESIGN.md §10.3 の撤退基準の判断に使う項目（総合V1/V2/V3のIC・t値、分位曲線、
ベースライン(c)との比較、切り口別）が一通り載るようにする。

CLAUDE.md「検証結果を解釈しない」に従い、数値の解釈・採否判断はしない。表を
並べるだけ。図（チャート画像）は作らない（このリポジトリはプロット用ライブラリに
依存していないため。数値表で同じ情報を確認できる）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .layer1 import CUT_DEFINITIONS, VARIANTS, TestCounter, _df_to_markdown


def _read_csv(output_dir: Path, name: str) -> pd.DataFrame:
    p = Path(output_dir) / f"{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        # 列が1つも無い表（例: 対象特徴量が無かった0_sanity_distribution）はヘッダ行すら
        # 無い0バイトのcsvになる。読み込めないだけで「表が空」という正しい結果なので
        # 空のDataFrameとして扱う
        return pd.DataFrame()


def build_l1_report(
    layer1_output_dir: Path, output_path: Path,
    test_counter: Optional[TestCounter] = None,
    window_label: str = "",
    calibration_report_path: Optional[Path] = None,
    holdout_used: bool = False,
) -> Path:
    """T-403 の出力（layer1_output_dir 以下の csv）を1つの markdown にまとめる。

    test_counter を渡せばその件数を、渡さなければ 5_test_count_log.csv の行数を
    「検定数」として使う。holdout_used は必ず明示的に渡す（既定 False にはしない
    設計も検討したが、呼び出し側の意図を毎回明記させるため引数自体は残しつつ、
    実態としてこのリポジトリのどの呼び出し経路もホールドアウトを読む手段を持たない
    ―― True を渡すこと自体が誤りである）。
    """
    layer1_output_dir = Path(layer1_output_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    test_count_log = _read_csv(layer1_output_dir, "5_test_count_log")
    n_tests = test_counter.count if test_counter is not None else len(test_count_log)

    lines: list[str] = []
    lines.append("# L1 レポート（DESIGN.md §10.2 / TASKS.md T-405）")
    lines.append("")
    lines.append(f"- 検定数: {n_tests}")
    lines.append(f"- ホールドアウト使用: {'あり' if holdout_used else 'なし'}")
    if window_label:
        lines.append(f"- 対象期間: {window_label}")
    lines.append("")
    lines.append("数値の解釈・採否判断はしない（CLAUDE.md）。継続/見直し/撤退は設計責任者が"
                 "DESIGN.md §10.3 の基準に照らして判断する。以下は判断材料の表のみ。")
    lines.append("")

    lines.append("## 0. 健全性")
    lines.append("")
    lines.append("### 欠損率")
    lines.append(_df_to_markdown(_read_csv(layer1_output_dir, "0_sanity_missing_rate")))
    lines.append("### 相関が高い組（|ρ|>0.7、除外はしていない）")
    lines.append(_df_to_markdown(_read_csv(layer1_output_dir, "0_sanity_high_correlation_pairs")))

    lines.append("## 1. 単変量（特徴量ごとの10分位/値ごと、h=10）")
    lines.append("")
    lines.append(_df_to_markdown(_read_csv(layer1_output_dir, "1_univariate_deciles")))

    lines.append("## 2. 次元スコアIC（D1〜D7、日次Spearman IC → Newey-West平均・t、Q5-Q1）")
    lines.append("")
    lines.append(_df_to_markdown(_read_csv(layer1_output_dir, "2_dimension_ic")))

    lines.append("## 3. 総合 V1/V2/V3")
    lines.append("")
    lines.append("### IC・Q5-Q1・分位別成功率")
    lines.append(_df_to_markdown(_read_csv(layer1_output_dir, "3_composite_ic")))
    for v in VARIANTS:
        lines.append(f"### {v} の10分位曲線（r_10平均・成功率）")
        lines.append(_df_to_markdown(_read_csv(layer1_output_dir, f"3_composite_{v}_deciles")))
    for cut in CUT_DEFINITIONS:
        lines.append(f"### 切り口別: {cut}")
        for v in VARIANTS:
            df = _read_csv(layer1_output_dir, f"3_composite_by_{v}_{cut}")
            if len(df):
                lines.append(f"#### {v}")
                lines.append(_df_to_markdown(df))

    lines.append("## 4. ベースライン (a) SMA200ランダム抽出 / (b) 西村ルール / (c) 押し目等加重")
    lines.append("")
    lines.append(_df_to_markdown(_read_csv(layer1_output_dir, "4_baseline_comparison")))
    if calibration_report_path is not None and Path(calibration_report_path).exists():
        lines.append(f"(b) の詳細（期間別・出典との差分）は "
                     f"`{Path(calibration_report_path).name}` を参照（T-404）。")
        lines.append("")

    lines.append("## 5. 多重検定ログ")
    lines.append("")
    lines.append(_df_to_markdown(test_count_log))

    lines.append("## 6. 生存バイアス上限評価（SPEC §4）")
    lines.append("")
    lines.append(_df_to_markdown(_read_csv(layer1_output_dir, "6_survivorship_note")))

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main(argv: Optional[list] = None) -> int:
    """python -m stockbot.validation.report --layer1-dir ... [--output ...] で実行する。
    validation.layer1.main() が書いた出力を束ねる。
    """
    import argparse

    ap = argparse.ArgumentParser(prog="python -m stockbot.validation.report")
    ap.add_argument("--layer1-dir", required=True, help="validation.layer1 の出力ディレクトリ")
    ap.add_argument("--output", default=None,
                    help="既定: src/stockbot/validation/reports/l1_report.md")
    ap.add_argument("--window-label", default="")
    ap.add_argument("--calibration-report", default=None)
    args = ap.parse_args(argv)

    default_path = Path(__file__).resolve().parent / "reports" / "l1_report.md"
    output_path = Path(args.output) if args.output else default_path
    calib_path = Path(args.calibration_report) if args.calibration_report else None
    build_l1_report(args.layer1_dir, output_path, window_label=args.window_label,
                    calibration_report_path=calib_path, holdout_used=False)
    print(f"[report] {output_path} に出力")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
