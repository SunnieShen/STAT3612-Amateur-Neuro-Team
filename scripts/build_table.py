"""
构建并审计 train/val/test 的统一样本级大表。

运行方式（在仓库根目录）：
    python scripts/build_table.py

建议先运行本脚本，再开始做各自 EDA。
"""

from typing import List

import pandas as pd

from data_utils import merge_all_sources


def audit_split(split: str, top_missing: int = 20) -> pd.DataFrame:
    df = merge_all_sources(split)

    print(f"\n[{split}] 统一表信息")
    print("-" * 40)
    print("shape:", df.shape)
    print("case_id 行数:", len(df), "唯一个数:", df["case_id"].nunique())

    # 缺失率
    miss = df.isna().mean().sort_values(ascending=False).head(top_missing)
    print("\n缺失率最高的前若干列：")
    print(miss.to_string())

    # 标签分布（仅 train/val 有）
    if "Overall_class" in df.columns:
        print("\n标签分布 Overall_class：")
        print(df["Overall_class"].value_counts(dropna=False))

    return df


def main(splits: List[str] = None) -> None:
    if splits is None:
        splits = ["train", "val", "test"]

    for s in splits:
        _ = audit_split(s)


if __name__ == "__main__":
    main()

