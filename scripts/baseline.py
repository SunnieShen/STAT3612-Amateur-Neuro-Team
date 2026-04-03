"""
最小可运行 baseline 示例：
- 只用结构化 + radiomics 特征
- 模型：LogisticRegression 带 class_weight='balanced'

运行方式（在仓库根目录）：
    python scripts/baseline.py

运行前请确保：
    pip install pandas numpy scikit-learn
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import merge_all_sources


def build_feature_table(split: str) -> pd.DataFrame:
    """
    基于 merge_all_sources 的结果，选择一批简单可用的特征列。
    你可以根据 EDA 结果再调整特征列表。
    """
    df = merge_all_sources(split)

    # 简单示例：选取所有 radiomics 数值 + 一些基础临床特征
    radiomics_cols = [c for c in df.columns if "__rad_" in c or "rad_firstorder" in c or "rad_glcm" in c]

    basic_num = []
    for c in ["Age", "age"]:
        if c in df.columns:
            basic_num.append(c)

    cat_cols = []
    for c in [
        "Sex",
        "sex",
        "Tumor Location",
        "Signal Intensity (T1)",
        "Signal Intensity (T1c)",
        "Signal Intensity (T2)",
        "Signal Intensity (T2-FLAIR)",
    ]:
        if c in df.columns:
            cat_cols.append(c)

    feature_cols: List[str] = radiomics_cols + basic_num + cat_cols
    feature_cols = sorted(set(feature_cols))

    print(f"[{split}] 使用特征列数: {len(feature_cols)}")

    keep_cols = ["case_id"] + feature_cols
    if "Overall_class" in df.columns:
        keep_cols.append("Overall_class")

    return df[keep_cols].copy()


def build_preprocessor(X: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor


def train_and_eval():
    train_df = build_feature_table("train")
    val_df = build_feature_table("val")

    label_col = "Overall_class"

    X_train = train_df.drop(columns=[label_col, "case_id"])
    y_train = train_df[label_col]

    X_val = val_df.drop(columns=[label_col, "case_id"])
    y_val = val_df[label_col]

    # 数值和类别列自动分拆
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    print("数值特征列数:", len(num_cols), "类别特征列数:", len(cat_cols))

    preprocessor = build_preprocessor(X_train, num_cols, cat_cols)

    clf = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    print("开始训练 baseline 模型...")
    model.fit(X_train, y_train)

    print("在 val 集上评估...")
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")

    print(f"Val Accuracy = {acc:.4f}")
    print(f"Val F1-macro = {f1_macro:.4f}")

    return model


def predict_test_and_save(model, output_path: str = "outputs/sample_submission_v1.csv") -> None:
    test_df = build_feature_table("test")
    X_test = test_df.drop(columns=["case_id"], errors="ignore")

    num_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_test.columns if c not in num_cols]

    # 注意：这里假设 train/val/test 的列集一致；实际使用中建议与训练阶段统一。
    print("在 test 集上生成预测...")
    y_test_pred = model.predict(X_test)

    sub = pd.DataFrame(
        {
            "case_id": test_df["case_id"].values,
            "Overall_class": y_test_pred,
        }
    )
    sub.to_csv(output_path, index=False)
    print(f"提交文件已保存到: {output_path}")


if __name__ == "__main__":
    mdl = train_and_eval()
    predict_test_and_save(mdl)

