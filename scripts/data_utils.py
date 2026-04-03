import os
import json
from typing import List

import numpy as np
import pandas as pd


DATA_DIR = "kaggle-dataset"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去掉列名里的 BOM / 空格，并统一为字符串。"""
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def load_split_json(split: str) -> pd.DataFrame:
    """
    读取 train/val/test.json 并转为 DataFrame。

    - 顶层 key 作为 case_id 列。
    - 每行一个样本。
    """
    path = os.path.join(DATA_DIR, f"{split}.json")
    with open(path, "r") as f:
        d = json.load(f)

    rows = []
    for case_id, rec in d.items():
        row = {"case_id": str(case_id)}
        row.update(rec)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def load_clinical(split: str) -> pd.DataFrame:
    """读取 clinical_information 下的结构化临床信息。"""
    path = os.path.join(DATA_DIR, "clinical_information", f"{split}_patient_info.csv")
    df = pd.read_csv(path)
    df = _clean_columns(df)
    if "case_id" not in df.columns:
        raise KeyError(f"'case_id' column not found in {path}, got: {df.columns}")
    df["case_id"] = df["case_id"].astype(str)
    return df


def load_raw_report(split: str) -> pd.DataFrame:
    """读取 original_raw_report 下的原始报告文本。"""
    path = os.path.join(DATA_DIR, "original_raw_report", f"{split}_patient_info.csv")
    df = pd.read_csv(path)
    df = _clean_columns(df)
    if "case_id" not in df.columns:
        raise KeyError(f"'case_id' column not found in {path}, got: {df.columns}")
    df["case_id"] = df["case_id"].astype(str)
    return df


def load_radiomics_modality(split: str, modality: str) -> pd.DataFrame:
    """
    读取某个模态的 radiomics 特征表，并给特征列加前缀：
    - 输入 modality: ax_t1 / ax_t1c / ax_t2 / ax_t2f
    - 输出列名形如: ax_t1__rad_firstorder_Mean
    """
    filename = f"{modality}_radiomics_{split}.csv"
    path = os.path.join(DATA_DIR, "radiomics_info", split, filename)
    df = pd.read_csv(path)
    df = _clean_columns(df)
    if "case_id" not in df.columns:
        raise KeyError(f"'case_id' column not found in {path}, got: {df.columns}")
    df["case_id"] = df["case_id"].astype(str)

    # 给非 case_id 列加前缀，避免冲突
    rename_map = {
        c: f"{modality}__{c}"
        for c in df.columns
        if c != "case_id"
    }
    df = df.rename(columns=rename_map)
    return df


def merge_all_sources(split: str, radiomics_modalities: List[str] = None) -> pd.DataFrame:
    """
    按 case_id 将 JSON + clinical + raw_report + radiomics 横向拼接成一个大表。

    radiomics_modalities: 要加载的模态列表，默认 4 个模态全用。
    """
    if radiomics_modalities is None:
        radiomics_modalities = ["ax_t1", "ax_t1c", "ax_t2", "ax_t2f"]

    base = load_split_json(split)

    clinical = load_clinical(split)
    raw_report = load_raw_report(split)

    merged = base.merge(clinical, on="case_id", how="left")
    merged = merged.merge(raw_report, on="case_id", how="left")

    for m in radiomics_modalities:
        r = load_radiomics_modality(split, m)
        merged = merged.merge(r, on="case_id", how="left")

    return merged


def fix_image_path(rel_path: str) -> str:
    """
    将 JSON 中的 image_path 映射到本地实际路径。

    JSON 里通常是: image_features/2146/ax_t1/image.npy
    本地结构是:  kaggle-dataset/image_features/image_features/2146/ax_t1/image.npy
    """
    if not isinstance(rel_path, str):
        raise ValueError(f"rel_path must be str, got {type(rel_path)}")
    fixed = rel_path.replace("image_features/", "image_features/image_features/", 1)
    return os.path.join(DATA_DIR, fixed)


def load_image_feature_vector(rel_path: str) -> np.ndarray:
    """根据 JSON 中的 image_path 读取单个 .npy 向量。"""
    full = fix_image_path(rel_path)
    return np.load(full)

