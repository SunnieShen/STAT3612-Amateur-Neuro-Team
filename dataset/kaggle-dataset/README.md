# Dataset（训练/验证侧）

本文件夹为 **训练与验证** 数据的主目录；若官方将测试集单独打包，测试集 JSON 与文件在同级目录 **`../new_test/`** 中（见上方 `dataset/README.md`）。若本地合并为一份数据，也可将 `test.json` 放在本目录下。

## 主键：`case_id`

所有文件以 **`case_id`** 对齐：

- 在 `train.json` / `val.json` / `test.json`（若存在）中，**顶层 JSON 的键**即 `case_id`（如 `"2146"`）。
- `radiomics_info/` 下各 CSV 含 `case_id` 列。

合并不同模态时务必按 **`case_id`** 连接，禁止按行号拼接。

---

## 文件：`train.json` / `val.json` / `test.json`

每个文件为一个字典：

- **键**：`case_id`（字符串）
- **值**：该病例的记录

常用字段：

- **`modality`**：该病例涉及的序列名称列表  
- **`available_modalities`**：实际有文件的序列  
- **`image_path`**：相对于本数据目录的 `.npy` 深度特征路径列表（与 `modality` 对齐）  
- **`report`**：影像学描述文本  
- **`Sex`**、**`Age`**：人口学信息（字符串）

标签：

- **`Overall_class`**：仅在 **train / val** 的 JSON 中出现；测试集无标签。

示例结构（节选）：

```json
{
  "2146": {
    "modality": ["ax t1", "ax t1c+", "ax t2"],
    "Overall_class": "Meningioma",
    "available_modalities": ["ax t1", "ax t1c+", "ax t2"],
    "image_path": [
      "image_features/2146/ax_t1/image.npy",
      "image_features/2146/ax_t1c/image.npy",
      "image_features/2146/ax_t2/image.npy"
    ],
    "report": "...",
    "Sex": "female",
    "Age": "51"
  }
}
```

## `sample_submission.csv`

（若测试集在 `new_test/`：使用该目录下的模板。）

- `case_id`
- `Overall_class`：提交时填入预测类别

---

## 子文件夹

### `image_features/`

按病例与序列存放 ResNet 等深度特征的 `.npy` 文件；路径需与 JSON 中的 `image_path` 一致。仓库内 notebook 在加载时还会使用一层 `image_features/image_features/<case_id>/<seq>/image.npy` 的实际目录（与压缩包目录结构一致）。

### `radiomics_info/`

按 split 分子目录：

- `radiomics_info/train/*.csv`
- `radiomics_info/val/*.csv`
- 测试集通常在 **`../new_test/radiomics_info/test/`**

各 CSV 含 `case_id`，列说明见 `radiomics_info/README.md`。

### `clinical_information/`

结构化临床与报告衍生字段：

- `train_patient_info.csv`
- `val_patient_info.csv`
- 测试集若在 `new_test/`：**`clinical_information/test/test_patient_info.csv`**（多一层 `test` 子目录）

### `original_raw_report/`

原始放射学报告文本 CSV（与 `clinical_information` 并行， train/val 为 `*_patient_info.csv`；测试集路径规则类似临床侧）。

---

## 推荐流程

1. 以 `train.json` / `val.json` 为主表载入。  
2. 按 `image_path` 读取 `image_features`。  
3. 需要时合并 `radiomics_info/<split>/` 与 `clinical_information` 或 `original_raw_report`。  
4. 在 `train` 上训练，`val` 上调参；对 **`../new_test/test.json`**（或同目录下的 `test.json`）生成预测并参照 `sample_submission.csv` 提交。
