# 数据目录说明

比赛与本地实验用的多模态数据放在本目录下，分为 **训练/验证包** 与 **测试包** 两块目录（测试可与训练分离存放）。

## 目录一览

| 路径 | 内容 |
|------|------|
| `kaggle-dataset/` | `train.json`、`val.json`、训练/验证用的影像特征、radiomics、临床 CSV、原始报告等 |
| `new_test/` | `test.json`、`sample_submission.csv`、测试集对应的各模态文件 |

主键在所有文件中均为 **`case_id`**（字符串或整型需与 JSON 键一致）。

## 获取与放置数据

1. 从课程/Kaggle 提供的压缩包下载数据（若分批下发：先 train/val，后 test）。
2. 将 **训练/验证** 相关文件解压到 **`dataset/kaggle-dataset/`**，确保存在：
   - `train.json`、`val.json`
   - `radiomics_info/train/`、`radiomics_info/val/`
   - `clinical_information/train_patient_info.csv`、`val_patient_info.csv`
   - `image_features/`（与 JSON 中 `image_path` 一致）
3. 将 **测试集** 解压到 **`dataset/new_test/`**，确保存在：
   - `test.json`
   - `sample_submission.csv`
   - `radiomics_info/test/*.csv`
   - `clinical_information/test/test_patient_info.csv`（注意为子文件夹 `test/`）
   - `image_features/`（结构同训练侧）
4. 在仓库根目录或 `notebooks/` 下启动 Jupyter；`pipeline_macro.ipynb` / `pipeline_micro.ipynb` 会自动向上查找 `dataset/kaggle-dataset/train.json` 并配对同级的 `dataset/new_test/test.json`。

若你更喜欢**单一目录**（历史布局），也可把 `test.json` 与测试相关子目录直接放在 `kaggle-dataset/` 下，与 `train.json` 并列；此时无需 `new_test/` 文件夹。

## 更细的字段说明

参见同目录内：

- `kaggle-dataset/README.md` — JSON 字段、`case_id`、各子文件夹约定
- `kaggle-dataset/radiomics_info/README.md` — 各 radiomics 列含义
