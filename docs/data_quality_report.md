## Data Quality Report（草稿模板）

本文件由 `scripts/build_table.py` 的输出 + 手工补充说明组成。

### 1. 基本信息

- **train 样本数**: （运行脚本后填写）
- **val 样本数**: （运行脚本后填写）
- **test 样本数**: （运行脚本后填写）
- **主键列**: `case_id`（各表是否一一对应、是否有缺失）

### 2. 各 split 统一表信息（来自 build_table.py）

#### 2.1 train

- shape:  
- case_id 唯一个数:  
- 缺失率最高列（前 20）：  
  ```text
  （将脚本输出复制到这里）
  ```
- 标签分布 Overall_class:  
  ```text
  （将脚本输出复制到这里）
  ```

#### 2.2 val

- shape:  
- case_id 唯一个数:  
- 缺失率最高列（前 20）：  
  ```text
  ```
- 标签分布 Overall_class:  
  ```text
  ```

#### 2.3 test

- shape:  
- case_id 唯一个数:  
- 缺失率最高列（前 20）：  
  ```text
  ```

### 3. 关键字段质量摘要

- **Age / age**
  - 缺失率（train/val/test）：  
  - 建议填补策略（例如：从 clinical_information 覆盖；剩余用中位数填充等）

- **Sex / sex**
  - 缺失率（train/val/test）：  
  - 类别分布：  
  - 特殊取值（如 unknown/None）的处理建议：

- **Tumor Location**
  - Top-N 位置：  
  - 是否存在明显拼写错误/多种写法（如 Left frontal vs Left frontal lobe）：

- **Radiomics 特征**
  - 是否存在异常极大/极小值：  
  - 是否存在常数特征（方差≈0）：  
  - 是否需要标准化/归一化：

### 4. ID 对齐与缺失情况

- JSON vs clinical_information
  - 各 split 中 `case_id` 是否完全对齐？  
  - JSON 中是否有找不到对应临床行的样本？数量：

- JSON vs original_raw_report
  - 是否存在缺失 raw_report 的样本？数量：

### 5. 结论与风险

- **可用性结论**
  - 哪些字段质量较高，适合作为主要特征？
  - 哪些字段问题较多，只能作为辅助特征或需要重加工？

- **风险点**
  - 是否存在潜在标签泄漏字段？  
  - 是否存在某些类别样本极少的问题？  
  - 后续建模时需要特别注意的点：

