# Dataset

## Key concept: `case_id` (global primary key)

Across **all** provided files, the most important identifier is **`case_id`**:

- In `train.json / val.json / test.json`, the **top-level JSON key** (e.g. `"2146"`) is the `case_id`.
- In CSV files under `radiomics_info/` and `report_info/`, there is a column named `case_id`.

When joining different modalities/features/reports, always join on **`case_id`**.

---

## Files

### `train.json` / `val.json` / `test.json`

Each file is a JSON **dictionary**:

- **key**: `case_id` (string, e.g. `"2146"`)
- **value**: a record describing this case

Common fields in each record:

- **`modality`**: list of modality names for this case (e.g. `["ax t2f", "ax t2", "ax t1c+", "ax t1"]`)
- **`available_modalities`**: list of modalities actually available
- **`image_path`**: list of relative paths to `.npy` feature files (aligned with modalities)
- **`report`**: radiology report text (string)
- **`Sex`**, **`Age`**: basic demographics (strings)

Label field:

- **`Overall_class`**: **only present in `train.json` and `val.json`** (this is the classification label)
- In **`test.json` there is no `Overall_class`** (no labels provided to students)

Minimal example (from `train.json`):

```json
{
  "2146": {
    "modality": ["ax t1", "ax t1c+", "ax t2"],
    "Overall_class": "Meningioma",
    "available_modalities": ["ax t1", "ax t1c+", "ax t2"],
    "image_path": [
      "image_features_filtered/2146/ax_t1/image.npy",
      "image_features_filtered/2146/ax_t1c/image.npy",
      "image_features_filtered/2146/ax_t2/image.npy"
    ],
    "report": "...",
    "Sex": "female",
    "Age": "51"
  }
}
```

### `sample_submission.csv`

Kaggle submission template:

- **`case_id`**: test case id
- **`Overall_class`**: your predicted class for that `case_id`

---

## Folders

### `image_features/`

Per-case feature arrays saved as NumPy `.npy`.
Paths referenced by `image_path` are relative to this dataset folder.

### `radiomics_info/`

Radiomics features in CSV format, organized by split:

- `radiomics_info/train/*.csv`
- `radiomics_info/val/*.csv`
- `radiomics_info/test/*.csv`

All radiomics CSVs contain a `case_id` column for joining.

### `report_info/`

Structured patient/report info in CSV format, organized by split:

- `report_info/train_patient_info.csv`
- `report_info/val_patient_info.csv`
- `report_info/test_patient_info.csv`

All report CSVs contain a `case_id` column for joining.

---

## Recommended workflow

1. Load `train.json` (or `val.json`) as the base table.
2. For each `case_id`, use `image_path` to load the `.npy` features.
3. Optionally join additional features from:
   - `radiomics_info/<split>/*.csv`
   - `report_info/<split>_patient_info.csv`
4. Train on `train.json`, validate on `val.json`, and predict `Overall_class` for `test.json`.
