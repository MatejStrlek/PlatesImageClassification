# Dataset overview

This project uses the US license plates image classification dataset, which contains clean and labeled images of license plates from all 50 U.S. states, the District of Columbia, and several U.S. territories.

**Total samples**: 8,721 images

**Total classes**: 56 (balanced across all states and regions)

**Format**: JPG images with CSV metadata

Each image is associated with a record in the provided CSV file, which includes the following:

| Column     | Description                                              |
|------------|----------------------------------------------------------|
| `class id` | Integer identifier (0–55) for each class                 |
| `filepaths`| Relative path to the license plate image                 |
| `labels`   | Text label (e.g., `"TEXAS"`, `"HAWAI"`, `"GUAM"`)        |
| `data set` | Data split assignment: `train`, `valid`, or `test`       |

## Dataset splitting

The dataset is pre-divided into training, validation and test sets. The distribution is stratified to maintain equal class representation in each set.

| Split       | Percentage |
|-------------|------------|
| Train       | ~94%       |
| Validation  | ~3%        |
| Test        | ~3%        |

Each split contains samples from all 56 classes.

## Data inspection & cleaning

Before model training, dataset was checked for consistency and cleanless:

| Check                            | Result     |
|----------------------------------|------------|
| Missing labels or filepaths      | None     |
| Corrupted or unreadable images   | None     |
| Duplicate records                | None     |
| NA rows                          | None     |
| Class imbalance                  | None (balanced) |

No images or labels required removal or imputation. The dataset is clean and ready for modeling.

## Preprocessing pipeline

All images were preprocessed to ensure consistent size and normalization:

### Resize
Every image was resized to **224 × 128 pixels**, maintaining consistent dimensions for model input.

### Normalization
All image tensors were normalized to the `[-1, 1]` range using:

```python
transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```
