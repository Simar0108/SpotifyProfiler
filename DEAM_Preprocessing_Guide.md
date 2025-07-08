# 🧹 DEAM Dataset Preprocessing Task Guide

This guide outlines the preprocessing steps needed to prepare the DEAM dataset for training a mood classification neural network. It is optimized to be used with AI-assisted coding tools like Cursor.

---

## 🎯 Objective

Create a training-ready dataset by merging DEAM's audio features with averaged valence/arousal annotations, and optionally generate discrete mood labels for classification.

---

## ✅ Task Breakdown

### 1. Load Song-Level Audio Features
- Source: `features/audio_features/`
- Action:
  - Load all relevant CSVs.
  - Merge them into a single `DataFrame` using `song_id` or consistent indexing.
  - Drop any duplicate or NaN rows.

---

### 2. Load Averaged Valence & Arousal Labels
- Source: `annotations/averaged_annotations/song_level/static_annotations_averaged_songs_*.csv`
- Action:
  - Extract `song_id`, `valence_mean`, and `arousal_mean` columns.
  - This will serve as your target label(s).

---

### 3. Merge Features with Labels
- Join the features `DataFrame` with the labels using `song_id` or `filename`.
- Drop rows with missing or misaligned values.

---

### 4. (Optional) Generate Discrete Mood Labels
- Create a new column `mood_class` using this logic:

```python
def mood_label(valence, arousal):
    if valence >= 0.5 and arousal >= 0.5:
        return "Energetic"
    elif valence >= 0.5 and arousal < 0.5:
        return "Chill"
    elif valence < 0.5 and arousal >= 0.5:
        return "Tense"
    else:
        return "Melancholic"
```

---

### 5. Save the Final Processed Dataset
- Output file: `processed_dataset.csv`
- Columns should include:
  - Feature columns (e.g. `tempo`, `mfcc_1`, `rms`, ...)
  - `valence`, `arousal`
  - (Optional) `mood_class`

---

## 📦 Final Output Format

| tempo | mfcc_1 | ... | valence | arousal | mood_class |
|-------|--------|-----|---------|---------|------------|
| 120.0 | 0.123  | ... | 0.71    | 0.62    | Energetic  |

---

## 💡 Notes
- Normalize or standardize features if needed during model training.
- Save as `.csv` or `.parquet` for fast loading later.
- Ensure reproducibility by setting a random seed for any data splits.

---
