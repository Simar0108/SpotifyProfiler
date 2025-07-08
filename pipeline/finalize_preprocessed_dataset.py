#!/usr/bin/env python3
"""
Finalize Preprocessed Dataset for DEAM
- Minimal feature engineering (mean + std)
- Feature selection (top 50 features)
- Saves final dataset as CSV
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# --- CONFIG ---
DATA_DIR = Path('data/deam')
FEATURES_DIR = DATA_DIR / 'features' / 'features'
ANNOTATIONS_DIR = DATA_DIR / 'DEAM_Annotations' / 'annotations' / 'annotations averaged per song' / 'song_level'
OUTPUT_PATH = Path('data/final_preprocessed_dataset.csv')
K_BEST = 50

# --- LOAD FEATURES (minimal engineering: mean + std) ---
feature_files = glob.glob(str(FEATURES_DIR / '*.csv'))
features_list = []
for file in feature_files:
    song_id = Path(file).stem
    df = pd.read_csv(file, sep=';')
    feature_stats = {}
    for col in df.columns[1:]:  # Skip frameTime
        feature_stats[f"{col}_mean"] = df[col].mean()
        feature_stats[f"{col}_std"] = df[col].std()
    feature_stats['song_id'] = song_id
    features_list.append(feature_stats)
features_df = pd.DataFrame(features_list)

# --- LOAD ANNOTATIONS ---
annotation_files = glob.glob(str(ANNOTATIONS_DIR / '*.csv'))
annotations_list = [pd.read_csv(f) for f in annotation_files]
annotations_df = pd.concat(annotations_list, ignore_index=True)
features_df['song_id'] = features_df['song_id'].astype(str)
annotations_df['song_id'] = annotations_df['song_id'].astype(str)

# --- MERGE ---
merged_df = pd.merge(features_df, annotations_df, on='song_id', how='inner')

# DEBUG: Print columns after merge
print('Columns in merged_df:', merged_df.columns.tolist())
print('Columns in annotations_df:', annotations_df.columns.tolist())

merged_df.columns = merged_df.columns.str.strip()

# --- CLEANUP ---
annotation_cols = ['song_id', 'valence_mean', 'valence_std', 'arousal_mean', 'arousal_std',
                  'valence_max_mean', 'valence_max_std', 'valence_min_mean', 'valence_min_std',
                  'arousal_max_mean', 'arousal_max_std', 'arousal_min_mean', 'arousal_min_std']
feature_cols = [col for col in merged_df.columns if col not in annotation_cols]

# --- NA HANDLING ---
nan_frac = merged_df[feature_cols].isnull().mean()
if not isinstance(nan_frac, pd.Series):
    nan_frac = pd.Series(nan_frac, index=feature_cols)
high_nan_cols = nan_frac[nan_frac > 0.5].index.tolist()
if high_nan_cols:
    merged_df = merged_df.drop(columns=high_nan_cols)
    feature_cols = [col for col in merged_df.columns if col not in annotation_cols]

# Remove constant features
constant_features = [col for col in feature_cols if merged_df[col].std() == 0]
if constant_features:
    merged_df = merged_df.drop(columns=constant_features)
    feature_cols = [col for col in merged_df.columns if col not in annotation_cols]

# Remove highly correlated features
if len(feature_cols) > 1:
    corr_matrix = merged_df[feature_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [col for col in upper_tri.columns if (upper_tri[col] > 0.95).any()]
    if high_corr_features:
        merged_df = merged_df.drop(columns=high_corr_features)
        feature_cols = [col for col in merged_df.columns if col not in annotation_cols]

# Impute any remaining NaNs
if merged_df[feature_cols].isnull().sum().sum() > 0:
    merged_df[feature_cols] = merged_df[feature_cols].fillna(merged_df[feature_cols].median())

# --- CREATE LABEL ---
merged_df['valence_bin'] = pd.cut(merged_df['valence_mean'], bins=3, labels=['Low', 'Medium', 'High'])
merged_df['arousal_bin'] = pd.cut(merged_df['arousal_mean'], bins=3, labels=['Low', 'Medium', 'High'])
merged_df['mood'] = merged_df['valence_bin'].astype(str) + '_' + merged_df['arousal_bin'].astype(str)

# --- FINAL FEATURE SELECTION PIPELINE ---
X = merged_df[feature_cols].values
y = merged_df['mood'].values
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('variance_threshold', VarianceThreshold(threshold=0.0)),
    ('scaler', RobustScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=K_BEST))
])
X_selected = pipeline.fit_transform(X, y)

# --- SAVE FINAL DATASET ---
final_df = pd.DataFrame(X_selected, columns=[f'feature_{i+1}' for i in range(X_selected.shape[1])])
final_df['mood'] = y
final_df['song_id'] = merged_df['song_id'].values
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Final preprocessed dataset saved to {OUTPUT_PATH} with shape {final_df.shape}") 