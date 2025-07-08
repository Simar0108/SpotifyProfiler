#!/usr/bin/env python3
"""
Debug script to understand DEAM data loading issues
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_data_loading():
    data_dir = Path("data/deam")
    features_dir = data_dir / "features" / "features"
    annotations_dir = data_dir / "DEAM_Annotations" / "annotations" / "annotations averaged per song" / "song_level"
    
    print("🔍 DEBUGGING DEAM DATA LOADING")
    print("="*50)
    
    # Check if directories exist
    print(f"Features directory exists: {features_dir.exists()}")
    print(f"Annotations directory exists: {annotations_dir.exists()}")
    
    # List some feature files
    feature_files = list(features_dir.glob("*.csv"))
    print(f"\nFound {len(feature_files)} feature files")
    print(f"First 5 feature files: {[f.name for f in feature_files[:5]]}")
    
    # Check one feature file
    if feature_files:
        sample_file = feature_files[0]
        print(f"\n📊 Sample feature file: {sample_file.name}")
        df = pd.read_csv(sample_file)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns[:10])}...")  # First 10 columns
        print(f"First few rows:")
        print(df.head(3))
    
    # Check annotation files
    annotation_files = list(annotations_dir.glob("*.csv"))
    print(f"\nFound {len(annotation_files)} annotation files")
    print(f"Annotation files: {[f.name for f in annotation_files]}")
    
    if annotation_files:
        sample_annotation = annotation_files[0]
        print(f"\n📊 Sample annotation file: {sample_annotation.name}")
        df = pd.read_csv(sample_annotation)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head(3))
    
    # Try to load features like the original script
    print(f"\n🔄 Testing feature loading...")
    features_list = []
    
    for file in feature_files[:5]:  # Just first 5 for testing
        try:
            df = pd.read_csv(file)
            # Extract song_id from filename
            song_id = file.stem  # filename without extension
            df['song_id'] = song_id
            features_list.append(df)
            print(f"Loaded {file.name}: shape {df.shape}")
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    
    if features_list:
        features_df = pd.concat(features_list, ignore_index=True)
        print(f"\nCombined features shape: {features_df.shape}")
        print(f"Feature columns: {len([col for col in features_df.columns if col != 'song_id'])}")
        
        # Check for NaN values
        nan_count = features_df.isna().sum().sum()
        print(f"Total NaN values: {nan_count}")
        
        if nan_count > 0:
            print("Columns with NaN values:")
            nan_cols = features_df.columns[features_df.isna().any()].tolist()
            print(f"First 10: {nan_cols[:10]}")

if __name__ == "__main__":
    debug_data_loading() 