#!/usr/bin/env python3
"""
DEAM Dataset Preprocessing Script
Follows the preprocessing guide to create a training-ready dataset
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

class DEAMPreprocessor:
    def __init__(self, data_dir="data/deam"):
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features" / "features"
        self.annotations_dir = self.data_dir / "DEAM_Annotations" / "annotations" / "annotations averaged per song" / "song_level"
        
    def load_audio_features(self):
        """Load and merge all audio feature CSV files"""
        logger.info("🎵 Loading audio features...")
        
        # Get all CSV files in the features directory
        csv_files = glob.glob(str(self.features_dir / "*.csv"))
        logger.info(f"Found {len(csv_files)} feature files")
        
        all_features = []
        
        for file_path in csv_files:
            try:
                # Extract song_id from filename (remove .csv extension)
                song_id = int(Path(file_path).stem)
                
                # Read the CSV file
                df = pd.read_csv(file_path, sep=';')
                
                # Calculate mean and std for each feature across all frames
                feature_stats = {}
                
                # Skip the first column (frameTime) and calculate statistics
                for col in df.columns[1:]:  # Skip frameTime column
                    feature_stats[f"{col}_mean"] = df[col].mean()
                    feature_stats[f"{col}_std"] = df[col].std()
                
                # Add song_id
                feature_stats['song_id'] = song_id
                
                all_features.append(feature_stats)
                
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        logger.info(f"✅ Loaded features for {len(features_df)} songs")
        
        return features_df
    
    def load_annotations(self):
        """Load valence and arousal annotations"""
        logger.info("📊 Loading annotations...")
        
        # Load both annotation files
        annotations_files = [
            "static_annotations_averaged_songs_1_2000.csv",
            "static_annotations_averaged_songs_2000_2058.csv"
        ]
        
        all_annotations = []
        
        for file_name in annotations_files:
            file_path = self.annotations_dir / file_name
            if file_path.exists():
                df = pd.read_csv(file_path)
                all_annotations.append(df)
            else:
                logger.warning(f"Annotation file not found: {file_path}")
        
        if all_annotations:
            annotations_df = pd.concat(all_annotations, ignore_index=True)
            logger.info(f"✅ Loaded annotations for {len(annotations_df)} songs")
            return annotations_df
        else:
            raise FileNotFoundError("No annotation files found")
    
    def merge_features_with_labels(self, features_df, annotations_df):
        """Merge features with valence/arousal labels"""
        logger.info("🔗 Merging features with labels...")
        
        # Merge on song_id
        merged_df = features_df.merge(annotations_df, on='song_id', how='inner')
        
        # Drop rows with missing values
        initial_count = len(merged_df)
        merged_df = merged_df.dropna()
        final_count = len(merged_df)
        
        logger.info(f"✅ Merged dataset: {final_count} songs (dropped {initial_count - final_count} with missing values)")
        
        return merged_df
    
    def create_mood_labels(self, df):
        """Create discrete mood labels based on valence and arousal"""
        logger.info("🎭 Creating mood labels...")
        
        def mood_label(valence, arousal):
            if valence >= 0.5 and arousal >= 0.5:
                return "Energetic"
            elif valence >= 0.5 and arousal < 0.5:
                return "Chill"
            elif valence < 0.5 and arousal >= 0.5:
                return "Tense"
            else:
                return "Melancholic"
        
        # Normalize valence and arousal to 0-1 scale (assuming they're on 1-9 scale)
        df['valence_norm'] = (df['valence_mean'] - 1) / 8
        df['arousal_norm'] = (df['arousal_mean'] - 1) / 8
        
        # Create mood labels
        df['mood_class'] = df.apply(lambda row: mood_label(row['valence_norm'], row['arousal_norm']), axis=1)
        
        # Log mood distribution
        mood_counts = df['mood_class'].value_counts()
        logger.info("Mood distribution:")
        for mood, count in mood_counts.items():
            logger.info(f"  {mood}: {count} songs")
        
        return df
    
    def save_processed_dataset(self, df, output_path="data/processed_dataset.csv"):
        """Save the final processed dataset"""
        logger.info(f"💾 Saving processed dataset to {output_path}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Dataset saved with {len(df)} songs and {len(df.columns)} features")
        
        return output_path
    
    def preprocess(self, output_path="data/processed_dataset.csv", include_mood_labels=True):
        """Complete preprocessing pipeline"""
        logger.info("🚀 Starting DEAM dataset preprocessing...")
        
        try:
            # Step 1: Load audio features
            features_df = self.load_audio_features()
            
            # Step 2: Load annotations
            annotations_df = self.load_annotations()
            
            # Step 3: Merge features with labels
            merged_df = self.merge_features_with_labels(features_df, annotations_df)
            
            # Step 4: Create mood labels (optional)
            if include_mood_labels:
                merged_df = self.create_mood_labels(merged_df)
            
            # Step 5: Save processed dataset
            output_path = self.save_processed_dataset(merged_df, output_path)
            
            logger.info("🎉 Preprocessing completed successfully!")
            
            # Print dataset summary
            logger.info(f"Final dataset shape: {merged_df.shape}")
            logger.info(f"Features: {len(merged_df.columns) - 5} audio features + 5 labels")
            
            return merged_df, output_path
            
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {e}")
            raise

def main():
    """Main function to run preprocessing"""
    preprocessor = DEAMPreprocessor()
    
    # Run preprocessing
    df, output_path = preprocessor.preprocess(include_mood_labels=True)
    
    print(f"\n✅ Preprocessing completed!")
    print(f"📁 Output saved to: {output_path}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎭 Mood labels included: {'mood_class' in df.columns}")

if __name__ == "__main__":
    main()
