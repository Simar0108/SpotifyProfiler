#!/usr/bin/env python3
"""
Enhanced Dimensionality Reduction Comparison for DEAM Dataset
Uses sklearn pipelines, robust preprocessing, and improved feature engineering
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import logging
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings
import time
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDimensionalityComparison:
    def __init__(self, data_dir="data/deam"):
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features" / "features"
        self.annotations_dir = self.data_dir / "DEAM_Annotations" / "annotations" / "annotations averaged per song" / "song_level"
        self.results = {}
        
    def load_data_with_enhanced_engineering(self):
        """Load and prepare the DEAM dataset with enhanced feature engineering"""
        logger.info("Loading DEAM dataset with enhanced feature engineering...")
        
        # Load audio features
        feature_files = glob.glob(str(self.features_dir / "*.csv"))
        logger.info(f"Found {len(feature_files)} feature files")
        features_list = []
        
        for file in feature_files:
            song_id = Path(file).stem
            df = pd.read_csv(file, sep=';')
            
            # Enhanced feature engineering with multiple statistics
            feature_stats = {}
            for col in df.columns[1:]:  # Skip frameTime column
                # Basic statistics
                feature_stats[f"{col}_mean"] = df[col].mean()
                feature_stats[f"{col}_std"] = df[col].std()
                #feature_stats[f"{col}_min"] = df[col].min()
                #feature_stats[f"{col}_max"] = df[col].max()
                #feature_stats[f"{col}_median"] = df[col].median()
                
                # Additional statistical features
                #feature_stats[f"{col}_q25"] = df[col].quantile(0.25)
                #feature_stats[f"{col}_q75"] = df[col].quantile(0.75)
                #feature_stats[f"{col}_range"] = df[col].max() - df[col].min()
                #feature_stats[f"{col}_iqr"] = df[col].quantile(0.75) - df[col].quantile(0.25)
                #feature_stats[f"{col}_skew"] = df[col].skew()
                #feature_stats[f"{col}_kurtosis"] = df[col].kurtosis()
                
                # Zero-crossing rate (useful for audio features)
                #feature_stats[f"{col}_zero_crossings"] = np.sum(np.diff(np.signbit(df[col] - df[col].mean())))
                
                # Peak features
                #feature_stats[f"{col}_peak_count"] = len(df[col][df[col] > df[col].mean() + df[col].std()])
                #feature_stats[f"{col}_valley_count"] = len(df[col][df[col] < df[col].mean() - df[col].std()])
                
            feature_stats['song_id'] = song_id
            
            if len(features_list) == 0:
                logger.info(f"Columns in first feature file ({file}): {df.columns.tolist()}")
                logger.info(f"Enhanced features created: {len(feature_stats)} features per song")
            
            features_list.append(feature_stats)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Created enhanced features DataFrame with {len(features_df)} songs and {len(features_df.columns)} columns")
        
        # Load annotations
        annotation_files = glob.glob(str(self.annotations_dir / "*.csv"))
        logger.info(f"Found {len(annotation_files)} annotation files")
        annotations_list = []
        for file in annotation_files:
            df = pd.read_csv(file)
            annotations_list.append(df)
        annotations_df = pd.concat(annotations_list, ignore_index=True)
        logger.info(f"Created annotations DataFrame with {len(annotations_df)} songs")
        
        # Ensure song_id columns are both strings
        features_df['song_id'] = features_df['song_id'].astype(str)
        annotations_df['song_id'] = annotations_df['song_id'].astype(str)
        
        # Merge features with annotations
        merged_df = pd.merge(features_df, annotations_df, left_on='song_id', right_on='song_id', how='inner')
        logger.info(f"After merge: {len(merged_df)} songs")
        
        # Clean column names (remove leading/trailing spaces)
        merged_df.columns = merged_df.columns.str.strip()
        
        # Define annotation columns
        annotation_cols = ['song_id', 'valence_mean', 'valence_std', 'arousal_mean', 'arousal_std',
                          'valence_max_mean', 'valence_max_std', 'valence_min_mean', 'valence_min_std',
                          'arousal_max_mean', 'arousal_max_std', 'arousal_min_mean', 'arousal_min_std']
        
        # Get feature columns
        feature_cols = [col for col in merged_df.columns if col not in annotation_cols]
        logger.info(f"Initial feature columns: {len(feature_cols)}")
        
        # Robust NaN handling with detailed logging
        logger.info("Starting robust NaN handling...")
        
        # 1. Check initial NaN status
        initial_nan_count = merged_df[feature_cols].isnull().sum().sum()
        logger.info(f"Initial NaN count: {initial_nan_count}")
        
        # 2. Drop columns with >50% NaNs
        nan_frac = merged_df[feature_cols].isnull().mean()
        # Ensure nan_frac is a pandas Series
        if not isinstance(nan_frac, pd.Series):
            nan_frac = pd.Series(nan_frac, index=feature_cols)
        high_nan_cols = nan_frac[nan_frac > 0.5].index.tolist()
        if high_nan_cols:
            logger.info(f"Dropping {len(high_nan_cols)} columns with >50% NaNs")
            merged_df = merged_df.drop(columns=high_nan_cols)
            feature_cols = [col for col in merged_df.columns if col not in annotation_cols]
        
        # 3. Remove constant features (zero variance)
        constant_features = []
        for col in feature_cols:
            if merged_df[col].std() == 0:
                constant_features.append(col)
        if constant_features:
            logger.info(f"Removing {len(constant_features)} constant features")
            merged_df = merged_df.drop(columns=constant_features)
            feature_cols = [col for col in merged_df.columns if col not in annotation_cols]
        
        # 4. Remove highly correlated features (>0.95 correlation)
        if len(feature_cols) > 1:
            logger.info("Checking for highly correlated features...")
            corr_matrix = merged_df[feature_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_features = []
            for column in upper_tri.columns:
                if (upper_tri[column] > 0.95).any():
                    high_corr_features.append(column)
            if high_corr_features:
                logger.info(f"Removing {len(high_corr_features)} highly correlated features")
                merged_df = merged_df.drop(columns=high_corr_features)
                feature_cols = [col for col in merged_df.columns if col not in annotation_cols]
        
        # 5. Final NaN check and imputation
        remaining_nan_count = merged_df[feature_cols].isnull().sum().sum()
        if remaining_nan_count > 0:
            logger.info(f"Imputing {remaining_nan_count} remaining NaN values with median")
            merged_df[feature_cols] = merged_df[feature_cols].fillna(merged_df[feature_cols].median())
        
        # Create mood labels with more granular bins
        merged_df['valence_bin'] = pd.cut(merged_df['valence_mean'], bins=3, labels=['Low', 'Medium', 'High'])
        merged_df['arousal_bin'] = pd.cut(merged_df['arousal_mean'], bins=3, labels=['Low', 'Medium', 'High'])
        merged_df['mood'] = merged_df['valence_bin'].astype(str) + '_' + merged_df['arousal_bin'].astype(str)
        
        # Log final statistics
        final_nan_count = merged_df[feature_cols].isnull().sum().sum()
        mood_counts = merged_df['mood'].value_counts()
        
        logger.info(f"Final dataset: {len(merged_df)} songs, {len(feature_cols)} features")
        logger.info(f"Final NaN count: {final_nan_count}")
        logger.info(f"Mood class distribution: {mood_counts.to_dict()}")
        
        return merged_df, feature_cols
    
    def create_sklearn_pipelines(self, k_values=[5, 10, 20, 50]):
        """Create sklearn pipelines for different dimensionality reduction methods"""
        logger.info("Creating sklearn pipelines...")
        
        # Load data
        df, feature_cols = self.load_data_with_enhanced_engineering()
        
        # Prepare data
        X = df[feature_cols].values
        y = df['mood'].values
        
        # Final safety check for NaNs
        if np.isnan(X).any():
            logger.error("X contains NaN values! Replacing with 0.")
            X = np.nan_to_num(X.astype(np.float64), nan=0.0)
        
        logger.info(f"Final X shape: {X.shape}, y shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        results = []
        
        # 1. Baseline Pipeline (No DR)
        logger.info("Testing Baseline Pipeline (No DR)")
        t0 = time.time()
        
        baseline_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('variance_threshold', VarianceThreshold(threshold=0.0)),
            ('scaler', RobustScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        baseline_pipeline.fit(X_train, y_train)
        y_pred = baseline_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        t1 = time.time()
        
        results.append({
            'method': 'Baseline Pipeline',
            'k': None,
            'n_features': X.shape[1],
            'accuracy': acc,
            'f1': f1,
            'variance_retained': None,
            'recon_loss': None,
            'runtime': t1-t0
        })
        
        # 2. SVD Pipeline
        for k in k_values:
            if k >= X.shape[1]:
                continue
            logger.info(f"Testing SVD Pipeline with k={k}")
            t0 = time.time()
            
            svd_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('variance_threshold', VarianceThreshold(threshold=0.0)),
                ('scaler', RobustScaler()),
                ('svd', TruncatedSVD(n_components=k, random_state=42)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
            svd_pipeline.fit(X_train, y_train)
            y_pred = svd_pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Get variance retained from the fitted SVD
            svd_step = svd_pipeline.named_steps['svd']
            var = svd_step.explained_variance_ratio_.sum()
            t1 = time.time()
            
            results.append({
                'method': 'SVD Pipeline',
                'k': k,
                'n_features': k,
                'accuracy': acc,
                'f1': f1,
                'variance_retained': var,
                'recon_loss': None,
                'runtime': t1-t0
            })
        
        # 3. PCA Pipeline
        for k in k_values:
            if k >= X.shape[1]:
                continue
            logger.info(f"Testing PCA Pipeline with k={k}")
            t0 = time.time()
            
            pca_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('variance_threshold', VarianceThreshold(threshold=0.0)),
                ('scaler', RobustScaler()),
                ('pca', PCA(n_components=k, random_state=42)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
            pca_pipeline.fit(X_train, y_train)
            y_pred = pca_pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Get variance retained from the fitted PCA
            pca_step = pca_pipeline.named_steps['pca']
            var = pca_step.explained_variance_ratio_.sum()
            t1 = time.time()
            
            results.append({
                'method': 'PCA Pipeline',
                'k': k,
                'n_features': k,
                'accuracy': acc,
                'f1': f1,
                'variance_retained': var,
                'recon_loss': None,
                'runtime': t1-t0
            })
        
        # 4. Feature Selection Pipeline
        for k in k_values:
            if k >= X.shape[1]:
                continue
            logger.info(f"Testing Feature Selection Pipeline with k={k}")
            t0 = time.time()
            
            feature_selection_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('variance_threshold', VarianceThreshold(threshold=0.0)),
                ('scaler', RobustScaler()),
                ('feature_selection', SelectKBest(score_func=f_classif, k=k)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
            feature_selection_pipeline.fit(X_train, y_train)
            y_pred = feature_selection_pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            t1 = time.time()
            
            results.append({
                'method': 'Feature Selection Pipeline',
                'k': k,
                'n_features': k,
                'accuracy': acc,
                'f1': f1,
                'variance_retained': None,
                'recon_loss': None,
                'runtime': t1-t0
            })
        
        # 5. Logistic Regression Pipeline (for comparison)
        logger.info("Testing Logistic Regression Pipeline (No DR)")
        t0 = time.time()
        
        lr_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('variance_threshold', VarianceThreshold(threshold=0.0)),
            ('scaler', RobustScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        lr_pipeline.fit(X_train, y_train)
        y_pred = lr_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        t1 = time.time()
        
        results.append({
            'method': 'Logistic Regression Pipeline',
            'k': None,
            'n_features': X.shape[1],
            'accuracy': acc,
            'f1': f1,
            'variance_retained': None,
            'recon_loss': None,
            'runtime': t1-t0
        })
        
        logger.info("All pipeline methods completed successfully!")
        return results
    
    def run_legacy_methods(self, k_values=[5, 10, 20, 50]):
        """Legacy method for comparison with original approach"""
        logger.info("Running legacy methods for comparison...")
        df, feature_cols = self.load_data_with_enhanced_engineering()
        results = []
        
        # Prepare data
        X = df[feature_cols].values
        y = df['mood'].values
        
        # Final NaN check
        if np.isnan(X).any():
            logger.warning("X contains NaN values! Replacing with 0.")
            X = np.nan_to_num(X.astype(np.float64), nan=0.0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Legacy Statistical Aggregation
        logger.info("Testing Legacy Statistical Aggregation")
        t0 = time.time()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        t1 = time.time()
        
        results.append({
            'method': 'Legacy Statistical Aggregation',
            'k': None,
            'n_features': X.shape[1],
            'accuracy': acc,
            'f1': f1,
            'variance_retained': None,
            'recon_loss': None,
            'runtime': t1-t0
        })
        
        # Legacy SVD
        for k in k_values:
            if k >= X.shape[1]:
                continue
            logger.info(f"Testing Legacy SVD with k={k}")
            t0 = time.time()
            svd = TruncatedSVD(n_components=k, random_state=42)
            X_train_svd = svd.fit_transform(X_train_scaled)
            X_test_svd = svd.transform(X_test_scaled)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train_svd, y_train)
            y_pred = clf.predict(X_test_svd)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            var = svd.explained_variance_ratio_.sum()
            t1 = time.time()
            
            results.append({
                'method': 'Legacy SVD',
                'k': k,
                'n_features': k,
                'accuracy': acc,
                'f1': f1,
                'variance_retained': var,
                'recon_loss': None,
                'runtime': t1-t0
            })
        
        return results

    def create_comparison_summary(self, results):
        logger.info("Creating comparison summary...")
        summary_df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("ENHANCED DIMENSIONALITY REDUCTION COMPARISON RESULTS")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
        
        # Save results
        os.makedirs('data', exist_ok=True)
        summary_df.to_csv('data/dimensionality_comparison_results.csv', index=False)
        logger.info("Results saved to data/dimensionality_comparison_results.csv")
        
        # Print best results
        best_acc = summary_df.loc[summary_df['accuracy'].idxmax()]
        best_f1 = summary_df.loc[summary_df['f1'].idxmax()]
        print(f"\n🏆 Best Accuracy: {best_acc['method']} (k={best_acc['k']}) - {best_acc['accuracy']:.4f}")
        print(f"🏆 Best F1 Score: {best_f1['method']} (k={best_f1['k']}) - {best_f1['f1']:.4f}")
        
        return summary_df

    def create_enhanced_visualizations(self, summary_df):
        logger.info("Creating enhanced visualizations...")
        os.makedirs('data/visualizations', exist_ok=True)
        
        # Create a comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Enhanced Dimensionality Reduction Comparison Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        for method in summary_df['method'].unique():
            sub = summary_df[summary_df['method'] == method]
            ax1.plot(sub['k'].fillna(0), sub['accuracy'], marker='o', label=method, linewidth=2, markersize=6)
        ax1.set_title('Accuracy vs k', fontweight='bold')
        ax1.set_xlabel('k (components/dimensions)')
        ax1.set_ylabel('Accuracy')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. F1 Score comparison
        ax2 = axes[0, 1]
        for method in summary_df['method'].unique():
            sub = summary_df[summary_df['method'] == method]
            ax2.plot(sub['k'].fillna(0), sub['f1'], marker='s', label=method, linewidth=2, markersize=6)
        ax2.set_title('F1 Score vs k', fontweight='bold')
        ax2.set_xlabel('k (components/dimensions)')
        ax2.set_ylabel('F1 Score')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Runtime comparison
        ax3 = axes[0, 2]
        for method in summary_df['method'].unique():
            sub = summary_df[summary_df['method'] == method]
            ax3.plot(sub['k'].fillna(0), sub['runtime'], marker='^', label=method, linewidth=2, markersize=6)
        ax3.set_title('Runtime vs k', fontweight='bold')
        ax3.set_xlabel('k (components/dimensions)')
        ax3.set_ylabel('Runtime (seconds)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Variance retained (for methods that have it)
        ax4 = axes[1, 0]
        variance_methods = summary_df[summary_df['variance_retained'].notna()]
        for method in variance_methods['method'].unique():
            sub = variance_methods[variance_methods['method'] == method]
            ax4.plot(sub['k'], sub['variance_retained'], marker='d', label=method, linewidth=2, markersize=6)
        ax4.set_title('Variance Retained vs k', fontweight='bold')
        ax4.set_xlabel('k')
        ax4.set_ylabel('Variance Retained')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # 5. Method comparison heatmap
        ax5 = axes[1, 1]
        pivot_acc = summary_df.pivot(index='method', columns='k', values='accuracy')
        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax5)
        ax5.set_title('Accuracy Heatmap', fontweight='bold')
        
        # 6. Best method summary
        ax6 = axes[1, 2]
        best_methods = summary_df.loc[summary_df.groupby('method')['accuracy'].idxmax()]
        bars = ax6.bar(range(len(best_methods)), best_methods['accuracy'], 
                       color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange'])
        ax6.set_title('Best Accuracy per Method', fontweight='bold')
        ax6.set_ylabel('Accuracy')
        ax6.set_xticks(range(len(best_methods)))
        ax6.set_xticklabels([f"{m}\n(k={k})" for m, k in zip(best_methods['method'], best_methods['k'])], 
                           rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('data/visualizations/enhanced_dimensionality_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("Enhanced visualizations saved to data/visualizations/enhanced_dimensionality_comparison.png")

def main():
    """Main function to run the enhanced comparison"""
    print("🎵 Enhanced DEAM Dataset Dimensionality Reduction Comparison")
    print("="*70)
    print("Features: sklearn pipelines, enhanced feature engineering, robust preprocessing")
    print("="*70)
    
    # Create comparison object
    comparison = EnhancedDimensionalityComparison()
    
    # Run sklearn pipeline methods
    print("\n🚀 Running sklearn pipeline methods...")
    pipeline_results = comparison.create_sklearn_pipelines(k_values=[5, 10, 20, 50])
    
    # Run legacy methods for comparison
    print("\n🔄 Running legacy methods for comparison...")
    legacy_results = comparison.run_legacy_methods(k_values=[5, 10, 20, 50])
    
    # Combine results
    all_results = pipeline_results + legacy_results
    
    # Create summary and visualizations
    summary_df = comparison.create_comparison_summary(all_results)
    comparison.create_enhanced_visualizations(summary_df)
    
    print("\n✅ Enhanced comparison complete! Check the results above and the generated files:")
    print("- data/dimensionality_comparison_results.csv")
    print("- data/visualizations/enhanced_dimensionality_comparison.png")
    print("\n🎯 Key improvements:")
    print("- Robust NaN handling with sklearn pipelines")
    print("- Enhanced feature engineering (12 stats per feature)")
    print("- Multiple dimensionality reduction methods")
    print("- Comprehensive preprocessing pipeline")

if __name__ == "__main__":
    main() 