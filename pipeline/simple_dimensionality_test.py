#!/usr/bin/env python3
"""
Simple Dimensionality Reduction Comparison
Tests SVD vs Statistical Aggregation (no TensorFlow required)
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import logging
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDimensionalityTest:
    def __init__(self, data_dir="data/deam"):
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features" / "features"
        self.annotations_dir = self.data_dir / "DEAM_Annotations" / "annotations" / "annotations averaged per song" / "song_level"
        self.results = {}
        
    def load_data(self):
        """Load and prepare the DEAM dataset"""
        logger.info("Loading DEAM dataset...")
        
        # Load audio features
        feature_files = glob.glob(str(self.features_dir / "*.csv"))
        features_list = []
        
        for file in feature_files:
            song_id = Path(file).stem
            df = pd.read_csv(file, sep=';')  # Use semicolon separator
            # Calculate mean and std for each feature
            feature_stats = {}
            for col in df.columns[1:]:  # Skip frameTime column
                feature_stats[f"{col}_mean"] = df[col].mean()
                feature_stats[f"{col}_std"] = df[col].std()
            
            feature_stats['song_id'] = song_id
            features_list.append(feature_stats)
        
        features_df = pd.DataFrame(features_list)
        
        # Load annotations
        annotation_files = glob.glob(str(self.annotations_dir / "*.csv"))
        annotations_list = []
        
        for file in annotation_files:
            df = pd.read_csv(file)
            annotations_list.append(df)
        
        annotations_df = pd.concat(annotations_list, ignore_index=True)
        
        # Merge features with annotations
        # Ensure song_id columns have the same data type
        features_df['song_id'] = features_df['song_id'].astype(str)
        annotations_df['song_id'] = annotations_df['song_id'].astype(str)
        
        merged_df = pd.merge(features_df, annotations_df, left_on='song_id', right_on='song_id', how='inner')
        
        # Create mood labels
        merged_df['valence_bin'] = pd.cut(merged_df[' valence_mean'], bins=2, labels=['Low', 'High'])
        merged_df['arousal_bin'] = pd.cut(merged_df[' arousal_mean'], bins=2, labels=['Low', 'High'])
        
        # Create 4-quadrant mood labels
        merged_df['mood'] = merged_df['valence_bin'].astype(str) + '_' + merged_df['arousal_bin'].astype(str)
        
        logger.info(f"Loaded {len(merged_df)} songs with {len(features_df.columns)-1} features")
        return merged_df
    
    def test_statistical_aggregation(self, df):
        """Test Method 1: Statistical Aggregation"""
        logger.info("Testing Method 1: Statistical Aggregation")
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in ['song_id', 'mood', 'valence_bin', 'arousal_bin', 
                                                                ' valence_mean', ' arousal_mean', ' valence_std', ' arousal_std']]
        
        X = df[feature_cols].values
        y = df['mood'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Handle NaN values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.results['statistical'] = {
            'method': 'Statistical Aggregation',
            'n_features': X.shape[1],
            'accuracy': accuracy,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'classifier': clf,
            'scaler': scaler
        }
        
        logger.info(f"Statistical Aggregation: {accuracy:.4f} accuracy with {X.shape[1]} features")
        return accuracy
    
    def test_svd_reduction(self, df, n_components_list=[50, 100, 200]):
        """Test Method 2: SVD with different component counts"""
        logger.info("Testing Method 2: SVD Reduction")
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in ['song_id', 'mood', 'valence_bin', 'arousal_bin', 
                                                                ' valence_mean', ' arousal_mean', ' valence_std', ' arousal_std']]
        
        X = df[feature_cols].values
        y = df['mood'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Handle NaN values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        svd_results = {}
        
        for n_components in n_components_list:
            logger.info(f"Testing SVD with {n_components} components...")
            
            # Apply SVD
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            X_train_svd = svd.fit_transform(X_train_scaled)
            X_test_svd = svd.transform(X_test_scaled)
            
            # Train classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train_svd, y_train)
            
            # Predict and evaluate
            y_pred = clf.predict(X_test_svd)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate explained variance
            explained_variance = svd.explained_variance_ratio_.sum()
            
            svd_results[n_components] = {
                'accuracy': accuracy,
                'explained_variance': explained_variance,
                'X_train': X_train_svd,
                'X_test': X_test_svd,
                'y_pred': y_pred,
                'classifier': clf,
                'svd': svd
            }
            
            logger.info(f"SVD ({n_components} components): {accuracy:.4f} accuracy ({explained_variance:.3f} variance explained)")
        
        # Store best SVD result
        best_n = max(svd_results.keys(), key=lambda x: svd_results[x]['accuracy'])
        best_result = svd_results[best_n]
        
        self.results['svd'] = {
            'method': f'SVD Reduction (best: {best_n} components)',
            'n_features': best_n,
            'accuracy': best_result['accuracy'],
            'explained_variance': best_result['explained_variance'],
            'X_train': best_result['X_train'],
            'X_test': best_result['X_test'],
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': best_result['y_pred'],
            'classifier': best_result['classifier'],
            'scaler': scaler,
            'svd': best_result['svd'],
            'all_results': svd_results
        }
        
        return svd_results
    
    def run_comparison(self):
        """Run the comparison between methods"""
        logger.info("Starting dimensionality reduction comparison...")
        
        # Load data
        df = self.load_data()
        
        # Test methods
        stat_accuracy = self.test_statistical_aggregation(df)
        svd_results = self.test_svd_reduction(df)
        
        # Create comparison summary
        self.create_comparison_summary()
        
        # Create visualizations
        self.create_visualizations()
        
        return self.results
    
    def create_comparison_summary(self):
        """Create a summary table of results"""
        logger.info("Creating comparison summary...")
        
        summary_data = []
        for method_name, result in self.results.items():
            if result is not None:
                summary_data.append({
                    'Method': result['method'],
                    'Features': result['n_features'],
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'Additional Info': f"Variance: {result.get('explained_variance', 'N/A'):.3f}" if 'explained_variance' in result else 'N/A'
                })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + "="*80)
        print("DIMENSIONALITY REDUCTION COMPARISON RESULTS")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
        
        # Save results
        summary_df.to_csv('data/dimensionality_comparison_results.csv', index=False)
        logger.info("Results saved to data/dimensionality_comparison_results.csv")
    
    def create_visualizations(self):
        """Create visualizations comparing the methods"""
        logger.info("Creating visualizations...")
        
        # Create output directory
        os.makedirs('data/visualizations', exist_ok=True)
        
        # 1. Accuracy comparison
        methods = []
        accuracies = []
        feature_counts = []
        
        for method_name, result in self.results.items():
            if result is not None:
                methods.append(result['method'])
                accuracies.append(result['accuracy'])
                feature_counts.append(result['n_features'])
        
        plt.figure(figsize=(15, 10))
        
        # Accuracy plot
        plt.subplot(2, 3, 1)
        bars = plt.bar(methods, accuracies, color=['skyblue', 'lightgreen'])
        plt.title('Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Feature count plot
        plt.subplot(2, 3, 2)
        bars = plt.bar(methods, feature_counts, color=['skyblue', 'lightgreen'])
        plt.title('Feature Count Comparison')
        plt.ylabel('Number of Features')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, feature_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(feature_counts)*0.01, 
                    str(count), ha='center', va='bottom')
        
        # 2. SVD component analysis
        if 'svd' in self.results and 'all_results' in self.results['svd']:
            plt.subplot(2, 3, 3)
            svd_results = self.results['svd']['all_results']
            components = list(svd_results.keys())
            accuracies_svd = [svd_results[c]['accuracy'] for c in components]
            variances = [svd_results[c]['explained_variance'] for c in components]
            
            plt.plot(components, accuracies_svd, 'o-', label='Accuracy', color='blue')
            plt.xlabel('Number of Components')
            plt.ylabel('Accuracy')
            plt.title('SVD: Accuracy vs Components')
            plt.grid(True, alpha=0.3)
            
            # Add variance plot on secondary axis
            ax2 = plt.twinx()
            ax2.plot(components, variances, 's-', label='Explained Variance', color='red')
            ax2.set_ylabel('Explained Variance', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        # 3. t-SNE visualization for each method
        for i, (method_name, result) in enumerate(self.results.items()):
            if result is not None:
                plt.subplot(2, 3, 4 + i)
                
                # Apply t-SNE for visualization
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(result['X_train'])-1))
                X_tsne = tsne.fit_transform(result['X_train'])
                
                # Plot
                unique_labels = np.unique(result['y_train'])
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
                
                for j, label in enumerate(unique_labels):
                    mask = result['y_train'] == label
                    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                              c=[colors[j]], label=label, alpha=0.7)
                
                plt.title(f'{result["method"]} - t-SNE')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig('data/visualizations/dimensionality_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Visualizations saved to data/visualizations/dimensionality_comparison.png")

def main():
    """Main function to run the comparison"""
    print("🎵 DEAM Dataset Dimensionality Reduction Comparison")
    print("="*60)
    
    # Create comparison object
    test = SimpleDimensionalityTest()
    
    # Run comparison
    results = test.run_comparison()
    
    print("\n✅ Comparison complete! Check the results above and the generated files:")
    print("- data/dimensionality_comparison_results.csv")
    print("- data/visualizations/dimensionality_comparison.png")
    
    # Print recommendations
    print("\n📊 RECOMMENDATIONS:")
    best_method = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_method]['accuracy']
    best_features = results[best_method]['n_features']
    
    print(f"- Best method: {results[best_method]['method']}")
    print(f"- Best accuracy: {best_accuracy:.4f}")
    print(f"- Features used: {best_features}")
    
    if best_method == 'svd':
        print("- SVD provides good dimensionality reduction while maintaining performance")
    else:
        print("- Statistical aggregation works well but uses many features")

if __name__ == "__main__":
    main() 