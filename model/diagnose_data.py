#!/usr/bin/env python3
"""
Data Diagnostic Script
Analyze data quality, feature importance, and class distribution
to understand why accuracy might be stuck at 63%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os

def analyze_feature_importance(X, y, feature_names):
    """Analyze feature importance using multiple methods."""
    print("🔍 Analyzing feature importance...")
    
    # Method 1: Mutual Information
    print("   Method 1: Mutual Information")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_ranking = np.argsort(mi_scores)[::-1]
    
    print("   Top 10 features by Mutual Information:")
    for i in range(10):
        idx = mi_ranking[i]
        print(f"      {i+1:2d}. {feature_names[idx]:<25} Score: {mi_scores[idx]:.4f}")
    
    # Method 2: F-statistic (ANOVA)
    print("\n   Method 2: F-statistic (ANOVA)")
    f_scores, _ = f_classif(X, y)
    f_ranking = np.argsort(f_scores)[::-1]
    
    print("   Top 10 features by F-statistic:")
    for i in range(10):
        idx = f_ranking[i]
        print(f"      {i+1:2d}. {feature_names[idx]:<25} Score: {f_scores[idx]:.2f}")
    
    # Method 3: Random Forest
    print("\n   Method 3: Random Forest Feature Importance")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    rf_ranking = np.argsort(rf_importance)[::-1]
    
    print("   Top 10 features by Random Forest:")
    for i in range(10):
        idx = rf_ranking[i]
        print(f"      {i+1:2d}. {feature_names[idx]:<25} Score: {rf_importance[idx]:.4f}")
    
    # Find common important features across methods
    top_mi = set(mi_ranking[:15])
    top_f = set(f_ranking[:15])
    top_rf = set(rf_ranking[:15])
    
    common_important = top_mi.intersection(top_f).intersection(top_rf)
    print(f"\n   Features important across all 3 methods: {len(common_important)}")
    if common_important:
        print("   Common important features:")
        for idx in sorted(common_important):
            print(f"      - {feature_names[idx]}")
    
    return {
        'mi_scores': mi_scores,
        'f_scores': f_scores,
        'rf_importance': rf_importance,
        'mi_ranking': mi_ranking,
        'f_ranking': f_ranking,
        'rf_ranking': rf_ranking,
        'common_important': common_important
    }

def analyze_class_distribution(y, y_train, y_test):
    """Analyze class distribution and balance."""
    print("\n📊 Analyzing class distribution...")
    
    # Overall distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"   Overall class distribution:")
    for cls, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        print(f"      Class {cls}: {count} samples ({percentage:.1f}%)")
    
    # Train/test distribution
    print(f"\n   Training set class distribution:")
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(train_unique, train_counts):
        percentage = (count / len(y_train)) * 100
        print(f"      Class {cls}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n   Test set class distribution:")
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    for cls, count in zip(test_unique, test_counts):
        percentage = (count / len(y_test)) * 100
        print(f"      Class {cls}: {count} samples ({percentage:.1f}%)")
    
    # Check for class imbalance
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    print(f"\n   Class imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 2:
        print("   ⚠️  Significant class imbalance detected!")
    elif imbalance_ratio > 1.5:
        print("   ⚠️  Moderate class imbalance detected!")
    else:
        print("   ✅ Classes are relatively balanced")
    
    return {
        'overall_distribution': dict(zip(unique, counts)),
        'train_distribution': dict(zip(train_unique, train_counts)),
        'test_distribution': dict(zip(test_unique, test_counts)),
        'imbalance_ratio': imbalance_ratio
    }

def analyze_feature_correlations(X, feature_names):
    """Analyze feature correlations to identify redundant features."""
    print("\n🔗 Analyzing feature correlations...")
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = abs(corr_matrix[i, j])
            if corr > 0.95:  # Very high correlation
                high_corr_pairs.append((i, j, corr))
    
    print(f"   Found {len(high_corr_pairs)} pairs of highly correlated features (|r| > 0.95):")
    for i, j, corr in high_corr_pairs[:10]:  # Show first 10
        print(f"      {feature_names[i]:<25} ↔ {feature_names[j]:<25} r = {corr:.3f}")
    
    if len(high_corr_pairs) > 10:
        print(f"      ... and {len(high_corr_pairs) - 10} more pairs")
    
    # Count features with high correlation to any other feature
    high_corr_features = set()
    for i, j, _ in high_corr_pairs:
        high_corr_features.add(i)
        high_corr_features.add(j)
    
    print(f"\n   Features with high correlation to others: {len(high_corr_features)}")
    print(f"   Potential for feature reduction: {len(high_corr_features)} features could be removed")
    
    return {
        'correlation_matrix': corr_matrix,
        'high_corr_pairs': high_corr_pairs,
        'high_corr_features': high_corr_features
    }

def test_feature_subset_performance(X, y, feature_names, importance_scores, n_features_list):
    """Test performance with different numbers of top features."""
    print("\n🧪 Testing performance with feature subsets...")
    
    # Sort features by importance
    feature_ranking = np.argsort(importance_scores)[::-1]
    
    results = []
    for n_features in n_features_list:
        if n_features > len(feature_names):
            continue
            
        # Select top n features
        selected_features = feature_ranking[:n_features]
        X_subset = X[:, selected_features]
        
        # Test with Random Forest (quick baseline)
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        scores = cross_val_score(rf, X_subset, y, cv=5, scoring='accuracy')
        
        mean_score = scores.mean()
        std_score = scores.std()
        
        print(f"   Top {n_features:2d} features: {mean_score:.4f} ± {std_score:.4f}")
        
        results.append({
            'n_features': n_features,
            'mean_score': mean_score,
            'std_score': std_score,
            'selected_features': selected_features
        })
    
    return results

def create_diagnostic_plots(X, y, feature_names, analysis_results):
    """Create comprehensive diagnostic plots."""
    print("\n📈 Creating diagnostic plots...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Feature importance comparison
    plt.subplot(3, 3, 1)
    top_features = 15
    mi_scores = analysis_results['mi_scores']
    f_scores = analysis_results['f_scores']
    rf_importance = analysis_results['rf_importance']
    
    # Normalize scores for comparison
    mi_norm = mi_scores / np.max(mi_scores)
    f_norm = f_scores / np.max(f_scores)
    rf_norm = rf_importance / np.max(rf_importance)
    
    top_indices = analysis_results['mi_ranking'][:top_features]
    x_pos = np.arange(len(top_indices))
    
    plt.bar(x_pos - 0.2, mi_norm[top_indices], 0.2, label='Mutual Info', alpha=0.8)
    plt.bar(x_pos, f_norm[top_indices], 0.2, label='F-statistic', alpha=0.8)
    plt.bar(x_pos + 0.2, rf_norm[top_indices], 0.2, label='Random Forest', alpha=0.8)
    
    plt.xlabel('Feature Rank')
    plt.ylabel('Normalized Importance Score')
    plt.title('Feature Importance Comparison (Top 15)')
    plt.xticks(x_pos, [feature_names[i][:15] for i in top_indices], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Class distribution
    plt.subplot(3, 3, 2)
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(unique, counts, color=['#ff9999', '#66b3ff', '#99ff99'])
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # 3. Feature correlation heatmap (top 20)
    plt.subplot(3, 3, 3)
    top_20_features = analysis_results['mi_ranking'][:20]
    corr_subset = analysis_results['correlation_matrix'][top_20_features][:, top_20_features]
    
    sns.heatmap(corr_subset, 
                xticklabels=[feature_names[i][:15] for i in top_20_features],
                yticklabels=[feature_names[i][:15] for i in top_20_features],
                cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Heatmap (Top 20)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 4. Feature variance distribution
    plt.subplot(3, 3, 4)
    feature_variance = np.var(X, axis=0)
    # Use fewer bins to avoid the error
    n_bins = min(20, len(np.unique(feature_variance)))
    plt.hist(feature_variance, bins=n_bins, alpha=0.7, edgecolor='black')
    plt.xlabel('Feature Variance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Feature Variances')
    plt.grid(True, alpha=0.3)
    
    # 5. Feature importance by category
    plt.subplot(3, 3, 5)
    # Group features by type (you might need to adjust this based on your feature names)
    feature_types = {}
    for i, name in enumerate(feature_names):
        if 'mfcc' in name.lower():
            feature_type = 'MFCC'
        elif 'spectral' in name.lower():
            feature_type = 'Spectral'
        elif 'rhythm' in name.lower():
            feature_type = 'Rhythm'
        elif 'chroma' in name.lower():
            feature_type = 'Chroma'
        else:
            feature_type = 'Other'
        
        if feature_type not in feature_types:
            feature_types[feature_type] = []
        feature_types[feature_type].append(i)
    
    # Calculate average importance by type
    type_importance = {}
    for feature_type, indices in feature_types.items():
        avg_importance = np.mean(mi_scores[indices])
        type_importance[feature_type] = avg_importance
    
    plt.bar(type_importance.keys(), type_importance.values(), color='skyblue')
    plt.title('Average Feature Importance by Type')
    plt.ylabel('Average Mutual Information')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6. Performance vs number of features
    plt.subplot(3, 3, 6)
    n_features_list = [5, 10, 15, 20, 25, 30, 40, 51]
    subset_results = test_feature_subset_performance(X, y, feature_names, mi_scores, n_features_list)
    
    n_features = [r['n_features'] for r in subset_results]
    scores = [r['mean_score'] for r in subset_results]
    errors = [r['std_score'] for r in subset_results]
    
    plt.errorbar(n_features, scores, yerr=errors, marker='o', capsize=5)
    plt.xlabel('Number of Top Features')
    plt.ylabel('Cross-validation Accuracy')
    plt.title('Performance vs Feature Count')
    plt.grid(True, alpha=0.3)
    
    # 7. Feature importance distribution
    plt.subplot(3, 3, 7)
    plt.hist(mi_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Mutual Information Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Feature Importance Scores')
    plt.grid(True, alpha=0.3)
    
    # 8. Correlation vs importance scatter
    plt.subplot(3, 3, 8)
    # Calculate average correlation for each feature
    avg_corr = np.mean(np.abs(analysis_results['correlation_matrix']), axis=1)
    plt.scatter(mi_scores, avg_corr, alpha=0.6)
    plt.xlabel('Feature Importance (Mutual Information)')
    plt.ylabel('Average Absolute Correlation')
    plt.title('Feature Importance vs Correlation')
    plt.grid(True, alpha=0.3)
    
    # 9. Summary statistics
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Calculate summary statistics
    total_features = len(feature_names)
    high_importance_features = np.sum(mi_scores > np.percentile(mi_scores, 75))
    redundant_features = len(analysis_results['high_corr_features'])
    
    summary_text = f"""
    Data Summary:
    
    Total Features: {total_features}
    High Importance: {high_importance_features}
    Redundant: {redundant_features}
    
    Recommendations:
    
    1. Consider using top {min(25, total_features//2)} features
    2. Remove {redundant_features} redundant features
    3. Focus on features with MI > {np.percentile(mi_scores, 75):.3f}
    4. Check class balance (ratio: {analysis_results.get('imbalance_ratio', 'N/A'):.1f})
    """
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    viz_dir = os.path.join(project_root, 'data', 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    save_path = os.path.join(viz_dir, 'data_diagnosis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Diagnostic plots saved to '{save_path}'")
    
    return subset_results

def main():
    """Main diagnostic pipeline."""
    print("🔍 Data Diagnostic Analysis")
    print("=" * 50)
    
    try:
        # Load data
        print("📊 Loading data...")
        # Fix path to work from model/ directory
        loader = DataLoader(data_path='../data/final_preprocessed_dataset.csv')
        X_train, X_test, y_train, y_test, feature_names = loader.prepare_all()
        
        # Combine train and test for overall analysis
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        
        print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Run analyses
        print("\n" + "="*50)
        importance_analysis = analyze_feature_importance(X, y, feature_names)
        
        print("\n" + "="*50)
        class_analysis = analyze_class_distribution(y, y_train, y_test)
        
        print("\n" + "="*50)
        correlation_analysis = analyze_feature_correlations(X, feature_names)
        
        # Combine results
        analysis_results = {
            **importance_analysis,
            **class_analysis,
            **correlation_analysis
        }
        
        print("\n" + "="*50)
        subset_results = create_diagnostic_plots(X, y, feature_names, analysis_results)
        
        # Final recommendations
        print("\n" + "="*50)
        print("💡 DIAGNOSTIC RECOMMENDATIONS:")
        print("="*50)
        
        # Feature selection recommendation
        best_subset = max(subset_results, key=lambda x: x['mean_score'])
        print(f"1. OPTIMAL FEATURE COUNT: Use top {best_subset['n_features']} features")
        print(f"   Expected accuracy improvement: {best_subset['mean_score']:.4f}")
        
        # Redundancy reduction
        redundant_count = len(correlation_analysis['high_corr_features'])
        if redundant_count > 0:
            print(f"2. REDUNDANCY REDUCTION: Remove {redundant_count} redundant features")
            print(f"   This could improve training speed and reduce overfitting")
        
        # Class balance
        imbalance_ratio = class_analysis['imbalance_ratio']
        if imbalance_ratio > 2:
            print(f"3. CLASS IMBALANCE: Current ratio is {imbalance_ratio:.1f}:1")
            print(f"   Consider: SMOTE, class weights, or balanced sampling")
        
        # Architecture recommendation
        print(f"4. ARCHITECTURE: Current 51→128→64→3 might be too complex")
        print(f"   Try: 51→{best_subset['n_features']}→32→3 for better generalization")
        
        # Training strategy
        print(f"5. TRAINING STRATEGY:")
        print(f"   - Use ReLU activation (better than sigmoid)")
        print(f"   - Add dropout (0.3-0.4) to prevent overfitting")
        print(f"   - Use momentum (0.9) for faster convergence")
        print(f"   - Reduce learning rate to 0.01-0.02")
        
        print(f"\n🎯 EXPECTED IMPROVEMENT: {best_subset['mean_score'] - 0.63:.3f} accuracy increase")
        print(f"   Target accuracy: {best_subset['mean_score']:.1%}")
        
        return analysis_results, subset_results
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    analysis_results, subset_results = main()
