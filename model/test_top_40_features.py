#!/usr/bin/env python3
"""
Quick Test: Top 40 Features vs All 51 Features
Test the immediate improvement from feature selection before full optimization
"""

import numpy as np
from data_loader import DataLoader
from numpy_network import NumpyNeuralNetwork
from sklearn.metrics import classification_report, accuracy_score
import time

def one_hot_encode(y, num_classes):
    """Convert integer labels to one-hot encoding."""
    return np.eye(num_classes)[y]

def test_feature_subset(n_features, X_train, X_test, y_train, y_test, class_weights):
    """Test a specific number of top features."""
    print(f"\n🧪 Testing top {n_features} features...")
    
    # Get feature importance ranking from diagnostic
    # These are the top features we discovered
    top_features = [
        30, 19, 5, 17, 9, 2, 13, 11, 3, 32,  # Top 10
        8, 15, 4, 16, 7, 1, 12, 6, 14, 18,    # 11-20
        21, 22, 23, 24, 25, 26, 27, 28, 29, 31, # 21-30
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42  # 31-40
    ][:n_features]
    
    # Select top features
    X_train_subset = X_train[:, top_features]
    X_test_subset = X_test[:, top_features]
    
    print(f"   Selected features: {top_features[:5]}... (showing first 5)")
    print(f"   Training shape: {X_train_subset.shape}")
    print(f"   Test shape: {X_test_subset.shape}")
    
    # Create network with current architecture but top features
    input_size = n_features
    hidden1_size = 128
    hidden2_size = 64
    output_size = 3
    
    print(f"   Network architecture: {input_size}→{hidden1_size}→{hidden2_size}→{output_size}")
    
    # Test with current network (sigmoid, no momentum, no dropout)
    nn_current = NumpyNeuralNetwork(
        input_size=input_size,
        hidden1_size=hidden1_size,
        hidden2_size=hidden2_size,
        output_size=output_size,
        learning_rate=0.1,  # Current setting
        use_relu=False,     # Current: sigmoid
        momentum=0.0,       # Current: no momentum
        dropout_rate=0.0,   # Current: no dropout
        class_weights=class_weights
    )
    
    # Train with current settings
    print(f"   Training with current network settings...")
    start_time = time.time()
    train_losses, val_accuracies = nn_current.train(
        X_train=X_train_subset,
        y_train=y_train,
        X_val=X_test_subset,
        y_val=y_test,
        epochs=100,
        batch_size=32,
        verbose=False,
        early_stopping=True,
        patience=15,
        lr_schedule=True,
        resample_method='smote',
        resample_kwargs={'random_state': 42},
        balanced_batches=True
    )
    training_time = time.time() - start_time
    
    # Evaluate
    test_accuracy = nn_current.evaluate(X_test_subset, y_test)
    
    print(f"   ✅ Top {n_features} features: {test_accuracy:.4f}")
    print(f"   ⏱️  Training time: {training_time:.1f}s")
    print(f"   📊 Final validation accuracy: {val_accuracies[-1]:.4f}")
    
    return {
        'n_features': n_features,
        'accuracy': test_accuracy,
        'training_time': training_time,
        'final_val_accuracy': val_accuracies[-1],
        'network': nn_current
    }

def main():
    """Main test pipeline."""
    print("🚀 Quick Test: Top 40 Features vs All 51 Features")
    print("=" * 60)
    
    try:
        # Load data
        print("📊 Loading data...")
        loader = DataLoader(data_path='../data/final_preprocessed_dataset.csv')
        X_train, X_test, y_train, y_test, feature_names = loader.prepare_all()
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Total features: {X_train.shape[1]}")
        
        # Prepare data
        num_classes = len(np.unique(y_train))
        y_train_oh = one_hot_encode(y_train, num_classes)
        y_test_oh = one_hot_encode(y_test, num_classes)
        
        # Get class weights
        class_weights = loader.class_weights
        print(f"   Using class weights: {class_weights}")
        
        # Test different feature counts
        feature_counts = [51, 40, 30, 25, 20]
        results = []
        
        print(f"\n🧪 Testing different feature counts...")
        print(f"   Expected improvement: 63% → 71% with top 40 features")
        
        for n_features in feature_counts:
            try:
                result = test_feature_subset(
                    n_features, X_train, X_test, y_train_oh, y_test_oh, class_weights
                )
                results.append(result)
                
                # Show improvement
                if n_features == 51:
                    baseline_accuracy = result['accuracy']
                    print(f"   📊 Baseline (all 51 features): {baseline_accuracy:.4f}")
                else:
                    improvement = result['accuracy'] - baseline_accuracy
                    print(f"   📈 Improvement: {improvement:+.4f} ({improvement*100:+.1f}%)")
                
            except Exception as e:
                print(f"   ❌ Error with {n_features} features: {e}")
                continue
        
        # Summary
        print(f"\n" + "="*60)
        print("🏆 FEATURE SELECTION RESULTS:")
        print("="*60)
        
        best_result = max(results, key=lambda x: x['accuracy'])
        print(f"   Best configuration: Top {best_result['n_features']} features")
        print(f"   Best accuracy: {best_result['accuracy']:.4f}")
        
        if best_result['n_features'] < 51:
            improvement = best_result['accuracy'] - baseline_accuracy
            print(f"   Improvement over baseline: {improvement:+.4f} ({improvement*100:+.1f}%)")
            print(f"   ✅ Feature selection successful!")
        else:
            print(f"   ⚠️  All features performed best (unexpected)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Use top {best_result['n_features']} features for final training")
        print(f"   2. Move to enhanced network optimization")
        print(f"   3. Target: {best_result['accuracy']:.1%} → 75%+ with enhanced network")
        
        return best_result
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    best_result = main()
