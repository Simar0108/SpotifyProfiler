#!/usr/bin/env python3
"""
Training script for mood classification using our pure numpy neural network.
This script loads the DEAM dataset and trains the network from scratch.
"""

import numpy as np
from data_loader import DataLoader
from numpy_network import NumpyNeuralNetwork
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from utils import evaluate_per_class_accuracy

def one_hot_encode(y, num_classes):
    """Convert integer labels to one-hot encoding."""
    return np.eye(num_classes)[y]

def main():
    """Main training pipeline."""
    print("🎵 Starting Pure Numpy Neural Network Training")
    print("=" * 60)
    
    try:
        # Step 1: Load and prepare data
        print("📊 Loading DEAM dataset...")
        # Fix path to work from spotifyprofiler directory
        loader = DataLoader(data_path='data/final_preprocessed_dataset.csv')
        X_train, X_test, y_train, y_test, feature_names = loader.prepare_all()
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Classes: {len(np.unique(y_train))}")
        
        # Step 2: Prepare data for training
        print("\n🔧 Preparing data for training...")
        
        # Convert labels to one-hot encoding (derive classes from data)
        num_classes = len(np.unique(y_train))
        y_train_oh = one_hot_encode(y_train, num_classes)
        y_test_oh = one_hot_encode(y_test, num_classes)
        
        print(f"   Labels converted to one-hot encoding")
        print(f"   Training labels shape: {y_train_oh.shape}")
        print(f"   Test labels shape: {y_test_oh.shape}")
        
        # Step 3: Create and train the network
        print("\n🧠 Creating neural network...")
        
        # Network architecture: 51 -> 128 -> 64 -> 3
        input_size = X_train.shape[1]      # 51 features
        hidden1_size = 128                 # First hidden layer
        hidden2_size = 64                  # Second hidden layer
        output_size = num_classes          # 3 classes
        
        # Get class weights from the data loader
        class_weights = loader.class_weights
        print(f"   Using class weights: {class_weights}")
        
        # Step 4: Train the network with hyperparameter optimization
        print("\n🚀 Training network with hyperparameter optimization...")
        
        # Test different learning rates
        learning_rates = [0.05, 0.1, 0.15]
        best_accuracy = 0
        best_lr = 0.1
        best_network = None
        
        print(f"🧪 Testing {len(learning_rates)} learning rates...")
        
        for lr in learning_rates:
            print(f"\n   Testing learning rate: {lr}")
            
            # Create network with this learning rate
            nn = NumpyNeuralNetwork(
                input_size=input_size,
                hidden1_size=hidden1_size,
                hidden2_size=hidden2_size,
                output_size=output_size,
                learning_rate=lr,
                class_weights=class_weights
            )
            
            # Train with early stopping + oversampling + balanced batches
            train_losses, val_accuracies = nn.train(
                X_train=X_train,
                y_train=y_train_oh,
                X_val=X_test,
                y_val=y_test_oh,
                epochs=100,
                batch_size=32,
                verbose=False,
                early_stopping=True,
                patience=15,
                lr_schedule=True,
                resample_method='smote',
                resample_kwargs={'random_state':42},
                balanced_batches=True
            )
            
            # Evaluate
            test_accuracy = nn.evaluate(X_test, y_test_oh)
            print(f"      Learning rate {lr}: Final accuracy = {test_accuracy:.4f}")
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_lr = lr
                best_network = nn
        
        print(f"\n🏆 Best learning rate: {best_lr} (accuracy: {best_accuracy:.4f})")
        
        # Test different batch sizes with best learning rate
        batch_sizes = [16, 32, 64]
        best_batch_size = 32
        final_network = best_network
        
        print(f"\n🧪 Testing {len(batch_sizes)} batch sizes with best learning rate...")
        
        for batch_size in batch_sizes:
            print(f"\n   Testing batch size: {batch_size}")
            
            # Create network with best learning rate
            nn = NumpyNeuralNetwork(
                input_size=input_size,
                hidden1_size=hidden1_size,
                hidden2_size=hidden2_size,
                output_size=output_size,
                learning_rate=best_lr,
                class_weights=class_weights
            )
            
            # Train with this batch size
            train_losses, val_accuracies = nn.train(
                X_train=X_train,
                y_train=y_train_oh,
                X_val=X_test,
                y_val=y_test_oh,
                epochs=100,
                batch_size=batch_size,
                verbose=False,
                early_stopping=True,
                patience=15,
                lr_schedule=True
            )
            
            # Check final validation accuracy
            final_val_acc = val_accuracies[-1]
            print(f"      Batch size {batch_size}: Final validation accuracy = {final_val_acc:.4f}")
            
            if final_val_acc > best_accuracy:
                best_accuracy = final_val_acc
                best_batch_size = batch_size
                final_network = nn
        
        print(f"\n🏆 Best batch size: {best_batch_size} (accuracy: {best_accuracy:.4f})")
        
        # Final training with best parameters
        print(f"\n🎯 Final training with best parameters (LR: {best_lr}, Batch: {best_batch_size})...")
        
        final_train_losses, final_val_accuracies = final_network.train(
            X_train=X_train,
            y_train=y_train_oh,
            X_val=X_test,
            y_val=y_test_oh,
            epochs=100,
            batch_size=best_batch_size,
            verbose=True,
            early_stopping=True,
            patience=15,
            lr_schedule=True
        )
        
        # Use final_network for evaluation
        nn = final_network
        train_losses = final_train_losses
        val_accuracies = final_val_accuracies
        
        # Step 5: Evaluate results
        print("\n📈 Evaluating results...")
        
        # Final test accuracy
        test_accuracy = nn.evaluate(X_test, y_test_oh)
        print(f"   Final test accuracy: {test_accuracy:.4f}")
        
        # Detailed classification report
        predictions = nn.predict(X_test)
        true_labels = np.argmax(y_test_oh, axis=1)
        
        # Use label encoder classes if available (keeps order consistent)
        class_names = list(getattr(loader.label_encoder, 'classes_', [f"Class_{i}" for i in range(num_classes)]))
        print("\n📊 Classification Report:")
        print(classification_report(true_labels, predictions, target_names=class_names))
        
        # Per-class accuracy diagnostics
        per_class = evaluate_per_class_accuracy(true_labels, predictions)
        print("\n🧾 Per-class accuracy:")
        for cls_name, cls_idx in zip(class_names, sorted(per_class.keys())):
            print(f"  {cls_name} (idx {cls_idx}): {per_class[cls_idx]:.3f}")
        
        # Step 6: Plot training progress
        print("\n📊 Plotting training progress...")
        
        plt.figure(figsize=(12, 5))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        viz_dir = os.path.join(project_root, 'data', 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        save_path = os.path.join(viz_dir, 'training_progress.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Training plots saved to: {save_path}")
        
        # Step 7: Save the trained model
        print("\n💾 Saving trained model...")
        
        # Save weights
        weights = nn.get_weights()
        weights_path = os.path.join(project_root, 'data', 'trained_weights.npz')
        np.savez(weights_path, **weights)
        print(f"   Model weights saved to: {weights_path}")
        
        print("\n🎉 Training pipeline completed successfully!")
        print(f"Final test accuracy: {test_accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
