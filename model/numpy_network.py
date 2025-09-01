#!/usr/bin/env python3
"""
Enhanced Pure Numpy Neural Network Implementation
Built from scratch for learning purposes - no external DL frameworks!

This implements:
- Forward pass with ReLU activation (better than sigmoid for deep networks)
- Backward pass with manual gradient calculation
- Mini-batch gradient descent with momentum
- Dropout regularization to prevent overfitting
- Gradient clipping for stability
- 3-class classification with softmax output
- Enhanced weight initialization
"""

import numpy as np
from sklearn.metrics import classification_report, accuracy_score

class NumpyNeuralNetwork:
    """
    An enhanced neural network implemented entirely in numpy.
    Architecture: Input -> Hidden -> Hidden -> Output
    """
    
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01, 
                 momentum=0.9, dropout_rate=0.3, class_weights=None, use_relu=True):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Number of input features (51 for your audio features)
            hidden1_size (int): Number of neurons in first hidden layer (128)
            hidden2_size (int): Number of neurons in second hidden layer (64)
            output_size (int): Number of output classes (3: High, Low, Medium)
            learning_rate (float): Learning rate for gradient descent (reduced from 0.1)
            momentum (float): Momentum coefficient for faster convergence
            dropout_rate (float): Dropout rate for regularization
            class_weights (dict): Class weights for handling imbalance
            use_relu (bool): Use ReLU instead of sigmoid activation
        """
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout_rate = dropout_rate
        self.class_weights = class_weights
        self.use_relu = use_relu
        
        # Initialize weights and biases with He initialization (better for ReLU)
        if use_relu:
            self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
            self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
            self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
        else:
            # Xavier initialization for sigmoid
            self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(1.0 / input_size)
            self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(1.0 / hidden1_size)
            self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(1.0 / hidden2_size)
        
        self.b1 = np.zeros((1, hidden1_size))
        self.b2 = np.zeros((1, hidden2_size))
        self.b3 = np.zeros((1, output_size))
        
        # Initialize momentum velocities
        self.vW1 = np.zeros_like(self.W1)
        self.vW2 = np.zeros_like(self.W2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb1 = np.zeros_like(self.b1)
        self.vb2 = np.zeros_like(self.b2)
        self.vb3 = np.zeros_like(self.b3)
        
        print(f"✅ Enhanced Neural Network initialized!")
        print(f"   Input layer: {input_size} features")
        print(f"   Hidden layer 1: {hidden1_size} neurons")
        print(f"   Hidden layer 2: {hidden2_size} neurons")
        print(f"   Output layer: {output_size} classes")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Momentum: {momentum}")
        print(f"   Dropout rate: {dropout_rate}")
        print(f"   Activation: {'ReLU' if use_relu else 'Sigmoid'}")
        if class_weights:
            print(f"   Class weights: {class_weights}")
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function."""
        return np.where(x > 0, 1, 0)
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)
    
    def softmax(self, x):
        """Softmax activation for output layer."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def dropout(self, x, rate, training=True):
        """Apply dropout during training."""
        if training and rate > 0:
            mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
            return x * mask
        return x
    
    def forward(self, X, training=True):
        """
        Forward pass through the network.
        
        Args:
            X (np.array): Input data of shape (batch_size, input_size)
            training (bool): Whether in training mode (for dropout)
            
        Returns:
            tuple: (hidden1_output, hidden2_output, final_output)
        """
        # Hidden layer 1: Input -> 128 neurons
        self.z1 = np.dot(X, self.W1) + self.b1  # Linear transformation
        
        if self.use_relu:
            self.a1 = self.relu(self.z1)          # ReLU activation
        else:
            self.a1 = self.sigmoid(self.z1)       # Sigmoid activation
        
        # Apply dropout to first hidden layer
        self.a1 = self.dropout(self.a1, self.dropout_rate, training)
        
        # Hidden layer 2: 128 -> 64 neurons
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Linear transformation
        
        if self.use_relu:
            self.a2 = self.relu(self.z2)                # ReLU activation
        else:
            self.a2 = self.sigmoid(self.z2)             # Sigmoid activation
        
        # Apply dropout to second hidden layer
        self.a2 = self.dropout(self.a2, self.dropout_rate, training)
        
        # Output layer: 64 -> 3 classes
        self.z3 = np.dot(self.a2, self.W3) + self.b3  # Linear transformation
        self.a3 = self.softmax(self.z3)                # Softmax activation
        
        return self.a1, self.a2, self.a3
    
    def backward(self, X, y, hidden1_output, hidden2_output, final_output, sample_weights=None):
        """
        Backward pass to compute gradients.

        Args:
            X (np.array): Input data
            y (np.array): True labels (one-hot encoded)
            hidden1_output (np.array): Output from first hidden layer
            hidden2_output (np.array): Output from second hidden layer
            final_output (np.array): Final network output
            sample_weights (np.array|None): shape (batch_size,) weight for each sample
        """
        batch_size = X.shape[0]

        # Output layer gradients (layer 3)
        delta3 = final_output - y  # Error at output layer

        # Apply per-sample weighting if provided (broadcast across classes)
        if sample_weights is not None:
            delta3 = delta3 * sample_weights[:, None]

        # Hidden layer 2 gradients (layer 2)
        if self.use_relu:
            delta2 = np.dot(delta3, self.W3.T) * self.relu_derivative(hidden2_output)
        else:
            delta2 = np.dot(delta3, self.W3.T) * self.sigmoid_derivative(hidden2_output)

        # Hidden layer 1 gradients (layer 1)
        if self.use_relu:
            delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(hidden1_output)
        else:
            delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(hidden1_output)

        # Compute gradients for weights and biases
        dW3 = np.dot(hidden2_output.T, delta3) / batch_size
        db3 = np.sum(delta3, axis=0, keepdims=True) / batch_size

        dW2 = np.dot(hidden1_output.T, delta2) / batch_size
        db2 = np.sum(delta2, axis=0, keepdims=True) / batch_size

        dW1 = np.dot(X.T, delta1) / batch_size
        db1 = np.sum(delta1, axis=0, keepdims=True) / batch_size

        return dW1, db1, dW2, db2, dW3, db3
    
    def clip_gradients(self, gradients, max_norm=1.0):
        """Clip gradients to prevent exploding gradients."""
        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            return [g * clip_coef for g in gradients]
        return gradients
    
    def update_weights(self, dW1, db1, dW2, db2, dW3, db3):
        """Update weights and biases using momentum and gradient clipping."""
        # Clip gradients
        gradients = [dW1, db1, dW2, db2, dW3, db3]
        clipped_gradients = self.clip_gradients(gradients, max_norm=1.0)
        dW1, db1, dW2, db2, dW3, db3 = clipped_gradients
        
        # Update with momentum
        self.vW1 = self.momentum * self.vW1 + self.learning_rate * dW1
        self.vb1 = self.momentum * self.vb1 + self.learning_rate * db1
        
        self.vW2 = self.momentum * self.vW2 + self.learning_rate * dW2
        self.vb2 = self.momentum * self.vb2 + self.learning_rate * db2
        
        self.vW3 = self.momentum * self.vW3 + self.learning_rate * dW3
        self.vb3 = self.momentum * self.vb3 + self.learning_rate * db3
        
        # Apply updates
        self.W1 -= self.vW1
        self.b1 -= self.vb1
        self.W2 -= self.vW2
        self.b2 -= self.vb2
        self.W3 -= self.vW3
        self.b3 -= self.vb3
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=True, 
              early_stopping=True, patience=10, min_delta=0.001, lr_schedule=True,
              resample_method=None, resample_kwargs=None, balanced_batches=True):
        """
        Train the neural network using mini-batch gradient descent.

        New args:
            resample_method: None | 'oversample' | 'smote'
            resample_kwargs: dict forwarded to resampler
            balanced_batches: bool - use balanced batches each epoch
        """
        # Optional resampling BEFORE training
        if resample_method:
            if resample_kwargs is None:
                resample_kwargs = {}
            if resample_method == 'oversample':
                X_train, y_train = self._oversample_minority(X_train, y_train, **resample_kwargs)
            elif resample_method == 'smote':
                X_train, y_train = self._resample_smote(X_train, y_train, **resample_kwargs)
            else:
                raise ValueError("resample_method must be one of: None, 'oversample', 'smote'")

        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        train_losses = []
        val_accuracies = []
        
        # Early stopping variables
        best_val_acc = 0
        patience_counter = 0
        best_weights = None
        
        # Learning rate scheduling
        initial_lr = self.learning_rate
        
        print(f"🚀 Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Training samples: {n_samples}")
        print(f"   Batches per epoch: {n_batches}")
        if early_stopping:
            print(f"   Early stopping: Enabled (patience: {patience})")
        if lr_schedule:
            print(f"   Learning rate scheduling: Enabled")
        
        for epoch in range(epochs):
            # Learning rate scheduling
            if lr_schedule:
                if epoch > 0 and epoch % 20 == 0:
                    self.learning_rate *= 0.9
                    if verbose:
                        print(f"   📉 Learning rate reduced to: {self.learning_rate:.6f}")
            
            # Prepare batches for this epoch
            if balanced_batches:
                batch_indices_list = self._balanced_batch_indices(y_train, batch_size, random_state=None)
            else:
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
                # ensure integer indices (np.arange sometimes yields non-int dtype in some contexts)
                batch_indices_list = [np.arange(i*batch_size, (i+1)*batch_size, dtype=int) for i in range(n_samples // batch_size)]
 
            epoch_loss = 0
            
            # Mini-batch training
            for idx_array in batch_indices_list:
                # guard: make sure indices are integer numpy array
                idx_array = np.asarray(idx_array).astype(int)
                if balanced_batches:
                    X_batch = X_train[idx_array]
                    y_batch = y_train[idx_array]
                else:
                    X_batch = X_shuffled[idx_array]
                    y_batch = y_shuffled[idx_array]
                
                # Forward pass (training mode for dropout)
                hidden1_output, hidden2_output, final_output = self.forward(X_batch, training=True)
                
                # Compute weighted loss (cross-entropy with class weights)
                epsilon = 1e-15  # Small value to prevent log(0)

                if self.class_weights:
                    # class weight vector (index -> weight)
                    weight_vec = np.array([self.class_weights.get(i, 1.0) for i in range(self.output_size)], dtype=float)
                    # per-sample weight determined by true class
                    sample_weights = np.sum(y_batch * weight_vec[None, :], axis=1)  # shape (batch_size,)
                    # normalize so mean(sample_weights) == 1 -> keeps gradient scale stable
                    sample_weights = sample_weights / (np.mean(sample_weights) + 1e-12)

                    # weighted per-sample cross-entropy
                    per_sample_ce = -np.sum(y_batch * np.log(final_output + epsilon), axis=1)
                    loss = np.mean(sample_weights * per_sample_ce)
                else:
                    sample_weights = None
                    loss = -np.mean(np.sum(y_batch * np.log(final_output + epsilon), axis=1))

                epoch_loss += loss

                # Backward pass (pass sample_weights to scale gradients consistently)
                dW1, db1, dW2, db2, dW3, db3 = self.backward(
                    X_batch, y_batch, hidden1_output, hidden2_output, final_output, sample_weights=sample_weights
                )
                
                # Update weights
                self.update_weights(dW1, db1, dW2, db2, dW3, db3)
            
            # Validation
            val_accuracy = self.evaluate(X_val, y_val)
            
            train_losses.append(epoch_loss / n_batches)
            val_accuracies.append(val_accuracy)
            
            # Early stopping logic
            if early_stopping:
                if val_accuracy > best_val_acc + min_delta:
                    best_val_acc = val_accuracy
                    patience_counter = 0
                    best_weights = self.get_weights()
                    if verbose:
                        print(f"Epoch {epoch + 1:3d}/{epochs} - Loss: {epoch_loss/n_batches:.4f} - Val Acc: {val_accuracy:.4f} ✨ NEW BEST!")
                else:
                    patience_counter += 1
                    if verbose and (epoch + 1) % 10 == 0:
                        print(f"Epoch {epoch + 1:3d}/{epochs} - Loss: {epoch_loss/n_batches:.4f} - Val Acc: {val_accuracy:.4f}")
                    
                    if patience_counter >= patience:
                        print(f"🛑 Early stopping triggered! No improvement for {patience} epochs.")
                        print(f"   Best validation accuracy: {best_val_acc:.4f}")
                        # Restore best weights
                        if best_weights:
                            self.set_weights(best_weights)
                        break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1:3d}/{epochs} - Loss: {epoch_loss/n_batches:.4f} - Val Acc: {val_accuracy:.4f}")
        
        if not early_stopping or patience_counter < patience:
            print(f"✅ Training completed!")
            if early_stopping and best_weights:
                print(f"   Best validation accuracy: {best_val_acc:.4f}")
        
        # Reset learning rate
        self.learning_rate = initial_lr
        
        return train_losses, val_accuracies
    
    def predict(self, X):
        """Make predictions on new data."""
        _, _, output = self.forward(X, training=False)  # No dropout during inference
        return np.argmax(output, axis=1)
    
    def evaluate(self, X, y):
        """Evaluate accuracy on given data."""
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        return accuracy_score(true_labels, predictions)
    
    def get_weights(self):
        """Get current weights and biases."""
        return {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3
        }
    
    def set_weights(self, weights):
        """Set weights and biases from a dictionary."""
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']
        self.W3 = weights['W3']
        self.b3 = weights['b3']

    # -------------------------
    # Resampling / batching helpers
    # -------------------------
    def _oversample_minority(self, X, y_onehot, random_state=None):
        """Deterministic oversample minority classes to match majority count."""
        from collections import Counter
        rng = np.random.default_rng(random_state)
        y_int = np.argmax(y_onehot, axis=1)
        counts = Counter(y_int)
        max_count = max(counts.values())
        X_chunks = []
        y_chunks = []
        for cls in sorted(counts.keys()):
            idx = np.where(y_int == cls)[0]
            if len(idx) == 0:
                continue
            reps = int(np.ceil(max_count / len(idx)))
            sel = np.tile(idx, reps)[:max_count]
            perm = rng.permutation(sel)
            X_chunks.append(X[perm])
            y_chunks.append(y_onehot[perm])
        X_res = np.vstack(X_chunks)
        y_res = np.vstack(y_chunks)
        perm_all = rng.permutation(X_res.shape[0])
        return X_res[perm_all], y_res[perm_all]

    def _resample_smote(self, X, y_onehot, random_state=42):
        """SMOTE resampling (requires imblearn). Returns X_res, y_res_onehot."""
        try:
            from imblearn.over_sampling import SMOTE
        except Exception as e:
            raise ImportError("imblearn not installed. Install with: pip install imbalanced-learn") from e
        y_int = np.argmax(y_onehot, axis=1)
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X, y_int)
        y_res_onehot = np.eye(self.output_size)[y_res]
        return X_res, y_res_onehot

    def _balanced_batch_indices(self, y_onehot, batch_size, random_state=None):
        """
        Create balanced batches indices so each batch contains examples from all classes.
        Returns list of index arrays for one epoch.
        """
        rng = np.random.default_rng(random_state)
        y_int = np.argmax(y_onehot, axis=1)
        classes = np.unique(y_int)
        idx_by_class = {c: list(np.where(y_int == c)[0]) for c in classes}
        for c in classes:
            rng.shuffle(idx_by_class[c])
        batches = []
        num_classes = len(classes)
        per_class = max(1, batch_size // num_classes)
        # continue building batches until no class can supply per_class without replacement;
        # if class runs out we sample with replacement from that class to keep batches balanced
        pointers = {c: 0 for c in classes}
        while True:
            batch = []
            for c in classes:
                start = pointers[c]
                end = start + per_class
                cls_idx = idx_by_class[c]
                if end <= len(cls_idx):
                    batch.extend(cls_idx[start:end])
                    pointers[c] = end
                else:
                    # sample with replacement if exhausted
                    need = per_class
                    if len(cls_idx) == 0:
                        continue
                    pick = list(rng.choice(cls_idx, size=need, replace=True))
                    batch.extend(pick)
                    pointers[c] = len(cls_idx)  # mark exhausted
            if len(batch) == 0:
                break
            # fill remainder randomly if batch shorter than batch_size
            if len(batch) < batch_size:
                remaining = np.hstack([idx_by_class[c][pointers[c]:] for c in classes])
                if remaining.size > 0:
                    extra_needed = batch_size - len(batch)
                    extra = list(rng.choice(remaining, size=extra_needed, replace=True))
                    batch.extend(extra)
                else:
                    # all exhausted, but we still have current batch — allow smaller final batch
                    pass
            batches.append(np.array(batch))
            # stop when all pointers at end
            if all(pointers[c] >= len(idx_by_class[c]) for c in classes):
                break
        return batches
