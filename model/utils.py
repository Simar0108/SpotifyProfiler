import numpy as np
from collections import Counter

def calculate_class_weights(y_train, method='balanced'):
    """
    Calculate class weights to handle imbalanced datasets.
    
    This function computes weights that give higher importance to minority classes
    during training, helping the neural network learn from all classes equally.
    
    Args:
        y_train (np.array): Training labels
        method (str): Weight calculation method
            - 'balanced': sklearn-style balanced weights
            - 'inverse': Simple inverse frequency weights
            - 'sqrt_inverse': Square root of inverse frequency
    
    Returns:
        dict: Class weights where key=class_label, value=weight
    """
    class_counts = Counter(y_train)
    n_samples = len(y_train)
    n_classes = len(class_counts)
    
    print(f"📊 Class distribution:")
    for class_label, count in sorted(class_counts.items()):
        print(f"  Class {class_label}: {count} samples ({count/n_samples:.1%})")
    
    if method == 'balanced':
        # sklearn-style balanced weights
        weights = {}
        for class_label in class_counts:
            weights[class_label] = n_samples / (n_classes * class_counts[class_label])
            
    elif method == 'inverse':
        # Simple inverse frequency
        weights = {}
        for class_label in class_counts:
            weights[class_label] = n_samples / class_counts[class_label]
            
    elif method == 'sqrt_inverse':
        # Square root of inverse frequency (less aggressive)
        weights = {}
        for class_label in class_counts:
            weights[class_label] = np.sqrt(n_samples / class_counts[class_label])
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"\n⚖️ Class weights ({method} method):")
    for class_label in sorted(weights.keys()):
        print(f"  Class {class_label}: {weights[class_label]:.2f}")
    
    return weights

def one_hot_encode(y, num_classes):
    """
    Convert integer labels to one-hot encoded format.
    
    Args:
        y (np.array): Integer labels
        num_classes (int): Number of classes
        
    Returns:
        np.array: One-hot encoded labels
    """
    n_samples = len(y)
    y_one_hot = np.zeros((n_samples, num_classes))
    y_one_hot[np.arange(n_samples), y] = 1
    return y_one_hot

def calculate_weighted_loss(y_true, y_pred, class_weights):
    """
    Calculate weighted cross-entropy loss.
    
    This function applies class weights to the cross-entropy loss,
    giving higher penalty for misclassifying minority classes.
    
    Args:
        y_true (np.array): True labels (one-hot encoded)
        y_pred (np.array): Predicted probabilities
        class_weights (dict): Class weights
        
    Returns:
        float: Weighted loss value
    """
    # Standard cross-entropy loss
    epsilon = 1e-15  # Small value to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    cross_entropy = -np.sum(y_true * np.log(y_pred), axis=1)
    
    # Apply class weights
    weighted_loss = 0
    for i, sample_loss in enumerate(cross_entropy):
        # Find which class this sample belongs to
        true_class = np.argmax(y_true[i])
        weight = class_weights.get(true_class, 1.0)
        weighted_loss += weight * sample_loss
    
    return weighted_loss / len(y_true)

def evaluate_per_class_accuracy(y_true, y_pred):
    """
    Calculate accuracy for each class separately.
    
    This is crucial for imbalanced datasets where overall accuracy
    can be misleading.
    
    Args:
        y_true (np.array): True labels (integer format)
        y_pred (np.array): Predicted labels (integer format)
        
    Returns:
        dict: Per-class accuracy
    """
    unique_classes = np.unique(y_true)
    per_class_accuracy = {}
    
    for class_label in unique_classes:
        # Find samples belonging to this class
        mask = (y_true == class_label)
        if np.sum(mask) > 0:
            # Calculate accuracy for this class
            class_accuracy = np.mean(y_pred[mask] == class_label)
            per_class_accuracy[class_label] = class_accuracy
        else:
            per_class_accuracy[class_label] = 0.0
    
    return per_class_accuracy

def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print a detailed classification report.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        class_names (list): Names of classes (optional)
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
    
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\n🔍 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Calculate per-class accuracy
    per_class_acc = evaluate_per_class_accuracy(y_true, y_pred)
    print("\n📊 Per-Class Accuracy:")
    for class_label, accuracy in sorted(per_class_acc.items()):
        class_name = class_names[class_label] if class_label < len(class_names) else f"Class_{class_label}"
        print(f"  {class_name}: {accuracy:.3f}")
    
    return per_class_acc 