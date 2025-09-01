import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import calculate_class_weights

# Simple data loading without visualizations
print("📊 Loading dataset...")
df = pd.read_csv('../data/final_preprocessed_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Mood distribution:\n{df['mood'].value_counts()}")

# Prepare features and target
print("\n🔧 Preparing features and target...")
feature_columns = [col for col in df.columns if col != 'mood']
X = df[feature_columns].values
y = df['mood'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Feature columns: {len(feature_columns)}")
print(f"Target classes: {np.unique(y)}")
print(f"Encoded classes: {label_encoder.classes_}")

# Standardize features
print("\n📏 Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
print("\n✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training class distribution: {np.bincount(y_train)}")

# Now test class weights
print("\n" + "="*50)
print("TESTING CLASS WEIGHT CALCULATION")
print("="*50)

# Test different weight calculation methods
methods = ['balanced', 'inverse', 'sqrt_inverse']

for method in methods:
    print(f"\n🎯 Testing {method.upper()} method:")
    print("-" * 30)
    weights = calculate_class_weights(y_train, method=method)
    
    # Show the weight range
    weight_values = list(weights.values())
    print(f"  Weight range: {min(weight_values):.2f} - {max(weight_values):.2f}")
    print(f"  Weight ratio (max/min): {max(weight_values)/min(weight_values):.2f}")
    
    # Show specific weights for minority classes
    print(f"  Minority class weights:")
    for class_label in [1, 3]:  # High_Low and Low_High (only 12 samples each)
        if class_label in weights:
            print(f"    Class {class_label}: {weights[class_label]:.2f}") 