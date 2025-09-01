import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

class DataLoader:
    """
    Handles loading and preprocessing of the DEAM dataset for neural network training.
    
    This class encapsulates all the data preparation steps:
    1. Load the CSV file
    2. Encode categorical mood labels into integers
    3. Separate features from target
    4. Standardize features
    5. Split into train/test sets
    6. Handle class imbalance with weights and class reduction
    """
    
    def __init__(self, data_path='data/final_preprocessed_dataset.csv', use_3_classes=True):
        self.data_path = data_path
        self.use_3_classes = use_3_classes
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.class_weights = None
        
    def load_data(self):
        """
        Load the dataset and perform initial inspection.
        
        Returns:
            pd.DataFrame: The loaded dataset
        """
        print("📊 Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Mood distribution:\n{df['mood'].value_counts()}")
        
        return df
    
    def collapse_to_3_classes(self, df):
        """
        Collapse 9 mood classes (Low_Low, Low_Medium, etc.) into 3 classes (Low, Medium, High)
        based on the overall emotional intensity.
        
        Args:
            df (pd.DataFrame): The loaded dataset
            
        Returns:
            pd.DataFrame: Dataset with collapsed mood classes
        """
        print("\n🔄 Collapsing 9 mood classes to 3 classes...")
        
        # Define mapping from 9 classes to 3 classes
        # This mapping considers both valence and arousal for overall emotional intensity
        mood_mapping = {
            'Low_Low': 'Low',      # Low valence, low arousal = Low emotional intensity
            'Low_Medium': 'Low',   # Low valence, medium arousal = Low emotional intensity
            'Low_High': 'Medium',  # Low valence, high arousal = Medium emotional intensity
            'Medium_Low': 'Low',   # Medium valence, low arousal = Low emotional intensity
            'Medium_Medium': 'Medium', # Medium valence, medium arousal = Medium emotional intensity
            'Medium_High': 'Medium',   # Medium valence, high arousal = Medium emotional intensity
            'High_Low': 'Medium',      # High valence, low arousal = Medium emotional intensity
            'High_Medium': 'High',     # High valence, medium arousal = High emotional intensity
            'High_High': 'High'        # High valence, high arousal = High emotional intensity
        }
        
        # Apply the mapping
        df['mood_3class'] = df['mood'].map(mood_mapping)
        
        # Show the new distribution
        print("Original 9-class distribution:")
        print(df['mood'].value_counts().sort_index())
        print("\nNew 3-class distribution:")
        print(df['mood_3class'].value_counts().sort_index())
        
        # Replace the original mood column if requested
        if self.use_3_classes:
            df['mood'] = df['mood_3class']
            df = df.drop('mood_3class', axis=1)
            print("\n✅ Successfully collapsed to 3 classes!")
        else:
            print("\n📝 Kept original 9 classes, but added 'mood_3class' column")
        
        return df
    
    def prepare_features_and_target(self, df):
        """
        Separate features from target and encode mood labels.
        
        Args:
            df (pd.DataFrame): The loaded dataset
            
        Returns:
            tuple: (X_features, y_target, feature_names)
        """
        print("\n🔧 Preparing features and target...")
        
        # Separate features from target
        # Assuming 'mood' is the target column and everything else is a feature
        feature_columns = [col for col in df.columns if col != 'mood']
        self.feature_names = feature_columns
        
        X = df[feature_columns].values
        y = df['mood'].values
        
        print(f"Feature columns: {len(feature_columns)}")
        print(f"Feature names: {feature_columns[:5]}...")  # Show first 5
        print(f"Target classes: {np.unique(y)}")
        
        # Encode mood labels into integers
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Encoded classes: {self.label_encoder.classes_}")
        print(f"Class mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return X, y_encoded, feature_columns
    
    def calculate_class_weights(self, y_train):
        """
        Calculate balanced class weights to handle class imbalance.
        
        Args:
            y_train (np.array): Training labels
            
        Returns:
            dict: Class weights dictionary
        """
        print("\n⚖️ Calculating class weights...")
        
        # Calculate balanced class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        # Convert to dictionary
        self.class_weights = dict(zip(np.unique(y_train), class_weights))
        
        print("Class weights:")
        for class_id, weight in self.class_weights.items():
            class_name = self.label_encoder.inverse_transform([class_id])[0]
            print(f"  Class {class_id} ({class_name}): {weight:.3f}")
        
        return self.class_weights
    
    def standardize_features(self, X):
        """
        Standardize features to have zero mean and unit variance.
        
        Why do we do this? Neural networks train much better when all features
        are on the same scale. Without standardization, features with larger
        values would dominate the learning process.
        
        Args:
            X (np.array): Feature matrix
            
        Returns:
            np.array: Standardized features
        """
        print("\n📏 Standardizing features...")
        
        # Fit scaler on training data and transform
        X_scaled = self.scaler.fit_transform(X)
        
        # Show before/after statistics
        print(f"Before standardization:")
        print(f"  Mean: {X.mean(axis=0)[:5]}...")  # First 5 features
        print(f"  Std: {X.std(axis=0)[:5]}...")
        
        print(f"After standardization:")
        print(f"  Mean: {X_scaled.mean(axis=0)[:5]}...")
        print(f"  Std: {X_scaled.std(axis=0)[:5]}...")
        
        return X_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
            test_size (float): Proportion for test set
            random_state (int): For reproducible splits
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"\n✂️ Splitting data (train: {1-test_size:.0%}, test: {test_size:.0%})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test

    
    def visualize_data(self, X, y, feature_names):
        """
        Create visualizations to understand our data better.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
            feature_names (list): Names of features
        """
        print("\n📈 Creating data visualizations...")
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature correlation heatmap (first 10 features)
        correlation_matrix = np.corrcoef(X[:, :10].T)
        sns.heatmap(correlation_matrix, 
                   xticklabels=feature_names[:10], 
                   yticklabels=feature_names[:10],
                   ax=axes[0, 0], cmap='coolwarm', center=0)
        axes[0, 0].set_title('Feature Correlation Matrix (First 10 Features)')
        
        # 2. Feature distributions (first 5 features)
        for i in range(5):
            axes[0, 1].hist(X[:, i], alpha=0.7, label=feature_names[i], bins=20)
        axes[0, 1].set_title('Feature Distributions (First 5 Features)')
        axes[0, 1].legend()
        axes[0, 1].set_xlabel('Standardized Value')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Class distribution
        unique, counts = np.unique(y, return_counts=True)
        axes[1, 0].bar(unique, counts)
        axes[1, 0].set_title('Class Distribution')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Count')
        
        # 4. Feature importance (variance)
        feature_variance = np.var(X, axis=0)
        top_features_idx = np.argsort(feature_variance)[-10:]  # Top 10
        axes[1, 1].barh(range(10), feature_variance[top_features_idx])
        axes[1, 1].set_yticks(range(10))
        axes[1, 1].set_yticklabels([feature_names[i] for i in top_features_idx])
        axes[1, 1].set_title('Top 10 Features by Variance')
        axes[1, 1].set_xlabel('Variance')
        
        plt.tight_layout()
        # Save to an absolute path inside the project
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        viz_dir = os.path.join(project_root, 'data', 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        save_path = os.path.join(viz_dir, 'data_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Avoid blocking the training flow in headless/CLI runs
        plt.close(fig)
        
        print(f"✅ Visualizations saved to '{save_path}'")
    
    def prepare_all(self):
        """
        Complete data preparation pipeline.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        print("🚀 Starting complete data preparation pipeline...")
        
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Collapse mood classes if requested
        if self.use_3_classes:
            df = self.collapse_to_3_classes(df)
        
        # Step 3: Prepare features and target
        X, y, feature_names = self.prepare_features_and_target(df)
        
        # Step 4: Standardize features
        X_scaled = self.standardize_features(X)
        
        # Step 5: Split data
        X_train, X_test, y_train, y_test = self.split_data(X_scaled, y)
        
        # Step 6: Visualize data
        self.visualize_data(X_scaled, y, feature_names)

        # Step 7: Analyze class imbalance
        self.analyze_class_imbalance(y_train)
        
        # Step 8: Calculate class weights
        self.calculate_class_weights(y_train)
        
        # Store for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print("\n✅ Data preparation complete!")
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        return X_train, X_test, y_train, y_test, feature_names

    def analyze_class_imbalance(self, y):
        """
        Analyze class distribution and imbalance.
        
        Args:
            y (np.array): Target labels
            
        Returns:
            tuple: (unique_classes, counts, imbalance_ratio)
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        print("\n📊 Class Distribution Analysis:")
        print("-" * 50)
        
        for i, (class_id, count) in enumerate(zip(unique, counts)):
            percentage = (count / total) * 100
            class_name = self.label_encoder.inverse_transform([class_id])[0]
            print(f"Class {class_id} ({class_name}): {count} samples ({percentage:.1f}%)")
        
        # Calculate imbalance ratio
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        
        print(f"\n⚖️ Imbalance Ratio: {imbalance_ratio:.2f}:1")
        print(f"   (Majority class has {imbalance_ratio:.1f}x more samples than minority)")
        
        return unique, counts, imbalance_ratio

# Test the data loader
if __name__ == "__main__":
    loader = DataLoader()
    X_train, X_test, y_train, y_test, feature_names = loader.prepare_all() 