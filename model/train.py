#!/usr/bin/env python3
import os
import numpy as np
from sklearn.decomposition import PCA

from data_loader import DataLoader
from model import Network
from utils import one_hot_encode, print_classification_report, calculate_class_weights


def main():
    # 1) Load and preprocess data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'final_preprocessed_dataset.csv')
    loader = DataLoader(data_path=data_path)
    X_train, X_test, y_train, y_test, feature_names = loader.prepare_all()

    # Reduce dimensionality to 50 for the network input
    n_components = min(50, X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    num_features = X_train.shape[1]  # should be 50
    num_classes = len(np.unique(y_train))  # should be 9

    # 2) Shape features to column vectors and one-hot encode labels
    # Our Network expects inputs as column vectors (n_features, 1)
    X_train_cols = [x.reshape(num_features, 1) for x in X_train]
    X_test_cols = [x.reshape(num_features, 1) for x in X_test]

    y_train_oh = one_hot_encode(y_train, num_classes)
    y_test_oh = one_hot_encode(y_test, num_classes)

    # 3) Build a simple network: input -> hidden -> output
    # Start small; you can tune sizes later
    net = Network([num_features, 128, 64, num_classes], use_softmax_ce=True)

    # 4) Class weights to mitigate imbalance
    class_weights = calculate_class_weights(y_train, method='balanced')

    # 5) Gentle oversampling of minority classes (cap at 3x duplication)
    class_counts = {c: int(np.sum(y_train == c)) for c in np.unique(y_train)}
    max_target = max(min(max(class_counts.values()), 3 * min(class_counts.values())), min(class_counts.values()) * 3)
    oversampled = []
    for x_col, y_vec, y_int in zip(X_train_cols, y_train_oh, y_train):
        reps = max(1, min(max_target // max(1, class_counts[y_int]), 3))
        for _ in range(reps):
            oversampled.append((x_col, y_vec.reshape(num_classes, 1)))

    training_data = oversampled
    test_data = list(zip(X_test_cols, y_test))  # evaluate() expects integer labels

    # 6) Train (softmax CE, weighted, with L2 and LR decay)
    epochs = 60
    mini_batch_size = 32
    learning_rate = 0.2
    l2_lambda = 1e-4
    lr_decay = 0.98  # decay per epoch
    net.SGD(
        training_data,
        epochs,
        mini_batch_size,
        learning_rate,
        test_data=test_data,
        class_weights=class_weights,
        l2_lambda=l2_lambda,
        lr_decay=lr_decay,
    )

    # 7) Final evaluation and report
    y_pred = []
    for x in X_test_cols:
        probs = net.feedforward(x)
        y_pred.append(int(np.argmax(probs)))
    y_pred = np.array(y_pred)

    class_names = list(getattr(loader.label_encoder, 'classes_', [f"Class_{i}" for i in range(num_classes)]))
    print_classification_report(y_test, y_pred, class_names=class_names)


if __name__ == "__main__":
    main()



