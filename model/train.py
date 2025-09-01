import numpy as np
from data_loader import DataLoader
from model import Network, FullyConnectedLayer, SoftmaxLayer, sigmoid
from sklearn.metrics import classification_report

def load_and_prepare_data():
    """
    Load the DEAM dataset and prepare it for training.
    Returns: training_data, test_data, feature_names
    """
    print("🚀 Loading and preparing data...")
    
    # Use your working DataLoader
    loader = DataLoader()
    X_train, X_test, y_train, y_test, feature_names = loader.prepare_all()
    
    print(f"✅ Data loaded successfully!")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {len(np.unique(y_train))}")
    
    return X_train, X_test, y_train, y_test, feature_names

def create_network():
    """
    Create the neural network architecture.
    Returns: network
    """
    print("🚀 Creating network...")

    layers = [
        # Input layer: 51 features -> 128 hidden neurons
        FullyConnectedLayer(n_in=51, n_out=128, activation_fn=sigmoid, p_dropout=0.0),
        # Hidden layer: 128 -> 64
        FullyConnectedLayer(n_in=128, n_out=64, activation_fn=sigmoid, p_dropout=0.0),
        # Output layer: 64 -> 3 classes
        SoftmaxLayer(n_in=64, n_out=3, p_dropout=0.0)
    ]

    mini_batch_size = 32

    net = Network(layers, mini_batch_size)
    print("✅ Network created successfully!")
    print(f"   Architecture: 51 → 128 → 64 → 3")
    print(f"   Mini-batch size: {mini_batch_size}")

    return net

def prepare_training_data(X_train, X_test, y_train, y_test):
    """
    Prepare the training data for the network.
    Returns: training_data, test_data
    """
    print("🚀 Preparing training data...")

    training_data = (X_train, y_train)
    test_data = (X_test, y_test)

    print(f"✅ Training data prepared successfully!")
    print(f"   Training samples: {training_data[0].shape[0]}")
    print(f"   Test samples: {len(test_data)}")

    return training_data, test_data

def train_network(net, training_data, test_data):
    """
    Train the network using stochastic gradient descent.
    """
    print("🚀 Training network...")

    epochs = 30
    mini_batch_size = 32
    learning_rate = 0.1
    lmbda = 0.0

    print(f"📋 Training configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Mini-batch size: {mini_batch_size}")
    print(f"   L2 regularization: {lmbda}")
    
    # Start training!
    net.SGD(
        training_data=training_data,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        eta=learning_rate,
        validation_data=test_data,  # Use test data as validation for now
        test_data=test_data,
        lmbda=lmbda
    )
    
    print("✅ Training completed!")
    print("✅ Network trained successfully!")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {learning_rate}")

def evaluate_network(net, X_test, y_test):
    """
    Evaluate the trained network on test data.
    """
    print("🚀 Evaluating network performance...")
    
    # Get predictions for all test samples
    predictions = []
    for i in range(len(X_test)):
        # Get the output probabilities for this sample
        output = net.layers[-1].output.eval({net.x: X_test[i:i+1]})
        # Convert to class prediction
        pred_class = np.argmax(output)
        predictions.append(pred_class)
    
    predictions = np.array(predictions)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    
    print(f"✅ Evaluation complete!")
    print(f"   Test accuracy: {accuracy:.2%}")
    
    # Show class-wise performance
    from sklearn.metrics import classification_report
    class_names = ['High', 'Low', 'Medium']
    print("\n📈 Detailed Classification Report:")
    print(classification_report(y_test, predictions, target_names=class_names))
    
    return predictions, accuracy

def main():
    """
    Main training pipeline.
    """
    print("🎵 Starting Mood Classification Training Pipeline")
    print("=" * 50)
    
    try:
        # Step 1: Load and prepare data
        X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
        
        # Step 2: Create network
        net = create_network()
        
        # Step 3: Prepare data format
        training_data, test_data = prepare_training_data(X_train, X_test, y_train, y_test)
        
        # Step 4: Train network
        train_network(net, training_data, test_data)
        
        # Step 5: Evaluate results
        predictions, accuracy = evaluate_network(net, X_test, y_test)
        
        print("\n🎉 Training pipeline completed successfully!")
        print(f"Final test accuracy: {accuracy:.2%}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

    
    