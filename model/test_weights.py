from data_loader import DataLoader
from utils import calculate_class_weights

# Load the data
loader = DataLoader()
X_train, X_test, y_train, y_test, feature_names = loader.prepare_all()

# Calculate class weights
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