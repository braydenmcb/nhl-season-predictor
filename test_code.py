"""
Test script for NHL Season Predictor
Run this to test individual functions and verify everything works.
"""

import pandas as pd
import numpy as np
from data_processor import load_data, handle_missing_data, get_training_data, process_data
from engineering.predictor import gradient_descent, predict
from engineering.evaluator import calculate_loss


def test_load_data():
    """Test loading data files"""
    print("=" * 60)
    print("TEST 1: Loading Data")
    print("=" * 60)
    
    try:
        X, y = load_data(2023)
        print(f"✓ Successfully loaded training data: {len(X)} rows")
        print(f"  Columns: {list(X.columns)[:10]}...")  # Show first 10 columns
        print(f"✓ Successfully loaded actual results: {len(y)} rows")
        print(f"  Columns: {list(y.columns)[:10]}...")
        return X, y
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None, None


def test_missing_data_handling(X):
    """Test missing data handling"""
    print("\n" + "=" * 60)
    print("TEST 2: Missing Data Handling")
    print("=" * 60)
    
    if X is None:
        print("✗ Skipping - no data loaded")
        return None
    
    # Check for missing values before handling
    print("\nMissing values BEFORE handling:")
    missing_stats = ['HIT', 'TAKE', 'GIVE']
    for stat in missing_stats:
        if stat in X.columns:
            missing_count = X[stat].isna().sum()
            print(f"  {stat}: {missing_count} missing values")
    
    # Handle missing data
    try:
        X_processed = handle_missing_data(X.copy())
        print("\n✓ Successfully processed missing data")
        
        # Check for missing values after handling
        print("\nMissing values AFTER handling:")
        for stat in missing_stats:
            if stat in X_processed.columns:
                missing_count = X_processed[stat].isna().sum()
                print(f"  {stat}: {missing_count} missing values")
                if missing_count > 0:
                    print(f"    ⚠ Warning: Still has {missing_count} missing values")
        
        return X_processed
    except Exception as e:
        print(f"✗ Error handling missing data: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_get_training_data(X):
    """Test creating training data"""
    print("\n" + "=" * 60)
    print("TEST 3: Creating Training Data")
    print("=" * 60)
    
    if X is None:
        print("✗ Skipping - no data loaded")
        return None
    
    try:
        training_data = get_training_data(X)
        print(f"✓ Successfully created training data: {len(training_data)} rows")
        print(f"  Columns: {list(training_data.columns)[:10]}...")
        
        # Check for NaN values
        nan_counts = training_data.isna().sum()
        if nan_counts.sum() > 0:
            print("\n⚠ Warning: Training data has NaN values:")
            print(nan_counts[nan_counts > 0])
        else:
            print("✓ No NaN values in training data")
        
        # Show sample
        print("\nSample training data (first 3 rows):")
        print(training_data.head(3))
        
        return training_data
    except Exception as e:
        print(f"✗ Error creating training data: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_feature_creation(training_data):
    """Test that features are created correctly"""
    print("\n" + "=" * 60)
    print("TEST 4: Feature Creation")
    print("=" * 60)
    
    if training_data is None:
        print("✗ Skipping - no training data")
        return False
    
    required_features = [
        'Points_Per_Game', 'Goals_Per_Game', 'Assists_Per_Game',
        'Hits', 'Takeaways', 'Giveaways'
    ]
    
    missing_features = [f for f in required_features if f not in training_data.columns]
    
    if missing_features:
        print(f"✗ Missing features: {missing_features}")
        return False
    
    print("✓ All required features present")
    
    # Check for infinite or invalid values
    feature_cols = [col for col in training_data.columns if col not in ['Player', 'Season', 'Next_Season', 'Position']]
    for col in feature_cols[:5]:  # Check first 5 features
        if training_data[col].dtype in [np.float64, np.int64]:
            inf_count = np.isinf(training_data[col]).sum()
            if inf_count > 0:
                print(f"  ⚠ Warning: {col} has {inf_count} infinite values")
    
    print("✓ Features look valid")
    return True


def test_gradient_descent():
    """Test gradient descent function"""
    print("\n" + "=" * 60)
    print("TEST 5: Gradient Descent")
    print("=" * 60)
    
    try:
        # Create simple test data
        X_test = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_test = np.array([[2], [3], [4], [5]])
        
        predictor, weights = gradient_descent(X_test, y_test, epochs=100, step_size=0.01)
        predictions = predictor(X_test)
        
        print(f"✓ Gradient descent completed")
        print(f"  Input shape: {X_test.shape}")
        print(f"  Output shape: {predictions.shape}")
        print(f"  Weights shape: {weights.shape}")
        print(f"  Sample predictions: {predictions[:2].flatten()}")
        
        return True
    except Exception as e:
        print(f"✗ Error in gradient descent: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predict_function(training_data):
    """Test prediction function"""
    print("\n" + "=" * 60)
    print("TEST 6: Prediction Function")
    print("=" * 60)
    
    if training_data is None or len(training_data) == 0:
        print("✗ Skipping - no training data")
        return None
    
    try:
        # Use a small subset for testing
        test_subset = training_data.head(20).copy()
        
        # Need to add 'Player Name' column if it doesn't exist
        if 'Player Name' not in test_subset.columns and 'Player' in test_subset.columns:
            test_subset['Player Name'] = test_subset['Player']
        
        predictions = predict(test_subset)
        
        print(f"✓ Successfully generated predictions: {len(predictions)} rows")
        print(f"  Prediction columns: {list(predictions.columns)[:5]}...")
        print("\nSample predictions:")
        print(predictions.head(3))
        
        return predictions
    except Exception as e:
        print(f"✗ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_evaluator():
    """Test evaluation functions"""
    print("\n" + "=" * 60)
    print("TEST 7: Evaluation Functions")
    print("=" * 60)
    
    try:
        # Create test data
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([11, 19, 31, 39, 51])
        
        metrics = calculate_loss(y_true, y_pred)
        
        print("✓ Successfully calculated metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("NHL SEASON PREDICTOR - TEST SUITE")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Test 1: Load data
    X, y = test_load_data()
    results['load_data'] = X is not None
    
    # Test 2: Missing data handling
    X_processed = test_missing_data_handling(X)
    results['missing_data'] = X_processed is not None
    
    # Test 3: Training data creation
    training_data = test_get_training_data(X_processed if X_processed is not None else X)
    results['training_data'] = training_data is not None
    
    # Test 4: Feature creation
    results['features'] = test_feature_creation(training_data)
    
    # Test 5: Gradient descent
    results['gradient_descent'] = test_gradient_descent()
    
    # Test 6: Prediction
    predictions = test_predict_function(training_data)
    results['prediction'] = predictions is not None
    
    # Test 7: Evaluation
    results['evaluation'] = test_evaluator()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Your code is ready to use.")
    else:
        print("\n⚠ Some tests failed. Check the errors above.")


if __name__ == '__main__':
    main()
