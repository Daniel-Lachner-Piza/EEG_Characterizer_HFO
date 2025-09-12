#!/usr/bin/env python3
"""
Test script to verify the improved scale_array function handles edge cases properly.
"""

import numpy as np
import sys
import os
import logging

# Add the project directory to the path
sys.path.insert(0, '/home/dlp/Documents/Development/EEG_Characterizer_HFO')

from hfo_spectral_detector.spectral_analyzer.characterize_events import scale_array

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_scale_array():
    """Test the scale_array function with various edge cases."""
    
    print("Testing scale_array function...")
    print("=" * 50)
    
    # Test 1: Normal array
    print("\n1. Normal array:")
    arr1 = np.array([1, 2, 3, 4, 5])
    result1 = scale_array(arr1)
    print(f"Input: {arr1}")
    print(f"Output: {result1}")
    print(f"Min: {np.min(result1)}, Max: {np.max(result1)}")
    assert np.isclose(np.min(result1), 0.0) and np.isclose(np.max(result1), 1.0)
    
    # Test 2: Array with negative values
    print("\n2. Array with negative values:")
    arr2 = np.array([-5, -2, 0, 3, 7])
    result2 = scale_array(arr2)
    print(f"Input: {arr2}")
    print(f"Output: {result2}")
    print(f"Min: {np.min(result2)}, Max: {np.max(result2)}")
    assert np.isclose(np.min(result2), 0.0) and np.isclose(np.max(result2), 1.0)
    
    # Test 3: Constant array (all values same)
    print("\n3. Constant array:")
    arr3 = np.array([5, 5, 5, 5, 5])
    result3 = scale_array(arr3)
    print(f"Input: {arr3}")
    print(f"Output: {result3}")
    print(f"All values should be 0.5: {np.all(np.isclose(result3, 0.5))}")
    assert np.all(np.isclose(result3, 0.5))
    
    # Test 4: Single element array
    print("\n4. Single element array:")
    arr4 = np.array([42])
    result4 = scale_array(arr4)
    print(f"Input: {arr4}")
    print(f"Output: {result4}")
    print(f"Should be 0.5: {np.isclose(result4[0], 0.5)}")
    assert np.isclose(result4[0], 0.5)
    
    # Test 5: Empty array
    print("\n5. Empty array:")
    arr5 = np.array([])
    result5 = scale_array(arr5)
    print(f"Input: {arr5}")
    print(f"Output: {result5}")
    print(f"Should be empty: {len(result5) == 0}")
    assert len(result5) == 0
    
    # Test 6: Array with NaN values
    print("\n6. Array with NaN values:")
    arr6 = np.array([1, 2, np.nan, 4, 5])
    result6 = scale_array(arr6)
    print(f"Input: {arr6}")
    print(f"Output: {result6}")
    print(f"NaN preserved: {np.isnan(result6[2])}")
    print(f"Non-NaN values scaled: min={np.nanmin(result6)}, max={np.nanmax(result6)}")
    assert np.isnan(result6[2])
    assert np.isclose(np.nanmin(result6), 0.0) and np.isclose(np.nanmax(result6), 1.0)
    
    # Test 7: Array with all NaN values
    print("\n7. Array with all NaN values:")
    arr7 = np.array([np.nan, np.nan, np.nan])
    try:
        result7 = scale_array(arr7)
        print("This should have raised an error!")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
    
    # Test 8: Array with infinite values
    print("\n8. Array with infinite values:")
    arr8 = np.array([1, 2, np.inf, 4, -np.inf])
    result8 = scale_array(arr8)
    print(f"Input: {arr8}")
    print(f"Output: {result8}")
    finite_result = result8[np.isfinite(result8)]
    print(f"Finite values - Min: {np.min(finite_result)}, Max: {np.max(finite_result)}")
    print(f"Infinite values converted to NaN: {np.isnan(result8[2]) and np.isnan(result8[4])}")
    assert np.isclose(np.min(finite_result), 0.0) and np.isclose(np.max(finite_result), 1.0)
    assert np.isnan(result8[2]) and np.isnan(result8[4])  # inf values should become NaN
    
    # Test 9: Very small differences (numerical precision)
    print("\n9. Array with very small differences:")
    arr9 = np.array([1.0000000001, 1.0000000002, 1.0000000003])
    result9 = scale_array(arr9)
    print(f"Input: {arr9}")
    print(f"Output: {result9}")
    print(f"Min: {np.min(result9)}, Max: {np.max(result9)}")
    assert np.isclose(np.min(result9), 0.0) and np.isclose(np.max(result9), 1.0)
    
    # Test 10: List input (not numpy array)
    print("\n10. List input:")
    arr10 = [1, 2, 3, 4, 5]
    result10 = scale_array(arr10)
    print(f"Input: {arr10}")
    print(f"Output: {result10}")
    print(f"Type: {type(result10)}")
    print(f"Min: {np.min(result10)}, Max: {np.max(result10)}")
    assert isinstance(result10, np.ndarray)
    assert np.isclose(np.min(result10), 0.0) and np.isclose(np.max(result10), 1.0)
    
    # Test 11: Array with only infinite values
    print("\n11. Array with only infinite values:")
    arr11 = np.array([np.inf, -np.inf, np.inf])
    result11 = scale_array(arr11)
    print(f"Input: {arr11}")
    print(f"Output: {result11}")
    print(f"All values should be NaN: {np.all(np.isnan(result11))}")
    assert np.all(np.isnan(result11))
    
    # Test 12: Very large values
    print("\n12. Very large values:")
    arr12 = np.array([1e10, 2e10, 3e10])
    result12 = scale_array(arr12)
    print(f"Input: {arr12}")
    print(f"Output: {result12}")
    print(f"Min: {np.min(result12)}, Max: {np.max(result12)}")
    assert np.isclose(np.min(result12), 0.0) and np.isclose(np.max(result12), 1.0)
    
    print("\n" + "=" * 50)
    print("All tests passed! ✅")
    print("The improved scale_array function properly handles all edge cases.")

if __name__ == "__main__":
    test_scale_array()
