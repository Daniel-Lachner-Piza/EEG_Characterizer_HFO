#!/usr/bin/env python3
"""
Comprehensive test runner for EEG_Characterizer_HFO project.
This script runs all unit tests and provides a summary of the test results.

Usage:
    # From project root, use virtual environment:
    .venv/bin/python tests/run_tests.py
    
    # Or if virtual environment is activated:
    python tests/run_tests.py
"""

import os
import sys
import unittest
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import all test modules
from tests.test_run_eeg_hfo_characterize_detect import *
from tests.test_eeg_validation import *
from tests.test_eeg_processing import *
from tests.test_argument_parsing import *
from tests.test_main_function import *
from tests.test_integration import *


def run_all_tests():
    """Run all tests and return results."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print("=" * 70)
    print("Running EEG Characterizer HFO Detection Tests")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[0]}")
    print(f"Test directory: {start_dir}")
    print("=" * 70)
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print("=" * 70)
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
            print(f"  {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
            # Print the last line of the error for brevity
            error_lines = traceback.strip().split('\n')
            if error_lines:
                print(f"  {error_lines[-1]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    return success


def run_specific_test_module(module_name):
    """Run tests from a specific module."""
    loader = unittest.TestLoader()
    
    try:
        module = __import__(f'tests.{module_name}', fromlist=[''])
        suite = loader.loadTestsFromModule(module)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return len(result.failures) == 0 and len(result.errors) == 0
    except ImportError as e:
        print(f"Error importing test module '{module_name}': {e}")
        return False


def list_available_tests():
    """List all available test modules."""
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob('test_*.py'))
    
    print("Available test modules:")
    for test_file in sorted(test_files):
        module_name = test_file.stem
        print(f"  - {module_name}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--list':
            list_available_tests()
        elif sys.argv[1].startswith('test_'):
            # Run specific test module
            module_name = sys.argv[1]
            if not module_name.startswith('test_'):
                module_name = 'test_' + module_name
            success = run_specific_test_module(module_name)
            sys.exit(0 if success else 1)
        else:
            print("Usage:")
            print("  python run_tests.py           # Run all tests")
            print("  python run_tests.py --list    # List available test modules")
            print("  python run_tests.py test_*    # Run specific test module")
            sys.exit(1)
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
