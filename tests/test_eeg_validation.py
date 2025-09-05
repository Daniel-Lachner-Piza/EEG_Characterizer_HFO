import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the modules to test
from run_eeg_hfo_characterize_detect import (
    _calculate_analysis_windows, _validate_eeg_data, _validate_analysis_windows,
    _log_eeg_info, EEGValidationError, AnalysisWindowError,
    DEFAULT_SAMPLING_RATE_THRESHOLD
)


class TestAnalysisWindows(unittest.TestCase):
    """Test analysis window calculation and validation functions."""

    def test_calculate_analysis_windows_normal(self):
        """Test normal analysis window calculation."""
        analysis_start = 0
        analysis_end = 1000
        window_length = 100
        window_step = 50
        
        result = _calculate_analysis_windows(
            analysis_start, analysis_end, window_length, window_step
        )
        
        # Check structure
        self.assertIn('start', result)
        self.assertIn('end', result)
        
        # Check values
        expected_starts = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 
                                   450, 500, 550, 600, 650, 700, 750, 800, 850, 900])
        expected_ends = expected_starts + window_length
        
        np.testing.assert_array_equal(result['start'], expected_starts)
        np.testing.assert_array_equal(result['end'], expected_ends)

    def test_calculate_analysis_windows_edge_case(self):
        """Test analysis window calculation edge case."""
        analysis_start = 0
        analysis_end = 100
        window_length = 100
        window_step = 25
        
        result = _calculate_analysis_windows(
            analysis_start, analysis_end, window_length, window_step
        )
        
        # Should only have one window starting at 0
        expected_starts = np.array([0])
        expected_ends = np.array([100])
        
        np.testing.assert_array_equal(result['start'], expected_starts)
        np.testing.assert_array_equal(result['end'], expected_ends)

    def test_calculate_analysis_windows_no_windows_fit(self):
        """Test when no windows can fit."""
        analysis_start = 0
        analysis_end = 50
        window_length = 100
        window_step = 25
        
        result = _calculate_analysis_windows(
            analysis_start, analysis_end, window_length, window_step
        )
        
        # Should have empty arrays
        self.assertEqual(len(result['start']), 0)
        self.assertEqual(len(result['end']), 0)

    def test_calculate_analysis_windows_different_step_size(self):
        """Test with different step sizes."""
        analysis_start = 100
        analysis_end = 500
        window_length = 50
        window_step = 100
        
        result = _calculate_analysis_windows(
            analysis_start, analysis_end, window_length, window_step
        )
        
        expected_starts = np.array([100, 200, 300, 400])
        expected_ends = expected_starts + window_length
        
        np.testing.assert_array_equal(result['start'], expected_starts)
        np.testing.assert_array_equal(result['end'], expected_ends)


class TestEEGValidation(unittest.TestCase):
    """Test EEG data validation functions."""

    def setUp(self):
        self.mock_eeg_reader = Mock()
        self.mock_eeg_reader.fs = 2000
        self.mock_eeg_reader.n_samples = 10000
        self.mock_eeg_reader.ch_names = ['F3', 'F4', 'C3', 'C4']

    def test_validate_eeg_data_valid(self):
        """Test EEG validation with valid data."""
        # Should not raise any exception
        _validate_eeg_data(self.mock_eeg_reader)

    def test_validate_eeg_data_low_sampling_rate(self):
        """Test EEG validation with low sampling rate."""
        self.mock_eeg_reader.fs = 500
        
        with self.assertRaises(EEGValidationError) as context:
            _validate_eeg_data(self.mock_eeg_reader)
        
        self.assertIn("Sampling Rate is 500 Hz", str(context.exception))
        self.assertIn(f"under {DEFAULT_SAMPLING_RATE_THRESHOLD} Hz", str(context.exception))

    def test_validate_eeg_data_custom_threshold(self):
        """Test EEG validation with custom threshold."""
        self.mock_eeg_reader.fs = 1500
        
        with self.assertRaises(EEGValidationError):
            _validate_eeg_data(self.mock_eeg_reader, fs_threshold=2000)

    def test_validate_eeg_data_zero_samples(self):
        """Test EEG validation with zero samples."""
        self.mock_eeg_reader.n_samples = 0
        
        with self.assertRaises(EEGValidationError) as context:
            _validate_eeg_data(self.mock_eeg_reader)
        
        self.assertIn("EEG data has no samples", str(context.exception))

    def test_validate_eeg_data_negative_samples(self):
        """Test EEG validation with negative samples."""
        self.mock_eeg_reader.n_samples = -1
        
        with self.assertRaises(EEGValidationError):
            _validate_eeg_data(self.mock_eeg_reader)

    def test_validate_eeg_data_no_channels(self):
        """Test EEG validation with no channels."""
        self.mock_eeg_reader.ch_names = []
        
        with self.assertRaises(EEGValidationError) as context:
            _validate_eeg_data(self.mock_eeg_reader)
        
        self.assertIn("EEG data has no channels", str(context.exception))


class TestAnalysisWindowValidation(unittest.TestCase):
    """Test analysis window validation."""

    def test_validate_analysis_windows_valid(self):
        """Test analysis window validation with valid windows."""
        windows_dict = {
            'start': np.array([0, 100, 200]),
            'end': np.array([100, 200, 300])
        }
        max_sample = 1000
        
        # Should not raise any exception
        _validate_analysis_windows(windows_dict, max_sample)

    def test_validate_analysis_windows_negative_start(self):
        """Test analysis window validation with negative start."""
        windows_dict = {
            'start': np.array([-10, 100, 200]),
            'end': np.array([90, 200, 300])
        }
        max_sample = 1000
        
        with self.assertRaises(AnalysisWindowError) as context:
            _validate_analysis_windows(windows_dict, max_sample)
        
        self.assertIn("Incorrectly defined analysis window start samples", str(context.exception))

    def test_validate_analysis_windows_end_exceeds_max(self):
        """Test analysis window validation with end exceeding max sample."""
        windows_dict = {
            'start': np.array([0, 100, 200]),
            'end': np.array([100, 200, 1100])  # Last end > max_sample
        }
        max_sample = 1000
        
        with self.assertRaises(AnalysisWindowError) as context:
            _validate_analysis_windows(windows_dict, max_sample)
        
        self.assertIn("Incorrectly defined analysis window end samples", str(context.exception))

    def test_validate_analysis_windows_edge_case_equal_max(self):
        """Test analysis window validation with end equal to max sample."""
        windows_dict = {
            'start': np.array([0, 100]),
            'end': np.array([100, 1000])  # Last end == max_sample
        }
        max_sample = 1000
        
        # Should not raise exception (end can equal max_sample)
        _validate_analysis_windows(windows_dict, max_sample)

    def test_validate_analysis_windows_empty_arrays(self):
        """Test analysis window validation with empty arrays."""
        windows_dict = {
            'start': np.array([]),
            'end': np.array([])
        }
        max_sample = 1000
        
        # Should not raise exception
        _validate_analysis_windows(windows_dict, max_sample)


class TestLogEEGInfo(unittest.TestCase):
    """Test EEG information logging."""

    def setUp(self):
        self.mock_logger = Mock()
        self.mock_eeg_reader = Mock()
        self.mock_eeg_reader.fs = 2000
        self.mock_eeg_reader.n_samples = 20000
        self.mock_eeg_reader.ch_names = ['F3', 'F4', 'C3', 'C4']
        
        self.mock_config = Mock()
        self.mock_config.start_sec = 0.0
        self.mock_config.end_sec = 10.0

    def test_log_eeg_info(self):
        """Test EEG information logging."""
        pat_name = "test_patient"
        analysis_start_sample = 0
        analysis_end_sample = 20000
        
        _log_eeg_info(
            self.mock_logger, pat_name, self.mock_eeg_reader,
            self.mock_config, analysis_start_sample, analysis_end_sample
        )
        
        # Check that info was logged
        self.mock_logger.info.assert_called_once()
        logged_message = self.mock_logger.info.call_args[0][0]
        
        # Check content
        self.assertIn("Patient: test_patient", logged_message)
        self.assertIn("EEG Duration: 10.00 seconds", logged_message)
        self.assertIn("EEG Sampling Rate: 2000 Hz", logged_message)
        self.assertIn("EEG Nr. Samples: 20000", logged_message)
        self.assertIn("Analysis start second: 0.0", logged_message)
        self.assertIn("Analysis end second: 10.0", logged_message)
        self.assertIn("Analysis start sample: 0", logged_message)
        self.assertIn("Analysis end sample: 20000", logged_message)
        self.assertIn("Nr. Channels: 4", logged_message)
        self.assertIn("Channels: ['F3', 'F4', 'C3', 'C4']", logged_message)

    def test_log_eeg_info_different_values(self):
        """Test EEG information logging with different values."""
        self.mock_eeg_reader.fs = 1000
        self.mock_eeg_reader.n_samples = 5000
        self.mock_config.start_sec = 1.0
        self.mock_config.end_sec = 5.0
        
        _log_eeg_info(
            self.mock_logger, "patient2", self.mock_eeg_reader,
            self.mock_config, 1000, 5000
        )
        
        logged_message = self.mock_logger.info.call_args[0][0]
        self.assertIn("EEG Duration: 5.00 seconds", logged_message)
        self.assertIn("EEG Sampling Rate: 1000 Hz", logged_message)


if __name__ == '__main__':
    unittest.main()
