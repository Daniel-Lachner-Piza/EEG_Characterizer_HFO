import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the modules to test
from run_eeg_hfo_characterize_detect import (
    run_eeg_characterization, process_single_eeg_file,
    EEGValidationError, AnalysisWindowError, EEGProcessingError,
    DEFAULT_WINDOW_LENGTH_SECONDS
)


class TestEEGProcessing(unittest.TestCase):
    """Test EEG processing functions."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_config = Mock()
        self.mock_config.output_folder = self.temp_dir / "output"
        self.mock_config.montage_type = "sb"
        self.mock_config.rm_vchann_bool = True
        self.mock_config.start_sec = 0.0
        self.mock_config.end_sec = -1
        self.mock_config.wdw_step_s = 0.1
        self.mock_config.montage_channels_list = []
        self.mock_config.power_line_freqs = 60
        self.mock_config.n_jobs = 1
        self.mock_config.force_characterization_bool = False
        self.mock_config.force_hfo_detection_bool = True
        self.mock_config.verbose_bool = True
        self.mock_config.save_spect_img = False
        
        self.mock_logger = Mock()
        
        # Create a test EEG file path
        self.test_eeg_file = self.temp_dir / "test.edf"
        self.test_eeg_file.touch()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('run_eeg_hfo_characterize_detect.HFO_Detector')
    @patch('run_eeg_hfo_characterize_detect.timing_context')
    @patch('os.makedirs')
    def test_run_eeg_characterization_empty_file_list(self, mock_makedirs, mock_timing, mock_detector_class):
        """Test run_eeg_characterization with empty file list."""
        files_to_process = []
        
        run_eeg_characterization(self.mock_config, files_to_process, self.mock_logger)
        
        # Should log warning and return early
        self.mock_logger.warning.assert_called_once_with("No files to process")
        
        # Should not create detector
        mock_detector_class.assert_not_called()

    @patch('run_eeg_hfo_characterize_detect.HFO_Detector')
    @patch('run_eeg_hfo_characterize_detect.process_single_eeg_file')
    @patch('run_eeg_hfo_characterize_detect.timing_context')
    @patch('os.makedirs')
    def test_run_eeg_characterization_success(self, mock_makedirs, mock_timing, mock_process, mock_detector_class):
        """Test successful run_eeg_characterization."""
        files_to_process = [self.test_eeg_file]
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        
        # Mock timing_context to act as a simple context manager
        mock_timing.return_value.__enter__ = Mock()
        mock_timing.return_value.__exit__ = Mock(return_value=False)
        
        run_eeg_characterization(self.mock_config, files_to_process, self.mock_logger)
        
        # Should create output directory
        mock_makedirs.assert_called_once()
        
        # Should create detector and load models
        mock_detector_class.assert_called_once()
        mock_detector.load_models.assert_called_once()
        
        # Should process the file
        mock_process.assert_called_once_with(
            self.mock_config, self.test_eeg_file, mock_detector, self.mock_logger
        )
        
        # Should log summary
        self.mock_logger.info.assert_called()
        log_calls = [call[0][0] for call in self.mock_logger.info.call_args_list]
        summary_logged = any("Processing completed: 1 successful, 0 errors" in call for call in log_calls)
        self.assertTrue(summary_logged)

    @patch('run_eeg_hfo_characterize_detect.HFO_Detector')
    @patch('run_eeg_hfo_characterize_detect.process_single_eeg_file')
    @patch('run_eeg_hfo_characterize_detect.timing_context')
    @patch('os.makedirs')
    def test_run_eeg_characterization_with_errors(self, mock_makedirs, mock_timing, mock_process, mock_detector_class):
        """Test run_eeg_characterization with processing errors."""
        files_to_process = [self.test_eeg_file]
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        
        # Mock timing_context
        mock_timing.return_value.__enter__ = Mock()
        mock_timing.return_value.__exit__ = Mock(return_value=False)
        
        # Make process_single_eeg_file raise an error
        mock_process.side_effect = EEGValidationError("Test validation error")
        
        run_eeg_characterization(self.mock_config, files_to_process, self.mock_logger)
        
        # Should log error
        self.mock_logger.error.assert_called()
        error_call = self.mock_logger.error.call_args[0][0]
        self.assertIn("Validation/Processing error", error_call)
        self.assertIn("Test validation error", error_call)
        
        # Should log summary with errors
        log_calls = [call[0][0] for call in self.mock_logger.info.call_args_list]
        summary_logged = any("Processing completed: 0 successful, 1 errors" in call for call in log_calls)
        self.assertTrue(summary_logged)

    @patch('run_eeg_hfo_characterize_detect.EEG_IO')
    @patch('run_eeg_hfo_characterize_detect._validate_eeg_data')
    @patch('run_eeg_hfo_characterize_detect._log_eeg_info')
    @patch('run_eeg_hfo_characterize_detect._calculate_analysis_windows')
    @patch('run_eeg_hfo_characterize_detect._validate_analysis_windows')
    @patch('run_eeg_hfo_characterize_detect.characterize_events')
    @patch('run_eeg_hfo_characterize_detect.timing_context')
    def test_process_single_eeg_file_success(self, mock_timing, mock_characterize, 
                                           mock_validate_windows, mock_calc_windows,
                                           mock_log_info, mock_validate_eeg, mock_eeg_io):
        """Test successful processing of single EEG file."""
        # Setup mocks
        mock_eeg_reader = Mock()
        mock_eeg_reader.fs = 2000
        mock_eeg_reader.n_samples = 20000
        mock_eeg_io.return_value = mock_eeg_reader
        
        mock_detector = Mock()
        mock_detector.set_fs = Mock()
        mock_detector.run_hfo_detection = Mock()
        
        mock_calc_windows.return_value = {'start': np.array([0]), 'end': np.array([2000])}
        mock_characterize.return_value = "/path/to/events.csv"
        
        # Mock timing_context
        mock_timing.return_value.__enter__ = Mock()
        mock_timing.return_value.__exit__ = Mock(return_value=False)
        
        # Run the function
        process_single_eeg_file(self.mock_config, self.test_eeg_file, mock_detector, self.mock_logger)
        
        # Verify EEG reader creation
        mock_eeg_io.assert_called_once_with(eeg_filepath=self.test_eeg_file, mtg_t="sb")
        
        # Verify virtual channels removal
        mock_eeg_reader.remove_natus_virtual_channels.assert_called_once()
        
        # Verify validation
        mock_validate_eeg.assert_called_once_with(mock_eeg_reader)
        
        # Verify characterization
        mock_characterize.assert_called_once()
        characterize_kwargs = mock_characterize.call_args[1]
        self.assertEqual(characterize_kwargs['pat_name'], 'test')
        self.assertEqual(characterize_kwargs['eeg_reader'], mock_eeg_reader)
        
        # Verify detection
        mock_detector.set_fs.assert_called_once_with(2000)
        mock_detector.run_hfo_detection.assert_called_once()

    @patch('run_eeg_hfo_characterize_detect.EEG_IO')
    def test_process_single_eeg_file_eeg_io_error(self, mock_eeg_io):
        """Test process_single_eeg_file with EEG_IO error."""
        mock_eeg_io.side_effect = Exception("Failed to read EEG")
        mock_detector = Mock()
        
        with self.assertRaises(EEGProcessingError) as context:
            process_single_eeg_file(self.mock_config, self.test_eeg_file, mock_detector, self.mock_logger)
        
        self.assertIn("Failed to read EEG file", str(context.exception))

    @patch('run_eeg_hfo_characterize_detect.EEG_IO')
    def test_process_single_eeg_file_invalid_sampling_rate(self, mock_eeg_io):
        """Test process_single_eeg_file with invalid sampling rate."""
        mock_eeg_reader = Mock()
        mock_eeg_reader.fs = 0
        mock_eeg_io.return_value = mock_eeg_reader
        mock_detector = Mock()
        
        with self.assertRaises(EEGProcessingError) as context:
            process_single_eeg_file(self.mock_config, self.test_eeg_file, mock_detector, self.mock_logger)
        
        self.assertIn("Invalid sampling rate: 0 Hz", str(context.exception))

    @patch('run_eeg_hfo_characterize_detect.EEG_IO')
    @patch('run_eeg_hfo_characterize_detect._validate_eeg_data')
    @patch('run_eeg_hfo_characterize_detect._log_eeg_info')
    @patch('run_eeg_hfo_characterize_detect._calculate_analysis_windows')
    @patch('run_eeg_hfo_characterize_detect._validate_analysis_windows')
    @patch('run_eeg_hfo_characterize_detect.characterize_events')
    @patch('run_eeg_hfo_characterize_detect.timing_context')
    def test_process_single_eeg_file_characterization_error(self, mock_timing, mock_characterize,
                                                          mock_validate_windows, mock_calc_windows,
                                                          mock_log_info, mock_validate_eeg, mock_eeg_io):
        """Test process_single_eeg_file with characterization error."""
        # Setup mocks
        mock_eeg_reader = Mock()
        mock_eeg_reader.fs = 2000
        mock_eeg_reader.n_samples = 20000
        mock_eeg_io.return_value = mock_eeg_reader
        
        mock_detector = Mock()
        mock_calc_windows.return_value = {'start': np.array([0]), 'end': np.array([2000])}
        
        # Make characterize_events raise an error
        mock_characterize.side_effect = Exception("Characterization failed")
        
        # Mock timing_context
        mock_timing.return_value.__enter__ = Mock()
        mock_timing.return_value.__exit__ = Mock(return_value=False)
        
        with self.assertRaises(EEGProcessingError) as context:
            process_single_eeg_file(self.mock_config, self.test_eeg_file, mock_detector, self.mock_logger)
        
        self.assertIn("Failed during characterization/detection", str(context.exception))

    @patch('run_eeg_hfo_characterize_detect.EEG_IO')
    @patch('run_eeg_hfo_characterize_detect._validate_eeg_data')
    @patch('run_eeg_hfo_characterize_detect._log_eeg_info')
    @patch('run_eeg_hfo_characterize_detect._calculate_analysis_windows')
    @patch('run_eeg_hfo_characterize_detect._validate_analysis_windows')
    def test_process_single_eeg_file_custom_end_time(self, mock_validate_windows, mock_calc_windows,
                                                   mock_log_info, mock_validate_eeg, mock_eeg_io):
        """Test process_single_eeg_file with custom end time."""
        # Setup with custom end time
        self.mock_config.end_sec = 10.0
        
        mock_eeg_reader = Mock()
        mock_eeg_reader.fs = 2000
        mock_eeg_reader.n_samples = 40000  # 20 seconds
        mock_eeg_io.return_value = mock_eeg_reader
        
        mock_detector = Mock()
        mock_calc_windows.return_value = {'start': np.array([0]), 'end': np.array([2000])}
        
        # Should not raise exception from just setting up analysis parameters
        try:
            process_single_eeg_file(self.mock_config, self.test_eeg_file, mock_detector, self.mock_logger)
        except EEGProcessingError:
            pass  # Expected due to missing mocks for characterization
        
        # Verify that calculation windows was called with correct end sample
        # end_sec * fs = 10.0 * 2000 = 20000
        expected_analysis_end = 20000
        mock_calc_windows.assert_called_once()
        call_args = mock_calc_windows.call_args[0]
        analysis_start_sample, analysis_end_sample, window_length_samples, window_step_samples = call_args
        self.assertEqual(analysis_end_sample, expected_analysis_end)

    @patch('run_eeg_hfo_characterize_detect.EEG_IO')
    @patch('run_eeg_hfo_characterize_detect._validate_eeg_data')
    def test_process_single_eeg_file_no_remove_virtual_channels(self, mock_validate_eeg, mock_eeg_io):
        """Test process_single_eeg_file without removing virtual channels."""
        self.mock_config.rm_vchann_bool = False
        
        mock_eeg_reader = Mock()
        mock_eeg_reader.fs = 2000
        mock_eeg_reader.n_samples = 20000
        mock_eeg_io.return_value = mock_eeg_reader
        
        mock_detector = Mock()
        
        # Should raise error due to missing mocks, but check that virtual channels removal wasn't called
        try:
            process_single_eeg_file(self.mock_config, self.test_eeg_file, mock_detector, self.mock_logger)
        except Exception:
            pass
        
        # Verify virtual channels removal was NOT called
        mock_eeg_reader.remove_natus_virtual_channels.assert_not_called()


if __name__ == '__main__':
    unittest.main()
