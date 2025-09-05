import os
import sys
import unittest
import tempfile
import shutil
import time
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

# Import the modules to test
from run_eeg_hfo_characterize_detect import (
    timing_context, init_logging, Characterization_Config, 
    InputDataExtractor, _calculate_analysis_windows,
    _validate_eeg_data, _validate_analysis_windows,
    _log_eeg_info, run_eeg_characterization, process_single_eeg_file,
    create_argument_parser, create_test_args,
    EEGValidationError, AnalysisWindowError, EEGProcessingError,
    DEFAULT_SAMPLING_RATE_THRESHOLD, DEFAULT_WINDOW_LENGTH_SECONDS
)


class TestTimingContext(unittest.TestCase):
    """Test the timing_context context manager."""

    def setUp(self):
        self.logger = Mock()
        
    def test_timing_context_normal_operation(self):
        """Test timing context with normal operation."""
        with timing_context("test_operation", self.logger):
            time.sleep(0.01)  # Small sleep to ensure measurable time
        
        # Check that info was logged
        self.logger.info.assert_called()
        call_args = self.logger.info.call_args[0][0]
        self.assertIn("test_operation completed in", call_args)
        self.assertIn("seconds", call_args)
    
    def test_timing_context_with_exception(self):
        """Test timing context when exception occurs."""
        with self.assertRaises(ValueError):
            with timing_context("test_operation", self.logger):
                raise ValueError("Test exception")
        
        # Logger should still be called
        self.logger.info.assert_called()
        call_args = self.logger.info.call_args[0][0]
        self.assertIn("test_operation completed in", call_args)
    
    @patch('builtins.print')
    def test_timing_context_prints_to_stdout(self, mock_print):
        """Test that timing context also prints to stdout."""
        with timing_context("test_operation", self.logger):
            pass
        
        # Should print to stdout as well as log
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        self.assertIn("test_operation completed in", call_args)


class TestInitLogging(unittest.TestCase):
    """Test the init_logging function."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.original_cwd = os.getcwd()
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
        # Reset root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)
    
    @patch('run_eeg_hfo_characterize_detect.Path')
    def test_init_logging_creates_logger(self, mock_path_class):
        """Test that init_logging creates logger with proper configuration."""
        mock_path_class.return_value.parent = self.temp_dir
        
        logger = init_logging()
        
        # Check logger is created
        self.assertIsInstance(logger, logging.Logger)
        
        # Check root logger has handlers
        root_logger = logging.getLogger()
        self.assertTrue(len(root_logger.handlers) >= 2)  # File and console handlers
    
    @patch('os.makedirs')
    def test_init_logging_creates_log_directory(self, mock_makedirs):
        """Test that init_logging creates logs directory."""
        # Call init_logging which should call os.makedirs to create the logs directory
        try:
            init_logging()
        except FileNotFoundError:
            # This is expected since we're not mocking the full file system
            pass
        
        # Verify that makedirs was called
        mock_makedirs.assert_called()
    
    def test_init_logging_avoids_duplicate_handlers(self):
        """Test that calling init_logging multiple times doesn't add duplicate handlers."""
        # First call
        logger1 = init_logging()
        handler_count_1 = len(logging.getLogger().handlers)
        
        # Second call
        logger2 = init_logging()
        handler_count_2 = len(logging.getLogger().handlers)
        
        # Should not add more handlers
        self.assertEqual(handler_count_1, handler_count_2)


class TestCharacterizationConfig(unittest.TestCase):
    """Test the Characterization_Config class."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.valid_args = Mock()
        self.valid_args.dataset_name = "test_dataset"
        self.valid_args.rm_vchann = "yes"
        self.valid_args.input_folder = str(self.temp_dir)
        self.valid_args.output_folder = str(self.temp_dir / "output")
        self.valid_args.eeg_format = "edf"
        self.valid_args.montage_type = "sb"
        self.valid_args.montage_channels = "F3-C3,C3-P3"
        self.valid_args.power_line_freq = 60
        self.valid_args.start_sec = 0.0
        self.valid_args.end_sec = 30.0
        self.valid_args.wdw_step_s = 0.1
        self.valid_args.force_characterization = "no"
        self.valid_args.force_hfo_detection = "yes"
        self.valid_args.n_jobs = 4
        self.valid_args.verbose = "yes"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_config_initialization_valid_args(self):
        """Test config initialization with valid arguments."""
        config = Characterization_Config(self.valid_args)
        
        self.assertEqual(config.dataset_name, "test_dataset")
        self.assertEqual(config.rm_vchann, "yes")
        self.assertEqual(config.input_folder, Path(self.temp_dir))
        self.assertEqual(config.eeg_format, "edf")
        self.assertEqual(config.montage_type, "sb")
        self.assertEqual(config.power_line_freqs, 60)
        self.assertEqual(config.start_sec, 0.0)
        self.assertEqual(config.end_sec, 30.0)
        self.assertEqual(config.wdw_step_s, 0.1)
        self.assertEqual(config.n_jobs, 4)

    def test_config_validation_invalid_rm_vchann(self):
        """Test config validation with invalid rm_vchann."""
        self.valid_args.rm_vchann = "maybe"
        with self.assertRaises(ValueError) as context:
            Characterization_Config(self.valid_args)
        self.assertIn("rm_vchann must be 'yes' or 'no'", str(context.exception))

    def test_config_validation_nonexistent_input_folder(self):
        """Test config validation with nonexistent input folder."""
        self.valid_args.input_folder = "/nonexistent/path"
        with self.assertRaises(ValueError) as context:
            Characterization_Config(self.valid_args)
        self.assertIn("Input folder does not exist", str(context.exception))

    def test_config_validation_invalid_eeg_format(self):
        """Test config validation with invalid EEG format."""
        self.valid_args.eeg_format = "invalid"
        with self.assertRaises(ValueError) as context:
            Characterization_Config(self.valid_args)
        self.assertIn("Unsupported EEG format", str(context.exception))

    def test_config_validation_invalid_montage_type(self):
        """Test config validation with invalid montage type."""
        self.valid_args.montage_type = "invalid"
        with self.assertRaises(ValueError) as context:
            Characterization_Config(self.valid_args)
        self.assertIn("Invalid montage type", str(context.exception))

    def test_config_validation_invalid_power_line_freq(self):
        """Test config validation with invalid power line frequency."""
        self.valid_args.power_line_freq = 45
        with self.assertRaises(ValueError) as context:
            Characterization_Config(self.valid_args)
        self.assertIn("Invalid power line frequency", str(context.exception))

    def test_config_validation_negative_start_time(self):
        """Test config validation with negative start time."""
        self.valid_args.start_sec = -1.0
        with self.assertRaises(ValueError) as context:
            Characterization_Config(self.valid_args)
        self.assertIn("Start time cannot be negative", str(context.exception))

    def test_config_validation_invalid_end_time(self):
        """Test config validation with invalid end time."""
        self.valid_args.end_sec = 5.0
        self.valid_args.start_sec = 10.0
        with self.assertRaises(ValueError) as context:
            Characterization_Config(self.valid_args)
        self.assertIn("End time must be greater than start time", str(context.exception))

    def test_config_validation_zero_window_step(self):
        """Test config validation with zero window step."""
        self.valid_args.wdw_step_s = 0.0
        with self.assertRaises(ValueError) as context:
            Characterization_Config(self.valid_args)
        self.assertIn("Window step size must be positive", str(context.exception))

    def test_config_properties(self):
        """Test config boolean and list properties."""
        config = Characterization_Config(self.valid_args)
        
        self.assertTrue(config.rm_vchann_bool)
        self.assertFalse(config.force_characterization_bool)
        self.assertTrue(config.force_hfo_detection_bool)
        self.assertTrue(config.verbose_bool)
        
        expected_channels = ["f3-c3", "c3-p3"]  # Should be lowercase
        self.assertEqual(config.montage_channels_list, expected_channels)

    def test_config_properties_empty_channels(self):
        """Test config properties with empty montage channels."""
        self.valid_args.montage_channels = ""
        config = Characterization_Config(self.valid_args)
        self.assertEqual(config.montage_channels_list, [])

    @patch('builtins.print')
    def test_display_config(self, mock_print):
        """Test display_config method."""
        config = Characterization_Config(self.valid_args)
        config.display_config()
        
        mock_print.assert_called_once()
        output = mock_print.call_args[0][0]
        self.assertIn("Dataset = test_dataset", output)
        self.assertIn("EEG format = edf", output)

    def test_log_config(self):
        """Test log_config method."""
        mock_logger = Mock()
        config = Characterization_Config(self.valid_args)
        config.log_config(mock_logger)
        
        mock_logger.info.assert_called_once()
        logged_message = mock_logger.info.call_args[0][0]
        self.assertIn("Dataset = test_dataset", logged_message)
        self.assertIn("EEG format = edf", logged_message)

    def test_save_spect_img_property(self):
        """Test save_spect_img property logic."""
        # Test mode False, n_jobs != 1 should be False
        self.valid_args.n_jobs = 4
        config = Characterization_Config(self.valid_args, test_mode=False)
        self.assertFalse(config.save_spect_img)
        
        # Test mode True, n_jobs == 1, but DEFAULT_SAVE_SPECT_IMAGE is False, so should be False
        self.valid_args.n_jobs = 1
        config = Characterization_Config(self.valid_args, test_mode=True)
        # The formula is: (n_jobs==1 and DEFAULT_SAVE_SPECT_IMAGE) and test_mode
        # Since DEFAULT_SAVE_SPECT_IMAGE is False, this should be False
        self.assertFalse(config.save_spect_img)


class TestInputDataExtractor(unittest.TestCase):
    """Test the InputDataExtractor class."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Mock()
        self.config.input_folder = self.temp_dir
        self.config.eeg_format = "edf"

        # Create some test files
        (self.temp_dir / "file1.edf").touch()
        (self.temp_dir / "file2.edf").touch()
        (self.temp_dir / "subdir").mkdir()
        (self.temp_dir / "subdir" / "file3.edf").touch()
        (self.temp_dir / "file4.txt").touch()  # Wrong extension

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_files_to_process(self):
        """Test getting files to process."""
        extractor = InputDataExtractor(self.config)
        files = extractor.get_files_to_process()
        
        # Should find .edf files
        edf_files = [f for f in files if f.suffix == '.edf']
        self.assertEqual(len(edf_files), 3)  # file1.edf, file2.edf, file3.edf
        
        # Should be sorted
        file_names = [f.name for f in files]
        self.assertEqual(file_names, sorted(file_names))

    def test_get_files_caching(self):
        """Test that file listing is cached."""
        extractor = InputDataExtractor(self.config)
        
        # First call
        files1 = extractor.get_files_to_process()
        
        # Second call should return same object (cached)
        files2 = extractor.get_files_to_process()
        
        self.assertIs(files1, files2)

    @patch('builtins.print')
    def test_display_files_to_process(self, mock_print):
        """Test displaying files to process."""
        extractor = InputDataExtractor(self.config)
        extractor.display_files_to_process()
        
        # Should print the files found
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        found_message = any("Found 3 files to process" in call for call in print_calls)
        self.assertTrue(found_message)

    @patch('builtins.print')
    def test_display_files_no_files_found(self, mock_print):
        """Test displaying when no files found."""
        self.config.eeg_format = "xyz"  # Non-existent format
        extractor = InputDataExtractor(self.config)
        extractor.display_files_to_process()
        
        mock_print.assert_called_once()
        output = mock_print.call_args[0][0]
        self.assertIn("No xyz files found", output)

    def test_log_files_to_process(self):
        """Test logging files to process."""
        mock_logger = Mock()
        extractor = InputDataExtractor(self.config)
        extractor.log_files_to_process(mock_logger)
        
        # Should log info about files found
        mock_logger.info.assert_called()
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        found_message = any("Found 3 EEG files to process" in call for call in info_calls)
        self.assertTrue(found_message)

    def test_log_files_no_files_found(self):
        """Test logging when no files found."""
        mock_logger = Mock()
        self.config.eeg_format = "xyz"  # Non-existent format
        extractor = InputDataExtractor(self.config)
        extractor.log_files_to_process(mock_logger)
        
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("No xyz files found", warning_msg)


if __name__ == '__main__':
    unittest.main()
