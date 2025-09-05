import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the modules to test
from run_eeg_hfo_characterize_detect import main


class TestMainFunction(unittest.TestCase):
    """Test the main function and integration scenarios."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('run_eeg_hfo_characterize_detect.init_logging')
    @patch('run_eeg_hfo_characterize_detect.create_test_args')
    @patch('run_eeg_hfo_characterize_detect.Characterization_Config')
    @patch('run_eeg_hfo_characterize_detect.InputDataExtractor')
    @patch('run_eeg_hfo_characterize_detect.run_eeg_characterization')
    @patch('run_eeg_hfo_characterize_detect.timing_context')
    @patch('socket.gethostname')
    @patch('os.cpu_count')
    @patch('sys.gettrace')
    @patch('builtins.print')
    def test_main_test_mode(self, mock_print, mock_gettrace, mock_cpu_count, 
                           mock_hostname, mock_timing, mock_run_eeg, mock_extractor_class,
                           mock_config_class, mock_create_test_args, mock_init_logging):
        """Test main function in test mode."""
        # Setup mocks
        mock_logger = Mock()
        mock_init_logging.return_value = mock_logger
        mock_gettrace.return_value = True  # Simulate debugger
        mock_cpu_count.return_value = 8
        mock_hostname.return_value = "test_host"
        
        mock_test_args = Mock()
        mock_create_test_args.return_value = mock_test_args
        
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        mock_extractor = Mock()
        mock_files = [Path("test1.edf"), Path("test2.edf")]
        mock_extractor.get_files_to_process.return_value = mock_files
        mock_extractor_class.return_value = mock_extractor
        
        # Mock timing_context
        mock_timing.return_value.__enter__ = Mock()
        mock_timing.return_value.__exit__ = Mock(return_value=False)
        
        # Run main
        main()
        
        # Verify test mode detection and logging
        mock_logger.info.assert_any_call("Running in test mode")
        
        # Verify test args were used
        mock_create_test_args.assert_called_once()
        mock_config_class.assert_called_once_with(mock_test_args)
        
        # Verify system info logging
        mock_logger.info.assert_any_call("Host: test_host")
        mock_logger.info.assert_any_call("Number of CPUs: 8")
        mock_logger.info.assert_any_call("Test mode: True")
        
        # Verify configuration and files display/logging
        mock_config.display_config.assert_called_once()
        mock_config.log_config.assert_called_once_with(mock_logger)
        mock_extractor.display_files_to_process.assert_called_once()
        mock_extractor.log_files_to_process.assert_called_once_with(mock_logger)
        
        # Verify main processing was called
        mock_run_eeg.assert_called_once_with(mock_config, mock_files, mock_logger)
        
        # Verify success logging
        mock_logger.info.assert_any_call("All processing completed successfully")

    @patch('run_eeg_hfo_characterize_detect.init_logging')
    @patch('run_eeg_hfo_characterize_detect.create_argument_parser')
    @patch('run_eeg_hfo_characterize_detect.Characterization_Config')
    @patch('run_eeg_hfo_characterize_detect.InputDataExtractor')
    @patch('run_eeg_hfo_characterize_detect.run_eeg_characterization')
    @patch('run_eeg_hfo_characterize_detect.timing_context')
    @patch('socket.gethostname')
    @patch('os.cpu_count')
    @patch('sys.gettrace')
    @patch('sys.argv', ['script.py', '--dataset_name', 'test', '--input_folder', '/test', 
                        '--output_folder', '/out', '--montage_type', 'sb'])
    def test_main_normal_mode(self, mock_gettrace, mock_cpu_count, mock_hostname,
                             mock_timing, mock_run_eeg, mock_extractor_class,
                             mock_config_class, mock_parser_func, mock_init_logging):
        """Test main function in normal mode with command line arguments."""
        # Setup mocks
        mock_logger = Mock()
        mock_init_logging.return_value = mock_logger
        mock_gettrace.return_value = None  # No debugger
        mock_cpu_count.return_value = 4
        mock_hostname.return_value = "prod_host"
        
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_parser_func.return_value = mock_parser
        
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        mock_extractor = Mock()
        mock_files = [Path("prod1.edf")]
        mock_extractor.get_files_to_process.return_value = mock_files
        mock_extractor_class.return_value = mock_extractor
        
        # Mock timing_context
        mock_timing.return_value.__enter__ = Mock()
        mock_timing.return_value.__exit__ = Mock(return_value=False)
        
        # Run main
        main()
        
        # Verify normal mode detection and logging
        mock_logger.info.assert_any_call("Running in normal mode")
        
        # Verify argument parsing was used
        mock_parser_func.assert_called_once()
        mock_parser.parse_args.assert_called_once()
        mock_config_class.assert_called_once_with(mock_args)
        
        # Verify system info with different values
        mock_logger.info.assert_any_call("Host: prod_host")
        mock_logger.info.assert_any_call("Number of CPUs: 4")
        mock_logger.info.assert_any_call("Test mode: False")
        
        # Verify processing was called
        mock_run_eeg.assert_called_once_with(mock_config, mock_files, mock_logger)

    @patch('run_eeg_hfo_characterize_detect.init_logging')
    @patch('run_eeg_hfo_characterize_detect.create_test_args')
    @patch('run_eeg_hfo_characterize_detect.Characterization_Config')
    @patch('sys.gettrace')
    def test_main_configuration_error(self, mock_gettrace, mock_config_class, 
                                     mock_create_test_args, mock_init_logging):
        """Test main function with configuration error."""
        # Setup mocks
        mock_logger = Mock()
        mock_init_logging.return_value = mock_logger
        mock_gettrace.return_value = True  # Test mode
        
        mock_test_args = Mock()
        mock_create_test_args.return_value = mock_test_args
        
        # Make config creation raise an error
        mock_config_class.side_effect = ValueError("Invalid configuration")
        
        # Should raise the error and log it
        with self.assertRaises(ValueError):
            main()
        
        mock_logger.error.assert_called()
        error_call = mock_logger.error.call_args[0][0]
        self.assertIn("Fatal error in main execution", error_call)

    @patch('run_eeg_hfo_characterize_detect.init_logging')
    @patch('run_eeg_hfo_characterize_detect.create_test_args')
    @patch('run_eeg_hfo_characterize_detect.Characterization_Config')
    @patch('run_eeg_hfo_characterize_detect.InputDataExtractor')
    @patch('run_eeg_hfo_characterize_detect.run_eeg_characterization')
    @patch('run_eeg_hfo_characterize_detect.timing_context')
    @patch('socket.gethostname')
    @patch('os.cpu_count')
    @patch('sys.gettrace')
    def test_main_processing_error(self, mock_gettrace, mock_cpu_count, mock_hostname,
                                  mock_timing, mock_run_eeg, mock_extractor_class,
                                  mock_config_class, mock_create_test_args, mock_init_logging):
        """Test main function with processing error."""
        # Setup mocks
        mock_logger = Mock()
        mock_init_logging.return_value = mock_logger
        mock_gettrace.return_value = True
        mock_cpu_count.return_value = 8
        mock_hostname.return_value = "test_host"
        
        mock_test_args = Mock()
        mock_create_test_args.return_value = mock_test_args
        
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        mock_extractor = Mock()
        mock_files = [Path("test.edf")]
        mock_extractor.get_files_to_process.return_value = mock_files
        mock_extractor_class.return_value = mock_extractor
        
        # Mock timing_context
        mock_timing.return_value.__enter__ = Mock()
        mock_timing.return_value.__exit__ = Mock(return_value=False)
        
        # Make processing raise an error
        mock_run_eeg.side_effect = RuntimeError("Processing failed")
        
        # Should raise the error and log it
        with self.assertRaises(RuntimeError):
            main()
        
        mock_logger.error.assert_called()
        error_call = mock_logger.error.call_args[0][0]
        self.assertIn("Fatal error in main execution", error_call)

    @patch('run_eeg_hfo_characterize_detect.init_logging')
    @patch('sys.argv', ['script.py'])  # No arguments - should trigger test mode
    @patch('sys.gettrace')
    def test_main_test_mode_no_args(self, mock_gettrace, mock_init_logging):
        """Test that main enters test mode when no arguments are provided."""
        mock_logger = Mock()
        mock_init_logging.return_value = mock_logger
        mock_gettrace.return_value = None  # No debugger
        
        # Mock all the other dependencies to avoid full execution
        with patch('run_eeg_hfo_characterize_detect.create_test_args') as mock_create_test:
            mock_create_test.side_effect = Exception("Test mode confirmed")
            
            with self.assertRaises(Exception) as context:
                main()
            
            # Should use test args, not argument parser
            mock_create_test.assert_called_once()
            self.assertEqual(str(context.exception), "Test mode confirmed")

    @patch('run_eeg_hfo_characterize_detect.init_logging')
    @patch('run_eeg_hfo_characterize_detect.create_test_args')
    @patch('run_eeg_hfo_characterize_detect.Characterization_Config')
    @patch('run_eeg_hfo_characterize_detect.InputDataExtractor')
    @patch('run_eeg_hfo_characterize_detect.run_eeg_characterization')
    @patch('run_eeg_hfo_characterize_detect.timing_context')
    @patch('socket.gethostname')
    @patch('os.cpu_count')
    @patch('sys.gettrace')
    @patch('builtins.print')
    def test_main_timing_contexts(self, mock_print, mock_gettrace, mock_cpu_count,
                                 mock_hostname, mock_timing, mock_run_eeg, 
                                 mock_extractor_class, mock_config_class,
                                 mock_create_test_args, mock_init_logging):
        """Test that main function uses timing contexts correctly."""
        # Setup mocks
        mock_logger = Mock()
        mock_init_logging.return_value = mock_logger
        mock_gettrace.return_value = True
        mock_cpu_count.return_value = 8
        mock_hostname.return_value = "test_host"
        
        mock_test_args = Mock()
        mock_create_test_args.return_value = mock_test_args
        
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        mock_extractor = Mock()
        mock_files = [Path("test.edf")]
        mock_extractor.get_files_to_process.return_value = mock_files
        mock_extractor_class.return_value = mock_extractor
        
        # Mock timing_context to track calls
        timing_contexts = []
        def mock_timing_factory(operation_name, logger):
            context_mock = Mock()
            context_mock.__enter__ = Mock()
            context_mock.__exit__ = Mock(return_value=False)
            timing_contexts.append(operation_name)
            return context_mock
        
        mock_timing.side_effect = mock_timing_factory
        
        # Run main
        main()
        
        # Verify timing contexts were used
        expected_contexts = ["Configuration setup", "Complete EEG characterization"]
        for expected in expected_contexts:
            self.assertIn(expected, timing_contexts)


if __name__ == '__main__':
    unittest.main()
