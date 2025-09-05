import os
import sys
import unittest
import argparse
from unittest.mock import Mock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the modules to test
from run_eeg_hfo_characterize_detect import (
    create_argument_parser, create_test_args
)


class TestArgumentParsing(unittest.TestCase):
    """Test argument parsing functions."""

    def test_create_argument_parser(self):
        """Test argument parser creation."""
        parser = create_argument_parser()
        
        # Check that it returns an ArgumentParser
        self.assertIsInstance(parser, argparse.ArgumentParser)
        
        # Check description
        self.assertIn("Characterize EEG to detect HFO", parser.description)

    def test_create_argument_parser_required_args(self):
        """Test that required arguments are properly configured."""
        parser = create_argument_parser()
        
        # Test with missing required arguments
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_create_argument_parser_all_args(self):
        """Test parsing with all arguments provided."""
        parser = create_argument_parser()
        
        test_args = [
            '--dataset_name', 'test_dataset',
            '--input_folder', '/test/input',
            '--output_folder', '/test/output',
            '--montage_type', 'sb',
            '--eeg_format', 'edf',
            '--montage_channels', 'F3-C3,C3-P3',
            '--rm_vchann', 'yes',
            '--power_line_freq', '60',
            '--start_sec', '0.0',
            '--end_sec', '30.0',
            '--wdw_step_s', '0.1',
            '--force_characterization', 'no',
            '--force_hfo_detection', 'yes',
            '--n_jobs', '4',
            '--verbose', 'yes'
        ]
        
        args = parser.parse_args(test_args)
        
        # Check parsed values
        self.assertEqual(args.dataset_name, 'test_dataset')
        self.assertEqual(args.input_folder, '/test/input')
        self.assertEqual(args.output_folder, '/test/output')
        self.assertEqual(args.montage_type, 'sb')
        self.assertEqual(args.eeg_format, 'edf')
        self.assertEqual(args.montage_channels, 'F3-C3,C3-P3')
        self.assertEqual(args.rm_vchann, 'yes')
        self.assertEqual(args.power_line_freq, 60)
        self.assertEqual(args.start_sec, 0.0)
        self.assertEqual(args.end_sec, 30.0)
        self.assertEqual(args.wdw_step_s, 0.1)
        self.assertEqual(args.force_characterization, 'no')
        self.assertEqual(args.force_hfo_detection, 'yes')
        self.assertEqual(args.n_jobs, 4)
        self.assertEqual(args.verbose, 'yes')

    def test_create_argument_parser_default_values(self):
        """Test that default values are correctly set."""
        parser = create_argument_parser()
        
        # Parse with only required arguments
        test_args = [
            '--dataset_name', 'test_dataset',
            '--input_folder', '/test/input',
            '--output_folder', '/test/output',
            '--montage_type', 'sb'
        ]
        
        args = parser.parse_args(test_args)
        
        # Check default values
        self.assertEqual(args.eeg_format, 'edf')
        self.assertEqual(args.montage_channels, '')
        self.assertEqual(args.rm_vchann, 'yes')
        self.assertEqual(args.power_line_freq, 60)
        self.assertEqual(args.start_sec, 0)
        self.assertEqual(args.end_sec, -1)
        self.assertEqual(args.wdw_step_s, 0.1)
        self.assertEqual(args.force_characterization, 'no')
        self.assertEqual(args.force_hfo_detection, 'yes')
        self.assertEqual(args.n_jobs, -1)
        self.assertEqual(args.verbose, 'yes')

    def test_create_argument_parser_choices_validation(self):
        """Test that choices validation works."""
        parser = create_argument_parser()
        
        # Test invalid rm_vchann choice
        test_args = [
            '--dataset_name', 'test_dataset',
            '--input_folder', '/test/input',
            '--output_folder', '/test/output',
            '--montage_type', 'sb',
            '--rm_vchann', 'invalid'
        ]
        
        with self.assertRaises(SystemExit):
            parser.parse_args(test_args)

    def test_create_argument_parser_type_conversion(self):
        """Test that argument types are correctly converted."""
        parser = create_argument_parser()
        
        test_args = [
            '--dataset_name', 'test_dataset',
            '--input_folder', '/test/input',
            '--output_folder', '/test/output',
            '--montage_type', 'sb',
            '--power_line_freq', '50',
            '--start_sec', '1.5',
            '--end_sec', '100.0',
            '--wdw_step_s', '0.2',
            '--n_jobs', '8'
        ]
        
        args = parser.parse_args(test_args)
        
        # Check types
        self.assertIsInstance(args.power_line_freq, int)
        self.assertIsInstance(args.start_sec, float)
        self.assertIsInstance(args.end_sec, float)
        self.assertIsInstance(args.wdw_step_s, float)
        self.assertIsInstance(args.n_jobs, int)
        
        # Check values
        self.assertEqual(args.power_line_freq, 50)
        self.assertEqual(args.start_sec, 1.5)
        self.assertEqual(args.end_sec, 100.0)
        self.assertEqual(args.wdw_step_s, 0.2)
        self.assertEqual(args.n_jobs, 8)


class TestCreateTestArgs(unittest.TestCase):
    """Test create_test_args function."""

    def test_create_test_args(self):
        """Test that create_test_args returns properly configured test arguments."""
        args = create_test_args()
        
        # Check that it has all required attributes
        self.assertTrue(hasattr(args, 'dataset_name'))
        self.assertTrue(hasattr(args, 'rm_vchann'))
        self.assertTrue(hasattr(args, 'input_folder'))
        self.assertTrue(hasattr(args, 'output_folder'))
        self.assertTrue(hasattr(args, 'eeg_format'))
        self.assertTrue(hasattr(args, 'montage_type'))
        self.assertTrue(hasattr(args, 'montage_channels'))
        self.assertTrue(hasattr(args, 'power_line_freq'))
        self.assertTrue(hasattr(args, 'force_characterization'))
        self.assertTrue(hasattr(args, 'force_hfo_detection'))
        self.assertTrue(hasattr(args, 'start_sec'))
        self.assertTrue(hasattr(args, 'end_sec'))
        self.assertTrue(hasattr(args, 'wdw_step_s'))
        self.assertTrue(hasattr(args, 'n_jobs'))
        self.assertTrue(hasattr(args, 'verbose'))

    def test_create_test_args_values(self):
        """Test that create_test_args has expected values."""
        args = create_test_args()
        
        # Check specific values
        self.assertEqual(args.dataset_name, "PhysioTest_DLP")
        self.assertEqual(args.rm_vchann, "yes")
        self.assertEqual(args.eeg_format, "edf")
        self.assertEqual(args.montage_type, "sb")
        self.assertEqual(args.power_line_freq, 60)
        self.assertEqual(args.force_characterization, "yes")
        self.assertEqual(args.force_hfo_detection, "yes")
        self.assertEqual(args.start_sec, 0.0)
        self.assertEqual(args.end_sec, 20.0)
        self.assertEqual(args.wdw_step_s, 1.0)
        self.assertEqual(args.n_jobs, -1)
        self.assertEqual(args.verbose, "yes")

    def test_create_test_args_types(self):
        """Test that create_test_args has correct types."""
        args = create_test_args()
        
        # Check types
        self.assertIsInstance(args.dataset_name, str)
        self.assertIsInstance(args.rm_vchann, str)
        self.assertIsInstance(args.input_folder, str)
        self.assertIsInstance(args.output_folder, str)
        self.assertIsInstance(args.eeg_format, str)
        self.assertIsInstance(args.montage_type, str)
        self.assertIsInstance(args.montage_channels, str)
        self.assertIsInstance(args.power_line_freq, int)
        self.assertIsInstance(args.force_characterization, str)
        self.assertIsInstance(args.force_hfo_detection, str)
        self.assertIsInstance(args.start_sec, float)
        self.assertIsInstance(args.end_sec, float)
        self.assertIsInstance(args.wdw_step_s, float)
        self.assertIsInstance(args.n_jobs, int)
        self.assertIsInstance(args.verbose, str)

    def test_create_test_args_paths(self):
        """Test that create_test_args has path-like strings."""
        args = create_test_args()
        
        # Check that input/output folders are path-like strings
        self.assertIn('/', args.input_folder)  # Should be an absolute path
        self.assertIn('/', args.output_folder)  # Should be an absolute path
        
        # Check that they start with / (absolute paths)
        self.assertTrue(args.input_folder.startswith('/'))
        self.assertTrue(args.output_folder.startswith('/'))


if __name__ == '__main__':
    unittest.main()
