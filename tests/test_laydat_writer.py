import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, mock_open
import numpy as np
import pandas as pd
from datetime import datetime
import mne

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the modules to test
from hfo_spectral_detector.eeg_io.laydat_writer import write_dat


class TestLaydatWriter(unittest.TestCase):
    """Test laydat_writer module functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.eeg_data_path = self.temp_dir / "eeg_data"
        self.output_path = self.temp_dir / "output"
        self.eeg_data_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create mock EEG files
        self.mock_eeg_file1 = self.eeg_data_path / "patient-001.lay"
        self.mock_eeg_file2 = self.eeg_data_path / "patient-002.lay"
        self.mock_eeg_file1.touch()
        self.mock_eeg_file2.touch()

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch('hfo_spectral_detector.eeg_io.laydat_writer.get_files_in_folder')
    def test_write_dat_empty_file_list(self, mock_get_files):
        """Test behavior when no EEG files are found."""
        mock_get_files.return_value = []
        
        result_df = write_dat(
            eeg_data_path=str(self.eeg_data_path),
            pat_id=1,
            output_path=str(self.output_path)
        )
        
        # Should return an empty DataFrame
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 0)

    @patch('hfo_spectral_detector.eeg_io.laydat_writer.get_files_in_folder')
    def test_write_dat_function_structure(self, mock_get_files):
        """Test that the function has proper structure and error handling."""
        # Test with empty file list
        mock_get_files.return_value = []
        
        result = write_dat(
            eeg_data_path=str(self.eeg_data_path),
            pat_id=1,
            output_path=str(self.output_path)
        )
        
        # Verify function calls and return type
        mock_get_files.assert_called_once_with(str(self.eeg_data_path), '.lay')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_write_dat_input_validation(self):
        """Test input validation and type conversion.""" 
        # Test that different patient ID types work
        with patch('hfo_spectral_detector.eeg_io.laydat_writer.get_files_in_folder') as mock_get_files:
            mock_get_files.return_value = []
            
            # Test with integer patient ID
            result1 = write_dat(
                eeg_data_path=str(self.eeg_data_path),
                pat_id=1,
                output_path=str(self.output_path)
            )
            self.assertIsInstance(result1, pd.DataFrame)
            
            # Test with different integer patient ID
            result2 = write_dat(
                eeg_data_path=str(self.eeg_data_path),
                pat_id=42,
                output_path=str(self.output_path)
            )
            self.assertIsInstance(result2, pd.DataFrame)

    def test_write_dat_basic_structure(self):
        """Test the basic structure and error handling of write_dat function."""
        # Test with invalid inputs
        with self.assertRaises(TypeError):
            write_dat()  # No arguments
        
        # Test with non-existent directory
        result = write_dat(
            eeg_data_path="/nonexistent/path",
            pat_id=1,
            output_path=str(self.output_path)
        )
        # Should return empty DataFrame since no files found
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()
