import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
import mne

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the modules to test
from hfo_spectral_detector.eeg_io.eeg_io import EEG_IO


class TestEEGIO(unittest.TestCase):
    """Test EEG_IO class methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_file = self.temp_dir / "test.edf"
        
        # Create mock MNE raw object
        self.mock_raw = Mock(spec=mne.io.BaseRaw)
        self.mock_raw.filenames = [Path("test.edf")]
        self.mock_raw.info = {
            "sfreq": 2048.0,
            "meas_date": "2024-01-01"
        }
        self.mock_raw._orig_units = "uV"
        self.mock_raw.times = np.linspace(0, 10, 20480)  # 10 seconds at 2048 Hz
        self.mock_raw.n_times = 20480
        self.mock_raw.ch_names = [
            "Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "A1",
            "Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6", "A2",
            "Fz", "Cz", "Pz", "ECG", "EMG"
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_init_scalp_referential(self, mock_read_edf):
        """Test EEG_IO initialization with scalp referential montage."""
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='sr')
        
        self.assertEqual(eeg_io.fs, 2048)
        self.assertEqual(eeg_io.filename, "test.edf")
        self.assertEqual(eeg_io.n_samples, 20480)
        self.assertEqual(eeg_io.units, "uV")
        self.assertEqual(eeg_io.get_ch_names, eeg_io.get_scalp_ref_chann_labels)
        self.assertEqual(eeg_io.get_data, eeg_io.get_referential_data)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_init_scalp_bipolar(self, mock_read_edf):
        """Test EEG_IO initialization with scalp bipolar montage."""
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='sb')
        
        self.assertEqual(eeg_io.get_ch_names, eeg_io.get_scalp_long_bip_chann_labels)
        self.assertEqual(eeg_io.get_data, eeg_io.get_bipolar_data)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_init_ieeg_referential(self, mock_read_edf):
        """Test EEG_IO initialization with iEEG referential montage."""
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='ir')
        
        self.assertEqual(eeg_io.get_ch_names, eeg_io.get_ieeg_ref_chann_labels)
        self.assertEqual(eeg_io.get_data, eeg_io.get_referential_data)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_init_ieeg_bipolar(self, mock_read_edf):
        """Test EEG_IO initialization with iEEG bipolar montage."""
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='ib')
        
        self.assertEqual(eeg_io.get_ch_names, eeg_io.get_ieeg_bip_chann_labels)
        self.assertEqual(eeg_io.get_data, eeg_io.get_bipolar_data)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_init_invalid_montage_type(self, mock_read_edf):
        """Test EEG_IO initialization with invalid montage type."""
        mock_read_edf.return_value = self.mock_raw
        
        with self.assertRaises(ValueError) as context:
            EEG_IO(str(self.test_file), mtg_t='invalid')
        
        self.assertIn("Invalid montage type", str(context.exception))

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_init_empty_units(self, mock_read_edf):
        """Test EEG_IO initialization with empty units."""
        self.mock_raw._orig_units = ''
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='sr')
        
        self.assertEqual(eeg_io.units, 'uV')

    def test_read_eeg_header_edf(self):
        """Test reading EEG header for EDF file."""
        with patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf') as mock_read:
            mock_read.return_value = self.mock_raw
            
            test_file_path = self.temp_dir / "test.edf"
            test_file_path.touch()  # Create the file
            
            eeg_io = EEG_IO.__new__(EEG_IO)  # Create instance without __init__
            result = eeg_io.read_eeg_header(str(test_file_path))
            
            mock_read.assert_called_once_with(str(test_file_path), verbose='ERROR')
            self.assertEqual(result, self.mock_raw)

    def test_read_eeg_header_lay(self):
        """Test reading EEG header for LAY file."""
        with patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_persyst') as mock_read:
            mock_read.return_value = self.mock_raw
            
            test_file_path = self.temp_dir / "test.lay"
            test_file_path.touch()  # Create the file
            
            eeg_io = EEG_IO.__new__(EEG_IO)
            result = eeg_io.read_eeg_header(str(test_file_path))
            
            mock_read.assert_called_once_with(str(test_file_path), verbose='ERROR')
            self.assertEqual(result, self.mock_raw)

    def test_read_eeg_header_vhdr(self):
        """Test reading EEG header for VHDR file."""
        with patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_brainvision') as mock_read:
            mock_read.return_value = self.mock_raw
            
            test_file_path = self.temp_dir / "test.vhdr"
            test_file_path.touch()  # Create the file
            
            eeg_io = EEG_IO.__new__(EEG_IO)
            result = eeg_io.read_eeg_header(str(test_file_path))
            
            mock_read.assert_called_once_with(str(test_file_path), verbose='ERROR')
            self.assertEqual(result, self.mock_raw)

    def test_clean_channel_labels(self):
        """Test channel label cleaning."""
        eeg_io = EEG_IO.__new__(EEG_IO)
        
        # Test removing spaces and -Ref
        input_labels = [" Fp1-Ref ", " C03 ", "F007-Ref", "Cz"]
        expected = ["Fp1", "C3", "F7", "Cz"]
        
        result = eeg_io.clean_channel_labels(input_labels)
        
        self.assertEqual(result, expected)

    def test_clean_channel_labels_leading_zeros(self):
        """Test channel label cleaning with leading zeros."""
        eeg_io = EEG_IO.__new__(EEG_IO)
        
        input_labels = ["F007", "C003", "P001", "O002"]
        expected = ["F7", "C3", "P1", "O2"]
        
        result = eeg_io.clean_channel_labels(input_labels)
        
        self.assertEqual(result, expected)

    def test_clean_channel_labels_no_digits(self):
        """Test channel label cleaning with no digits."""
        eeg_io = EEG_IO.__new__(EEG_IO)
        
        input_labels = ["Fpz", "Cz", "Oz"]
        expected = ["Fpz", "Cz", "Oz"]
        
        result = eeg_io.clean_channel_labels(input_labels)
        
        self.assertEqual(result, expected)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_get_scalp_ref_chann_labels(self, mock_read_edf):
        """Test getting scalp referential channel labels."""
        self.mock_raw.ch_names = ["Fp1", "F3", "C3", "P3", "O1", "ECG", "INVALID"]
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='sr')
        
        with patch('builtins.print'):  # Suppress print output
            labels, indices = eeg_io.get_scalp_ref_chann_labels()
        
        # Should exclude ECG and INVALID channels
        expected_labels = ["Fp1", "F3", "C3", "P3", "O1"]
        self.assertEqual(len(labels), 5)
        self.assertEqual(len(indices), 5)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_get_scalp_long_bip_chann_labels(self, mock_read_edf):
        """Test getting scalp long bipolar channel labels."""
        self.mock_raw.ch_names = ["Fp1", "F7", "T7", "P7", "O1", "Fp2", "F8"]
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='sb')
        
        with patch('builtins.print'):  # Suppress print output
            labels, indices = eeg_io.get_scalp_long_bip_chann_labels()
        
        # Should include Fp1-F7, F7-T7, T7-P7, P7-O1, Fp2-F8 (if they exist)
        self.assertIsInstance(labels, list)
        self.assertIsInstance(indices, list)
        self.assertEqual(len(labels), len(indices))

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_get_ieeg_ref_chann_labels(self, mock_read_edf):
        """Test getting iEEG referential channel labels."""
        # Mix of scalp, iEEG, and non-EEG channels
        self.mock_raw.ch_names = [
            "Fp1", "F3", "C3",  # Scalp channels
            "LAD1", "LAD2", "RAD1", "RAD2",  # iEEG channels
            "ECG", "EMG"  # Non-EEG channels
        ]
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='ir')
        
        labels, indices = eeg_io.get_ieeg_ref_chann_labels()
        
        # Should only include iEEG channels (LAD1, LAD2, RAD1, RAD2)
        expected_labels = ["LAD1", "LAD2", "RAD1", "RAD2"]
        self.assertEqual(len(labels), 4)
        self.assertEqual(len(indices), 4)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_get_ieeg_bip_chann_labels(self, mock_read_edf):
        """Test getting iEEG bipolar channel labels."""
        self.mock_raw.ch_names = ["LAD1", "LAD2", "LAD3", "RAD1", "RAD2"]
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='ib')
        
        with patch('builtins.print'):  # Suppress print output for invalid channel names
            labels, indices = eeg_io.get_ieeg_bip_chann_labels()
        
        # Should create bipolar pairs: LAD1-LAD2, LAD2-LAD3, RAD1-RAD2
        expected_pairs = ["LAD1-LAD2", "LAD2-LAD3", "RAD1-RAD2"]
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(indices), 3)
        
        # Check that indices are tuples of two channel indices
        for idx_pair in indices:
            self.assertIsInstance(idx_pair, tuple)
            self.assertEqual(len(idx_pair), 2)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_get_referential_data(self, mock_read_edf):
        """Test getting referential data."""
        # Create mock data
        mock_data = np.random.rand(5, 1000)  # 5 channels, 1000 samples
        self.mock_raw.get_data.return_value = mock_data
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='sr')
        eeg_io.ch_indices = [0, 1, 2, 3, 4]  # Mock channel indices
        
        result = eeg_io.get_referential_data(start=0, stop=1000)
        
        self.mock_raw.get_data.assert_called_once_with(picks=[0, 1, 2, 3, 4], start=0, stop=1000)
        np.testing.assert_array_equal(result, mock_data)
        np.testing.assert_array_equal(eeg_io.data, mock_data)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_get_referential_data_with_picks(self, mock_read_edf):
        """Test getting referential data with specific picks."""
        mock_data = np.random.rand(1, 1000)  # 1 channel, 1000 samples
        self.mock_raw.get_data.return_value = mock_data
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='sr')
        eeg_io.ch_indices = [0, 1, 2, 3, 4]
        
        result = eeg_io.get_referential_data(picks=2, start=0, stop=1000)
        
        self.mock_raw.get_data.assert_called_once_with(picks=2, start=0, stop=1000)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_get_bipolar_data(self, mock_read_edf):
        """Test getting bipolar data."""
        # Mock channel data
        mock_data_a = np.array([[1, 2, 3, 4, 5]])  # Channel A
        mock_data_b = np.array([[0.5, 1, 1.5, 2, 2.5]])  # Channel B
        
        def side_effect(picks, start, stop):
            if picks == 0:
                return mock_data_a
            elif picks == 1:
                return mock_data_b
            return np.array([[0, 0, 0, 0, 0]])
        
        self.mock_raw.get_data.side_effect = side_effect
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='sb')
        eeg_io.ch_indices = [(0, 1)]  # One bipolar pair
        eeg_io.n_samples = 5
        
        result = eeg_io.get_bipolar_data(start=0, stop=5)
        
        expected = np.array([[0.5, 1, 1.5, 2, 2.5]])  # A - B
        np.testing.assert_array_equal(result, expected)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_get_bipolar_data_single_channel(self, mock_read_edf):
        """Test getting bipolar data for single channel."""
        mock_data_a = np.array([[1, 2, 3, 4, 5]])
        mock_data_b = np.array([[0.5, 1, 1.5, 2, 2.5]])
        
        def side_effect(picks, start, stop):
            if picks == 0:
                return mock_data_a
            elif picks == 1:
                return mock_data_b
            return np.array([[0, 0, 0, 0, 0]])
        
        self.mock_raw.get_data.side_effect = side_effect
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='sb')
        eeg_io.ch_indices = [(0, 1)]  # One bipolar pair
        eeg_io.n_samples = 5
        
        result = eeg_io.get_bipolar_data(picks=0, start=0, stop=5)
        
        expected = np.array([0.5, 1, 1.5, 2, 2.5])  # Flattened A - B
        np.testing.assert_array_equal(result, expected)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_get_bipolar_data_invalid_samples(self, mock_read_edf):
        """Test getting bipolar data with invalid sample ranges."""
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='sb')
        eeg_io.ch_indices = [(0, 1)]
        eeg_io.n_samples = 1000
        
        # Test start > stop
        with self.assertRaises(ValueError) as context:
            eeg_io.get_bipolar_data(start=100, stop=50)
        self.assertIn("Start sample is greater than stop sample", str(context.exception))
        
        # Test start == stop
        with self.assertRaises(ValueError) as context:
            eeg_io.get_bipolar_data(start=50, stop=50)
        self.assertIn("Start sample is equal to stop sample", str(context.exception))

    def test_get_all_possible_scalp_long_bip_labels(self):
        """Test getting all possible scalp long bipolar labels."""
        eeg_io = EEG_IO.__new__(EEG_IO)
        
        labels = eeg_io.get_all_possible_scalp_long_bip_labels()
        
        expected_labels = [
            "Fp1-F7", "F7-T7", "T7-P7", "P7-O1", "F7-T3", "T3-T5", "T5-O1",
            "Fp2-F8", "F8-T8", "T8-P8", "P8-O2", "F8-T4", "T4-T6", "T6-O2",
            "Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "Fp2-F4", "F4-C4", "C4-P4",
            "P4-O2", "FZ-CZ", "CZ-PZ"
        ]
        
        self.assertEqual(labels, expected_labels)

    def test_get_valid_scalp_channel_labels(self):
        """Test getting valid scalp channel labels."""
        eeg_io = EEG_IO.__new__(EEG_IO)
        
        labels = eeg_io.get_valid_scalp_channel_labels()
        
        # Check that common scalp channels are included
        expected_channels = ["A1", "A2", "C3", "C4", "CZ", "F3", "F4", "F7", "F8",
                           "Fp1", "Fp2", "FZ", "O1", "O2", "P3", "P4", "T7", "T8"]
        
        for channel in expected_channels:
            self.assertIn(channel, labels)
        
        self.assertIsInstance(labels, list)
        self.assertGreater(len(labels), 30)  # Should have many valid scalp channels

    def test_get_non_eeg_channel_labels(self):
        """Test getting non-EEG channel labels."""
        eeg_io = EEG_IO.__new__(EEG_IO)
        
        labels = eeg_io.get_non_eeg_channel_labels()
        
        expected_labels = ["ECG", "EKG", "EKGR", "EKGL", "EOG", "EMG", "EOG", "TRIG", "OSAT", "PLETH", "EVENT"]
        
        self.assertEqual(labels, expected_labels)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    @patch('hfo_spectral_detector.eeg_io.eeg_io.plt.show')
    @patch('hfo_spectral_detector.eeg_io.eeg_io.plt.pause')
    @patch('hfo_spectral_detector.eeg_io.eeg_io.plt.close')
    def test_plot_sample_wdw_per_ch(self, mock_close, mock_pause, mock_show, mock_read_edf):
        """Test plotting sample window per channel."""
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='sr')
        eeg_io.ch_names = ["C3", "C4"]
        eeg_io.data = np.random.rand(2, 1000)
        eeg_io.time_s = np.linspace(0, 10, 1000)
        eeg_io.units = "uV"
        
        with patch.object(eeg_io, 'plot_ch_signal') as mock_plot:
            eeg_io.plot_sample_wdw_per_ch()
            
            # Should call plot_ch_signal for each channel
            self.assertEqual(mock_plot.call_count, 2)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.plt.show')
    @patch('hfo_spectral_detector.eeg_io.eeg_io.plt.pause') 
    @patch('hfo_spectral_detector.eeg_io.eeg_io.plt.close')
    def test_plot_ch_signal(self, mock_close, mock_pause, mock_show):
        """Test plotting channel signal."""
        eeg_io = EEG_IO.__new__(EEG_IO)
        
        # Create test data
        time_s = np.linspace(0, 30, 3000)  # 30 seconds
        ch_signal = np.sin(2 * np.pi * 10 * time_s)  # 10 Hz sine wave
        plt_wdw_s = (10, 20)  # 10-20 second window
        
        eeg_io.plot_ch_signal(
            filename="test.edf",
            ch_signal=ch_signal,
            ch_name="C3",
            units_str="uV",
            time_s=time_s,
            plt_wdw_s=plt_wdw_s
        )
        
        # Verify matplotlib functions were called
        mock_show.assert_called_once()
        mock_pause.assert_called_once_with(0.5)
        mock_close.assert_called_once()

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_remove_natus_virtual_channels_bipolar(self, mock_read_edf):
        """Test removing Natus virtual channels for bipolar montage."""
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='sb')
        eeg_io.ch_names = ["Fp1-F7", "C3-P3", "C001-C002", "DC01-DC02", "F3-C3"]
        eeg_io.ch_indices = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        
        with patch('builtins.print'):  # Suppress print output
            eeg_io.remove_natus_virtual_channels()
        
        # Should keep Fp1-F7, C3-P3, F3-C3 and exclude C001-C002, DC01-DC02
        expected_names = ["Fp1-F7", "C3-P3", "F3-C3"]
        self.assertEqual(len(eeg_io.ch_names), 3)

    @patch('hfo_spectral_detector.eeg_io.eeg_io.mne.io.read_raw_edf')
    def test_remove_natus_virtual_channels_referential(self, mock_read_edf):
        """Test removing Natus virtual channels for referential montage."""
        mock_read_edf.return_value = self.mock_raw
        
        eeg_io = EEG_IO(str(self.test_file), mtg_t='sr')
        eeg_io.ch_names = ["Fp1", "C3", "C001", "DC01", "F3", "Cz"]
        eeg_io.ch_indices = [0, 1, 2, 3, 4, 5]
        
        with patch('builtins.print'):  # Suppress print output
            eeg_io.remove_natus_virtual_channels()
        
        # Should keep Fp1, C3, F3, Cz and exclude C001, DC01
        expected_names = ["Fp1", "C3", "F3", "Cz"]
        self.assertEqual(len(eeg_io.ch_names), 4)


if __name__ == '__main__':
    unittest.main()
