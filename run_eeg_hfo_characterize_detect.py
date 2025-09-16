import sys
import os
import argparse
import socket
import logging
from typing import List, Dict, Optional
from contextlib import contextmanager
import time

import mne
import numpy as np

from pathlib import Path
from hfo_spectral_detector.spectral_analyzer.characterize_events import characterize_events
from hfo_spectral_detector.eeg_io.eeg_io import EEG_IO
from hfo_spectral_detector.prediction.predict_characterize_hfo import HFO_Detector

# Module-level constants for better performance
DEFAULT_SAMPLING_RATE_THRESHOLD = 1000
DEFAULT_WINDOW_LENGTH_SECONDS = 1.0  # do not change this value, it is used in the HFO detector
DEFAULT_SAVE_SPECT_IMAGE = False
LOG_FORMAT = '%(asctime)s %(levelname)-8s %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


@contextmanager
def timing_context(operation_name: str, logger: logging.Logger):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"{operation_name} completed in {elapsed_time:.2f} seconds")
        print(f"{operation_name} completed in {elapsed_time:.2f} seconds")


def init_logging() -> logging.Logger:
    """Initialize logging configuration and return logger instance."""
    # Configure the root logger so all child loggers inherit the configuration
    root_logger = logging.getLogger()
    
    # Avoid duplicate handlers if root logger already configured
    if root_logger.handlers:
        return logging.getLogger(__name__)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_fpath = Path(__file__).parent / "logs" / f"hfo_spectral_detector_{timestamp}.log"
    os.makedirs(log_fpath.parent, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    
    # File handler
    file_handler = logging.FileHandler(log_fpath)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)
    
    # Configure root logger so all child loggers inherit these handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    # Return logger for this module
    return logging.getLogger(__name__)


class Characterization_Config:
    """Configuration class for EEG characterization parameters with validation.
    
    Attributes:
        dataset_name: String identifier for the dataset
        rm_vchann: Whether to remove Natus virtual channels
        input_folder: Path object for input directory
        output_folder: Path object for output directory
        eeg_format: File format (edf, fif, etc.)
        montage_type: Electrode montage type (ib, ir, sb, sr)
        montage_channels: Comma separated montage channels to detect, empty detects all (e.g., "F3-C3,C3-P3, F4-C4,C4-P4")
        power_line_freqs: Power line frequency (0, 50 or 60 Hz)
        start_sec: Analysis start time in seconds
        end_sec: Analysis end time in seconds
        wdw_step_s: Window step size in seconds
        force_characterization: Whether to force recalculation
        force_hfo_detection: Whether to force HFO detection
        n_jobs: Number of parallel jobs
        save_spect_img: Whether to save spectrogram images
    """

    def __init__(self, args, test_mode: bool = False):
        self.dataset_name = args.dataset_name
        self.rm_vchann = args.rm_vchann.lower()
        self.input_folder = Path(args.input_folder)
        self.output_folder = Path(args.output_folder)
        self.eeg_format = args.eeg_format.lower()
        self.montage_type = args.montage_type.lower()
        self.montage_channels = str(args.montage_channels.lower()).lower().replace(" ", "")
        self.power_line_freqs = int(args.power_line_freq)
        self.start_sec = float(args.start_sec)
        self.end_sec = float(args.end_sec)
        self.wdw_step_s = float(args.wdw_step_s)
        self.force_characterization = args.force_characterization.lower()
        self.force_hfo_detection = args.force_hfo_detection.lower()
        self.n_jobs = int(args.n_jobs)
        self.verbose = args.verbose.lower()
        self.save_spect_img = (self.n_jobs==1 and DEFAULT_SAVE_SPECT_IMAGE) and test_mode

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.rm_vchann not in ['yes', 'no']:
            raise ValueError("rm_vchann must be 'yes' or 'no'")
        
        if not self.input_folder.exists():
            raise ValueError(f"Input folder does not exist: {self.input_folder}")
        
        if self.eeg_format not in ['edf', 'dat', 'vhdr']:
            raise ValueError(f"Unsupported EEG format: {self.eeg_format}")
        
        if self.montage_type not in ['ib', 'ir', 'sb', 'sr']:
            raise ValueError(f"Invalid montage type: {self.montage_type}")
                
        if self.power_line_freqs not in [0, 50, 60]:
            raise ValueError(f"Invalid power line frequency: {self.power_line_freqs}")
        
        if self.start_sec < 0:
            raise ValueError("Start time cannot be negative")
        
        if self.end_sec > 0 and self.end_sec <= self.start_sec:
            raise ValueError("End time must be greater than start time")
        elif self.end_sec != -1 and self.end_sec <= 0:
            raise ValueError("End time must be positive or -1 for full length")
        
        if self.wdw_step_s <= 0:
            raise ValueError("Window step size must be positive")
        
        if self.force_characterization not in ['yes', 'no']:
            raise ValueError("force_characterization must be 'yes' or 'no'")
        
        if self.force_hfo_detection not in ['yes', 'no']:
            raise ValueError("force_hfo_detection must be 'yes' or 'no'")
        
        if self.verbose not in ['yes', 'no']:
            raise ValueError("verbose must be 'yes' or 'no'")

    @property
    def montage_channels_list(self) -> list:
        """Get montage_channels as list."""
        return self.montage_channels.split(",") if self.montage_channels else []

    @property
    def rm_vchann_bool(self) -> bool:
        """Get rm_vchann as boolean."""
        return self.rm_vchann == "yes"
    
    @property
    def force_characterization_bool(self) -> bool:
        """Get force_characterization as boolean."""
        return self.force_characterization == "yes"

    @property
    def force_hfo_detection_bool(self) -> bool:
        """Get force_hfo_detection as boolean."""
        return self.force_hfo_detection == "yes"

    @property
    def verbose_bool(self) -> bool:
        """Get verbose as boolean."""
        return self.verbose == "yes"

    def display_config(self) -> None:
        """Display configuration parameters to console."""
        config_str = (
            f"Dataset = {self.dataset_name}\n"
            f"Remove Natus virtual channels = {self.rm_vchann}\n"
            f"Input folder = {self.input_folder}\n"
            f"Output folder = {self.output_folder}\n"
            f"EEG format = {self.eeg_format}\n"
            f"Montage type = {self.montage_type}\n"
            f"Montage channels = {self.montage_channels}\n"
            f"Power line frequencies = {self.power_line_freqs}\n"
            f"Force characterization = {self.force_characterization}\n"
            f"Force HFO detection = {self.force_hfo_detection}\n"
            f"Verbose = {self.verbose}\n"
            f"Save spectrogram images = {self.save_spect_img}"
        )
        print(config_str)

    def log_config(self, logger: logging.Logger) -> None:
        """Log configuration parameters efficiently."""
        config_items = [
            f"Dataset = {self.dataset_name}",
            f"Remove Natus virtual channels = {self.rm_vchann}",
            f"Input folder = {self.input_folder}",
            f"Output folder = {self.output_folder}",
            f"EEG format = {self.eeg_format}",
            f"Montage type = {self.montage_type}",
            f"Montage channels = {self.montage_channels}",
            f"Power line frequencies = {self.power_line_freqs}",
            f"Force characterization = {self.force_characterization}",
            f"Force HFO detection = {self.force_hfo_detection}",
            f"Verbose = {self.verbose}",
            f"Save spectrogram images = {self.save_spect_img}"
        ]
        logger.info("Configuration:\n" + "\n".join(config_items))


class InputDataExtractor:
    """Class to extract and manage input EEG files with caching."""
    
    def __init__(self, cfg: Characterization_Config):
        self.cfg = cfg
        self._files_cache: Optional[List[Path]] = None

    def get_files_to_process(self) -> List[Path]:
        """Get the filepaths from the EEG files in the input folder with caching."""
        if self._files_cache is None:
            pattern = f"**/*.{self.cfg.eeg_format}"
            self._files_cache = list(self.cfg.input_folder.glob(pattern, case_sensitive=False))
            self._files_cache.sort()  # Ensure consistent ordering
        return self._files_cache

    def display_files_to_process(self) -> None:
        """Display files to process to console."""
        files = self.get_files_to_process()
        if not files:
            print(f"No {self.cfg.eeg_format} files found in {self.cfg.input_folder}")
            return
        
        print(f"Found {len(files)} files to process:")
        for f in files:
            print(f"  {f}")
            
    def log_files_to_process(self, logger: logging.Logger) -> None:
        """Log files to process efficiently."""
        files = self.get_files_to_process()
        if not files:
            logger.warning(f"No {self.cfg.eeg_format} files found in {self.cfg.input_folder}")
            return
        
        logger.info(f"Found {len(files)} EEG files to process")
        for f in files:
            logger.info(f"EEG File to process: {f}")


class EEGValidationError(Exception):
    """Custom exception for EEG validation errors."""
    pass


class AnalysisWindowError(Exception):
    """Custom exception for analysis window validation errors."""
    pass


class EEGProcessingError(Exception):
    """Custom exception for EEG processing errors."""
    pass


def _calculate_analysis_windows(
    analysis_start_sample: int, 
    analysis_end_sample: int, 
    window_length_samples: int, 
    window_step_samples: int
) -> Dict[str, np.ndarray]:
    """Calculate analysis windows"""
    start_samples = np.arange(
        analysis_start_sample, 
        analysis_end_sample - window_length_samples + 1, 
        window_step_samples,
        dtype=np.int64
    )
    
    end_samples = start_samples + window_length_samples
    
    return {
        'start': start_samples,
        'end': end_samples
    }


def _validate_eeg_data(eeg_reader: EEG_IO, fs_threshold: int = DEFAULT_SAMPLING_RATE_THRESHOLD) -> None:
    """Validate EEG data properties."""
    if eeg_reader.fs <= fs_threshold:
        raise EEGValidationError(f"Sampling Rate is {eeg_reader.fs} Hz, which is under {fs_threshold} Hz!")
    
    if eeg_reader.n_samples <= 0:
        raise EEGValidationError("EEG data has no samples!")
    
    if len(eeg_reader.ch_names) == 0:
        raise EEGValidationError("EEG data has no channels!")


def _validate_analysis_windows(
    windows_dict: Dict[str, np.ndarray], 
    max_sample: int
) -> None:
    """Validate analysis windows."""
    if np.any(windows_dict['start'] < 0):
        raise AnalysisWindowError("Incorrectly defined analysis window start samples")
    
    if np.any(windows_dict['end'] > max_sample):
        raise AnalysisWindowError("Incorrectly defined analysis window end samples")


def _log_eeg_info(
    logger: logging.Logger, 
    pat_name: str, 
    eeg_reader: EEG_IO, 
    cfg: Characterization_Config,
    analysis_start_sample: int,
    analysis_end_sample: int
) -> None:
    """Log EEG information efficiently."""
    info_lines = [
        f"Patient: {pat_name}",
        f"EEG Duration: {eeg_reader.n_samples/eeg_reader.fs:.2f} seconds",
        f"EEG Sampling Rate: {eeg_reader.fs} Hz",
        f"EEG Nr. Samples: {eeg_reader.n_samples}",
        f"Analysis start second: {cfg.start_sec}",
        f"Analysis end second: {cfg.end_sec}",
        f"Analysis start sample: {analysis_start_sample}",
        f"Analysis end sample: {analysis_end_sample}",
        f"Nr. Channels: {len(eeg_reader.ch_names)}",
        f"Channels: {eeg_reader.ch_names}"
    ]
    logger.info("\n".join(info_lines))


def run_eeg_characterization(
    cfg: Characterization_Config, 
    files_to_process: List[Path], 
    logger: logging.Logger
) -> None:
    """
    Run EEG characterization and HFO detection on the provided files.
    
    Args:
        cfg: Configuration object containing analysis parameters
        files_to_process: List of EEG file paths to process
        logger: Logger instance for logging messages
    """
    if not files_to_process:
        logger.warning("No files to process")
        return

    os.makedirs(cfg.output_folder, exist_ok=True)

    # Create the detector object once for all files
    detector_results_path = cfg.output_folder / 'Elpi_Detector_Results'
    with timing_context("Detector initialization", logger):
        detector = HFO_Detector(output_path=detector_results_path)
        detector.load_models()

    processed_count = 0
    error_count = 0

    for eeg_fpath in files_to_process:
        pat_name = eeg_fpath.stem
        
        try:
            with timing_context(f"Processing {pat_name}", logger):
                process_single_eeg_file(cfg, eeg_fpath, detector, logger)
            processed_count += 1
            
        except (EEGValidationError, AnalysisWindowError, EEGProcessingError) as e:
            logger.error(f"Validation/Processing error for {pat_name}: {e}")
            error_count += 1
            continue
            
        except Exception as e:
            logger.error(f"Unexpected error processing {pat_name}: {e}", exc_info=True)
            error_count += 1
            continue

    # Log summary
    logger.info(f"Processing completed: {processed_count} successful, {error_count} errors")


def process_single_eeg_file(
    cfg: Characterization_Config,
    eeg_fpath: Path,
    detector: HFO_Detector,
    logger: logging.Logger
) -> None:
    """Process a single EEG file."""
    pat_name = eeg_fpath.stem
    
    # Read EEG Data
    try:
        eeg_reader = EEG_IO(eeg_filepath=eeg_fpath, mtg_t=cfg.montage_type)
    except Exception as e:
        raise EEGProcessingError(f"Failed to read EEG file: {e}")
    
    fs = eeg_reader.fs
    if fs <= 0:
        raise EEGProcessingError(f"Invalid sampling rate: {fs} Hz")    

    if cfg.rm_vchann_bool:
        eeg_reader.remove_natus_virtual_channels()
    
    # Validate EEG data
    _validate_eeg_data(eeg_reader)

    # Calculate analysis parameters
    window_length_samples = int(DEFAULT_WINDOW_LENGTH_SECONDS * fs)
    analysis_start_sample = int(cfg.start_sec * fs)
    analysis_end_sample = eeg_reader.n_samples
    
    if cfg.end_sec > 0:
        analysis_end_sample = int(cfg.end_sec * fs)
        
    window_step_samples = int(np.round(cfg.wdw_step_s * fs))

    # Log EEG information
    _log_eeg_info(logger, pat_name, eeg_reader, cfg, analysis_start_sample, analysis_end_sample)

    # Calculate analysis windows
    an_wdws_dict = _calculate_analysis_windows(
        analysis_start_sample, 
        analysis_end_sample, 
        window_length_samples, 
        window_step_samples
    )
    
    # Validate analysis windows
    _validate_analysis_windows(an_wdws_dict, analysis_end_sample)

    # Prepare parameters for characterization
    params = {
        "pat_name": pat_name,
        "eeg_reader": eeg_reader,
        "mtgs_to_detect": cfg.montage_channels_list,
        "an_wdws_dict": an_wdws_dict,
        "out_path": cfg.output_folder,
        "power_line_freqs": cfg.power_line_freqs,
        "n_jobs": cfg.n_jobs,
        "force_recalc": cfg.force_characterization_bool,
        "verbose": cfg.verbose_bool,
        "save_spect_img": cfg.save_spect_img
    }

    # Run characterization and detection
    try:
        with timing_context(f"Feature characterization for {pat_name}", logger):
            allch_events_fpath = characterize_events(**params)
        
        with timing_context(f"HFO detection for {pat_name}", logger):
            detector.set_fs(fs)
            detector.run_hfo_detection(
                pat_name, 
                allch_events_fpath, 
                force_recalc=cfg.force_hfo_detection_bool
            )
    except Exception as e:
        raise EEGProcessingError(f"Failed during characterization/detection: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='Characterize EEG to detect HFO',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset_name', type=str, required=True, 
                       help='Name of the dataset')
    parser.add_argument('--input_folder', type=str, required=True, 
                       help='Path to directory containing EEG files')
    parser.add_argument('--output_folder', type=str, required=True, 
                       help='Path to the output directory')
    parser.add_argument('--eeg_format', type=str, default="edf", 
                       help='File format of the EEG files')
    parser.add_argument('--montage_type', type=str, required=True, 
                       help='Name of the montage (ib, ir, sb, sr)')
    parser.add_argument('--montage_channels', type=str, default="",
                       help='Comma separated montage channels to detect, empty detects all (e.g., "F3-C3,C3-P3, F4-C4,C4-P4"')
    parser.add_argument('--rm_vchann', type=str, default="yes", 
                    choices=['yes', 'no'], help='Remove Natus virtual channels if present')
    parser.add_argument('--power_line_freq', type=int, default=60, 
                       help='Frequency of Power Lines (0 to turn off automatic power-line noise notch filtering, otherwise 50 or 60)')
    parser.add_argument('--start_sec', type=float, default=0,
                       help='Start analysis from a specific second')
    parser.add_argument('--end_sec', type=float, default=-1, 
                       help='End analysis from a specific second, -1 is full length')
    parser.add_argument('--wdw_step_s', type=float, default=0.1, 
                       help='Window step size in seconds')
    parser.add_argument('--force_characterization', type=str, default="no", 
                       choices=['yes', 'no'], help='Force recalculation of features')
    parser.add_argument('--force_hfo_detection', type=str, default="yes", 
                       choices=['yes', 'no'], help='Force HFO detection')
    parser.add_argument('--n_jobs', type=int, default=-1, 
                       help='Number of jobs to run in parallel, -1 uses all CPU cores')
    parser.add_argument('--verbose', type=str, default="yes", 
                       choices=['yes', 'no'], help='Enable verbose output')
    
    return parser


def create_test_args():
    """Create test arguments for debugging mode."""
    class TestArgs:
        def __init__(self):
            self.dataset_name = "Overcomplete_Validated_Blobs_Physio_Patient_Anonymized"
            self.rm_vchann = "yes"
            self.input_folder = "/home/dlp/Documents/Development/Data/Overcomplete_Validated_Blobs_Physio_Patient_Anonymized"
            self.output_folder = "/home/dlp/Documents/Development/Data/HFO_Output/"
            self.eeg_format = "edf"
            self.montage_type = "sb"
            self.montage_channels = "" #"F3-C3,C3-P3,F4-C4,C4-P4"
            self.power_line_freq = 60
            self.force_characterization = "yes"
            self.force_hfo_detection = "yes"
            self.start_sec = 0.0
            self.end_sec = 20.0
            self.wdw_step_s = 0.1
            self.n_jobs = 1
            self.verbose = "yes"
    
    return TestArgs()


def main() -> None:
    """Main execution function."""
    # Initialize logging
    logger = init_logging()

    # Automatically enable test mode when running in debugger or no arguments passed
    test_mode = sys.gettrace() is not None or len(sys.argv) == 1

    try:
        if test_mode:
            args = create_test_args()
            logger.info("Running in test mode")
        else:
            parser = create_argument_parser()
            args = parser.parse_args()
            logger.info("Running in normal mode")

        # Create configuration and validate
        with timing_context("Configuration setup", logger):
            cfg = Characterization_Config(args, test_mode=test_mode)
            data_extractor = InputDataExtractor(cfg)
            files_to_process = data_extractor.get_files_to_process()

        # Log system information
        system_info = [
            f"Host: {socket.gethostname()}",
            f"Number of CPUs: {os.cpu_count()}",
            f"Test mode: {test_mode}"
        ]
        
        for info in system_info:
            print(info)
            logger.info(info)

        # Display and log configuration
        cfg.display_config()
        cfg.log_config(logger)

        # Display and log files to process
        data_extractor.display_files_to_process()
        data_extractor.log_files_to_process(logger)

        # Run the main processing
        with timing_context("Complete EEG characterization", logger):
            run_eeg_characterization(cfg, files_to_process, logger)
        
        logger.info("All processing completed successfully")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
