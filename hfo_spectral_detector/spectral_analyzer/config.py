"""
Configuration constants and parameters for HFO characterization.

This module contains all configuration parameters, constants, and default
values used throughout the HFO analysis pipeline.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
from pathlib import Path


@dataclass
class SignalProcessingConfig:
    """Configuration for signal processing parameters."""
    
    # Filtering parameters
    notch_width: float = 10.0
    notch_filter_ntaps: int = 513
    bandpass_ntaps: int = 256
    bandpass_low_freq: float = 80.0
    bandpass_high_freq_ratio: float = 1.0 / 3.0  # fs/3
    
    # Wavelet parameters
    wavelet_freq_min: int = 80
    wavelet_freq_max: int = 500
    wavelet_freq_step: int = 5
    wavelet_nr_cycles: int = 6
    
    # Power line frequencies
    valid_power_line_freqs: List[float] = None
    
    def __post_init__(self):
        if self.valid_power_line_freqs is None:
            self.valid_power_line_freqs = [50.0, 60.0]


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction parameters."""
    
    # Peak detection parameters
    min_peak_prominence: float = 0.1
    min_peak_distance: int = 1
    
    # Hjorth parameters calculation
    use_hjorth_parameters: bool = True
    
    # Background analysis
    background_analysis_enabled: bool = True
    
    # Feature validation
    remove_nan_features: bool = True
    validate_feature_ranges: bool = True


@dataclass
class HFOValidationConfig:
    """Configuration for HFO validation criteria."""
    
    # Sinusoidal correlation thresholds
    min_sine_correlation: float = 0.75
    high_sine_correlation: float = 0.80
    excellent_sine_correlation: float = 0.99
    
    # Peak-related thresholds
    min_prominent_peaks: int = 4
    min_relevant_peaks: int = 4
    max_freq_deviation_ratio: float = 0.2
    max_freq_stddev: float = 15.0
    min_amplitude_stability: float = 0.2
    high_amplitude_stability: float = 0.70
    min_prominence_stability: float = 0.4
    
    # Spectral criteria
    min_hvr: float = 10.0  # Height-to-width ratio
    min_circularity: float = 30.0
    
    # Peak ratio criteria
    max_inverted_peak_diff: int = 2
    max_relevant_to_prominent_ratio: float = 0.5
    
    # Advanced validation criteria
    min_duration_ms: float = 6.0  # Minimum HFO duration
    max_duration_ms: float = 100.0  # Maximum HFO duration
    min_frequency_hz: float = 80.0  # Minimum HFO frequency
    max_frequency_hz: float = 500.0  # Maximum HFO frequency


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    
    # Figure parameters
    default_screen_size: Tuple[float, float] = (20.0, 11.0)
    signal_linewidth: float = 1.0
    
    # Image processing
    spectrogram_resize_dims: Tuple[int, int] = (1000, 250)
    colormap: str = 'jet'
    
    # Output directories
    create_tp_fp_fn_dir: bool = True
    create_tn_dir: bool = False
    save_all_figures: bool = False
    
    # Plot elements
    show_peak_annotations: bool = True
    show_threshold_lines: bool = True
    show_visual_validation_markers: bool = True
    include_feature_legend: bool = True


@dataclass
class ProcessingConfig:
    """Configuration for processing and parallelization."""
    
    # Parallelization
    default_n_jobs: int = -1  # Use all available CPUs
    max_n_jobs: Optional[int] = None
    
    # Memory management
    enable_garbage_collection: bool = True
    memory_intensive_fallback: bool = True
    
    # File handling
    force_recalculation: bool = False
    save_intermediate_results: bool = True
    compression_level: int = 1  # For parquet files
    
    # Logging
    log_processing_time: bool = True
    log_memory_usage: bool = False
    verbose_output: bool = False


@dataclass
class FileIOConfig:
    """Configuration for file input/output operations."""
    
    # File formats
    output_format: str = 'parquet'  # 'parquet', 'csv', 'hdf5'
    compression: str = 'snappy'  # For parquet files
    
    # File naming
    objects_filename_template: str = "{patient}_{channel}_objects.parquet"
    all_channels_filename: str = "All_Ch_Objects.parquet"
    
    # Directory structure
    create_patient_subdirs: bool = True
    create_channel_subdirs: bool = True
    
    # Error handling
    skip_corrupted_files: bool = True
    create_empty_file_on_no_events: bool = True
    empty_file_content: str = "NoEvents"


class DefaultValues:
    """Default values and constants used throughout the system."""
    
    # Analysis window defaults
    DEFAULT_ANALYSIS_WINDOW_MS = 1000.0
    
    # Frequency defaults
    DEFAULT_SAMPLING_FREQUENCY = 2000.0
    DEFAULT_POWER_LINE_FREQUENCY = 60.0
    
    # Processing defaults
    DEFAULT_CPU_COUNT_MULTIPLIER = 1.0
    
    # Validation defaults
    DEFAULT_VISUAL_VALIDATION = False
    
    # Feature defaults
    DEFAULT_FEATURE_VALUE = 0.0
    DEFAULT_BOOLEAN_VALUE = False
    
    # Path defaults
    DEFAULT_OUTPUT_SUBDIR = "HFO_Analysis"


class AnalysisConstants:
    """Mathematical and analysis constants."""
    
    # Mathematical constants
    EPS = 1e-12  # Small value to avoid division by zero
    
    # Signal processing constants
    NYQUIST_FACTOR = 0.5
    ANTI_ALIASING_FACTOR = 3.0  # For maximum frequency = fs/3
    
    # Feature extraction constants
    MIN_SIGNAL_LENGTH = 10  # Minimum samples for analysis
    MAX_SIGNAL_LENGTH = 1000000  # Maximum samples for analysis
    
    # Validation constants
    MIN_HFO_CYCLES = 4  # Minimum number of cycles for valid HFO
    MAX_OVERLAP_RATIO = 0.8  # Maximum overlap between events


@dataclass
class PipelineConfig:
    """Complete pipeline configuration combining all sub-configurations."""
    
    signal_processing: SignalProcessingConfig = None
    feature_extraction: FeatureExtractionConfig = None
    hfo_validation: HFOValidationConfig = None
    visualization: VisualizationConfig = None
    processing: ProcessingConfig = None
    file_io: FileIOConfig = None
    
    def __post_init__(self):
        """Initialize sub-configurations with defaults if not provided."""
        if self.signal_processing is None:
            self.signal_processing = SignalProcessingConfig()
        if self.feature_extraction is None:
            self.feature_extraction = FeatureExtractionConfig()
        if self.hfo_validation is None:
            self.hfo_validation = HFOValidationConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.file_io is None:
            self.file_io = FileIOConfig()
    
    @classmethod
    def create_default(cls) -> 'PipelineConfig':
        """Create a default pipeline configuration."""
        return cls()
    
    @classmethod
    def create_fast_processing(cls) -> 'PipelineConfig':
        """Create configuration optimized for fast processing."""
        config = cls()
        config.visualization.save_all_figures = False
        config.visualization.include_feature_legend = False
        config.processing.log_processing_time = False
        config.processing.save_intermediate_results = False
        return config
    
    @classmethod
    def create_high_quality(cls) -> 'PipelineConfig':
        """Create configuration optimized for high-quality analysis."""
        config = cls()
        config.signal_processing.wavelet_nr_cycles = 8
        config.hfo_validation.high_sine_correlation = 0.85
        config.hfo_validation.high_amplitude_stability = 0.80
        config.visualization.save_all_figures = True
        config.processing.force_recalculation = True
        return config
    
    @classmethod
    def create_memory_efficient(cls) -> 'PipelineConfig':
        """Create configuration optimized for memory efficiency."""
        config = cls()
        config.processing.enable_garbage_collection = True
        config.processing.memory_intensive_fallback = True
        config.visualization.save_all_figures = False
        config.processing.max_n_jobs = 1  # Single-threaded to save memory
        return config


def get_feature_column_names() -> List[str]:
    """Get list of all feature column names."""
    return [
        # Validation columns
        'bp_ok', 'visual_valid',
        
        # Bandpass signal features
        'bp_sig_ampl', 'bp_sig_avg_ampl', 'bp_sig_std', 'bp_sig_pow',
        'bp_sig_activity', 'bp_sig_avg_mobility', 'bp_sig_complexity',
        
        # Background signal features
        'bkgrnd_sig_ampl', 'bkgrnd_sig_avg_ampl', 'bkgrnd_sig_std', 'bkgrnd_sig_pow',
        'bkgrnd_sig_activity', 'bkgrnd_sig_avg_mobility', 'bkgrnd_sig_complexity',
        
        # HFO-specific features
        'max_hfo_sine_corr', 'all_relevant_peaks_nr', 'all_relevant_peaks_avg_freq',
        'all_relevant_peaks_freq_stddev', 'all_relevant_peaks_amplitude_stability',
        'all_relevant_peaks_prominence_stability', 'prom_peaks_nr', 'prom_peaks_avg_freq',
        'prom_peaks_freqs_stddev', 'prom_peaks_avg_amplitude_stability',
        'prom_peaks_prominence_stability',
        
        # Inverted signal features
        'inverted_max_hfo_sine_corr', 'inverted_all_relevant_peaks_amplitude_stability',
        'inverted_all_relevant_peaks_prominence_stability', 'inverted_prom_peaks_nr',
        'inverted_prom_peaks_avg_freq', 'inverted_prom_peaks_freqs_stddev',
        'inverted_prom_peaks_avg_amplitude_stability', 'inverted_prom_peaks_prominence_stability',
        
        # Time-frequency features
        'TF_Complexity', 'NrSpectrumPeaks', 'SumFreqPeakWidths', 'NI',
        
        # Relative features
        'EventBkgrndRatio_Power', 'EventBkgrndRatio_StdDev', 'EventBkgrndRatio_Activity',
        'EventBkgrndRatio_Mobility', 'EventBkgrndRatio_Complexity',
        
        # Overlap features
        'nr_overlapping_objs'
    ]


def get_metadata_column_names() -> List[str]:
    """Get list of metadata column names."""
    return [
        'Patient', 'channel', 'an_start_ms', 'notch_filtered',
        'start_ms', 'end_ms', 'center_ms', 'dur_ms'
    ]


def validate_config(config: PipelineConfig) -> List[str]:
    """
    Validate pipeline configuration and return list of warnings/errors.
    
    Args:
        config: Pipeline configuration to validate
        
    Returns:
        List of validation messages
    """
    warnings = []
    
    # Validate signal processing config
    if config.signal_processing.bandpass_low_freq >= config.signal_processing.wavelet_freq_min:
        warnings.append("Bandpass low frequency should be less than wavelet minimum frequency")
    
    # Validate HFO validation config
    if config.hfo_validation.min_sine_correlation > config.hfo_validation.high_sine_correlation:
        warnings.append("Min sine correlation should be less than high sine correlation")
    
    # Validate processing config
    if config.processing.max_n_jobs is not None and config.processing.max_n_jobs < 1:
        warnings.append("max_n_jobs should be at least 1 or None")
    
    return warnings
