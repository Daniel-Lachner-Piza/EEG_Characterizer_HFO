"""
HFO Spectral Analyzer Module

This module provides modular components for High Frequency Oscillation (HFO) 
characterization and analysis.

Main Components:
- signal_processing: Signal filtering, preprocessing, and wavelet transforms
- feature_extraction: Feature computation and data management
- hfo_validation: HFO event validation and classification
- visualization: Plotting and visualization utilities
- config: Configuration management and default parameters
- characterize_events: Main analysis pipeline (refactored)

Usage:
    from hfo_spectral_detector.spectral_analyzer import characterize_events
    from hfo_spectral_detector.spectral_analyzer.config import PipelineConfig
    
    # Create custom configuration
    config = PipelineConfig.create_high_quality()
    
    # Run analysis
    result_path = characterize_events(
        pat_name="Patient001",
        eeg_reader=eeg_data,
        mtgs_to_detect=channels,
        an_wdws_dict=analysis_windows,
        out_path=output_dir,
        config=config
    )
"""

# Import main functions for easy access
from .characterize_events import characterize_events, save_all_channel_events
from .config import PipelineConfig, DefaultValues, AnalysisConstants

# Import utility classes for advanced usage
from .signal_processing import SignalProcessor
from .feature_extraction import FeatureExtractor, ContourFeatureProcessor
from .hfo_validation import HFOValidator, HFOClassifier
from .visualization import HFOVisualizer

__version__ = "2.0.0"
__author__ = "HFO Analysis Team"

__all__ = [
    # Main functions
    "characterize_events",
    "save_all_channel_events",
    
    # Configuration
    "PipelineConfig", 
    "DefaultValues",
    "AnalysisConstants",
    
    # Processing classes
    "SignalProcessor",
    "FeatureExtractor", 
    "ContourFeatureProcessor",
    "HFOValidator",
    "HFOClassifier", 
    "HFOVisualizer"
]
