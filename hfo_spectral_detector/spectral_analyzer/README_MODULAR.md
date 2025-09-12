# HFO Spectral Analyzer - Modular Architecture

## Overview

The HFO (High Frequency Oscillation) Spectral Analyzer has been significantly refactored to improve modularity, maintainability, and extensibility. This document outlines the new modular architecture and improvements made to the original `characterize_events.py` module.

## Architecture Improvements

### 🔧 **Modular Design**
- **Separation of Concerns**: Each module handles a specific aspect of HFO analysis
- **Reusable Components**: Classes and functions can be used independently
- **Clear Interfaces**: Well-defined APIs between modules
- **Easier Testing**: Individual components can be unit tested

### 🎛️ **Configuration Management**
- **Centralized Configuration**: All parameters in one place
- **Multiple Configurations**: Default, fast, high-quality, memory-efficient presets
- **Parameter Validation**: Automatic validation of configuration parameters
- **Type Safety**: Using dataclasses for configuration

### 🚀 **Performance Improvements**
- **Memory Management**: Better handling of large datasets
- **Error Recovery**: Graceful fallback mechanisms
- **Parallel Processing**: Improved job distribution
- **Resource Cleanup**: Proper garbage collection

## Module Structure

```
hfo_spectral_detector/spectral_analyzer/
├── __init__.py                 # Main module interface
├── characterize_events.py      # Refactored main pipeline
├── signal_processing.py        # Signal filtering and preprocessing
├── feature_extraction.py       # Feature computation and management
├── hfo_validation.py          # Event validation and classification
├── visualization.py           # Plotting and visualization
└── config.py                  # Configuration management
```

## New Modules

### 1. `signal_processing.py`
**Purpose**: Handle all signal preprocessing operations

**Key Components**:
- `SignalProcessor`: Main class for signal operations
- `create_spectrogram_image()`: Convert wavelets to CV image format

**Features**:
- Power line noise detection and filtering
- Bandpass filtering with configurable parameters
- Wavelet transform computation
- Signal normalization utilities

**Example Usage**:
```python
from hfo_spectral_detector.spectral_analyzer.signal_processing import SignalProcessor

processor = SignalProcessor(fs=2000)
processed_signal, bp_signal, notch_applied = processor.preprocess_signal(
    raw_signal, power_line_freq=60
)
freqs, wavelet_coeffs = processor.compute_wavelet_transform(bp_signal)
```

### 2. `feature_extraction.py`
**Purpose**: Extract and manage features from HFO events

**Key Components**:
- `FeatureExtractor`: Extract various signal features
- `ContourFeatureProcessor`: Process contour-based features
- `safe_memory_intensive_concatenation()`: Handle large datasets

**Features**:
- Bandpass signal feature extraction
- Background signal analysis
- Relative feature calculation
- Overlap analysis between events
- Memory-safe data concatenation

**Example Usage**:
```python
from hfo_spectral_detector.spectral_analyzer.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(fs=2000)
bp_features = extractor.extract_bandpass_features(signal, hfo_freqs)
bg_features = extractor.extract_background_features(background_signal)
```

### 3. `hfo_validation.py`
**Purpose**: Validate and classify HFO events

**Key Components**:
- `HFOValidationCriteria`: Configuration for validation parameters
- `HFOValidator`: Individual validation checks
- `HFOClassifier`: Complete event classification

**Features**:
- Configurable validation criteria
- Multiple validation methods
- Comprehensive and high-confidence classification
- Batch processing capabilities

**Example Usage**:
```python
from hfo_spectral_detector.spectral_analyzer.hfo_validation import HFOClassifier

classifier = HFOClassifier()
classification = classifier.classify_event(
    bp_features, spectral_features, inverted_features, hfo_freqs, fs
)
```

### 4. `visualization.py`
**Purpose**: Create plots and visualizations

**Key Components**:
- `HFOVisualizer`: Main visualization class
- `SignalPlotter`: Utility plotting functions
- `setup_matplotlib_backend()`: Configure matplotlib

**Features**:
- Comprehensive HFO analysis plots
- Signal comparison plots
- Spectrogram visualization
- Automatic figure organization
- Memory-efficient plotting

**Example Usage**:
```python
from hfo_spectral_detector.spectral_analyzer.visualization import HFOVisualizer

visualizer = HFOVisualizer()
fig_path = visualizer.plot_hfo_analysis(
    pat_name, channel, time, signals, contours_df, fs, output_path
)
```

### 5. `config.py`
**Purpose**: Centralized configuration management

**Key Components**:
- `PipelineConfig`: Complete pipeline configuration
- Multiple sub-configurations for different aspects
- Default values and constants
- Configuration validation

**Features**:
- Type-safe configuration using dataclasses
- Pre-defined configuration presets
- Parameter validation
- Easy customization

**Example Usage**:
```python
from hfo_spectral_detector.spectral_analyzer.config import PipelineConfig

# Use preset configurations
config = PipelineConfig.create_high_quality()
config = PipelineConfig.create_fast_processing()
config = PipelineConfig.create_memory_efficient()

# Customize configuration
config = PipelineConfig()
config.signal_processing.bandpass_low_freq = 100.0
config.hfo_validation.min_sine_correlation = 0.85
```

## Refactored Main Functions

### `characterize_events()`
**Improvements**:
- Uses modular components instead of inline code
- Better error handling and logging
- Configurable via `PipelineConfig`
- Cleaner parameter interface
- Improved memory management

### `channel_specific_characterization()`
**Improvements**:
- Modular signal processing pipeline
- Configurable feature extraction
- Better error recovery
- Cleaner code structure

### `hfo_spectro_bp_wdw_analysis()`
**Improvements**:
- Uses modular feature extraction
- Configurable validation criteria
- Optional visualization
- Better data management

## Configuration Presets

### Default Configuration
Balanced settings for general use:
```python
config = PipelineConfig.create_default()
```

### Fast Processing
Optimized for speed:
```python
config = PipelineConfig.create_fast_processing()
# - Disables detailed visualization
# - Reduces logging overhead
# - Skips intermediate file saving
```

### High Quality
Optimized for accuracy:
```python
config = PipelineConfig.create_high_quality()
# - Increased wavelet cycles
# - Stricter validation criteria
# - Forces recalculation
# - Saves all figures
```

### Memory Efficient
Optimized for low memory usage:
```python
config = PipelineConfig.create_memory_efficient()
# - Enables garbage collection
# - Single-threaded processing
# - Memory-intensive fallbacks
# - Minimal visualization
```

## Migration Guide

### For Existing Code
The main interface remains compatible:

```python
# Old way (still works)
result = characterize_events(pat_name, eeg_reader, channels, windows, output_path)

# New way with configuration
config = PipelineConfig.create_high_quality()
result = characterize_events(pat_name, eeg_reader, channels, windows, output_path, config=config)
```

### For Advanced Users
Access individual components:

```python
from hfo_spectral_detector.spectral_analyzer import (
    SignalProcessor, FeatureExtractor, HFOClassifier
)

# Use components individually
processor = SignalProcessor(fs)
extractor = FeatureExtractor(fs)
classifier = HFOClassifier()
```

## Benefits

### 🎯 **Improved Maintainability**
- Clear separation of concerns
- Easier to understand and modify
- Better code organization
- Reduced complexity

### 🔬 **Enhanced Testability**
- Individual components can be unit tested
- Mocking and stubbing made easier
- Better error isolation
- Validation of individual features

### ⚡ **Better Performance**
- Improved memory management
- Better error recovery
- Configurable optimization levels
- Reduced redundant computations

### 🛠️ **Greater Flexibility**
- Easy to customize behavior
- Multiple configuration presets
- Modular replacement of components
- Extensible architecture

### 📊 **Improved Monitoring**
- Better logging and debugging
- Performance metrics
- Memory usage tracking
- Progress reporting

## Future Enhancements

The modular architecture enables easy addition of:

1. **New Signal Processing Methods**
   - Additional filtering techniques
   - Alternative wavelet transforms
   - Preprocessing algorithms

2. **Enhanced Feature Extraction**
   - Machine learning features
   - Time-frequency features
   - Connectivity measures

3. **Advanced Validation**
   - ML-based classification
   - Ensemble methods
   - Cross-validation

4. **Extended Visualization**
   - Interactive plots
   - 3D visualizations
   - Real-time displays

5. **Performance Optimization**
   - GPU acceleration
   - Distributed processing
   - Streaming analysis

## Conclusion

The modular refactoring of the HFO Spectral Analyzer provides a solid foundation for future development while maintaining backward compatibility. The new architecture improves code quality, maintainability, and extensibility while providing better performance and configurability.
