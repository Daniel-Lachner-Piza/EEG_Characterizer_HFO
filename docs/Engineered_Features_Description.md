# Engineered Features Description - EEG HFO Characterization Module

This document provides a comprehensive description of the engineered features calculated by the HFO (High-Frequency Oscillation) characterization module. The module performs temporal overlap analysis between ELPI (Event Localization and Parametrization Interface) events and HFO contours, transferring detailed spectral and temporal features from the best-matching HFO contours to the ELPI events.

## Overview

The characterization process involves:
1. **Spectral Analysis**: Time-frequency analysis using continuous wavelet transform (CWT)
2. **Contour Detection**: Computer vision-based detection of spectral contours in time-frequency representations
3. **Feature Extraction**: Calculation of morphological, spectral, and temporal features
4. **Temporal Matching**: Finding the best temporal overlap between ELPI events and HFO contours
5. **Feature Transfer**: Copying characterization features from matched contours to ELPI events

## Feature Categories

### 1. Morphological Features (Spectral Contour Shape)

#### `hvr` (Horizontal-to-Vertical Ratio)
- **Description**: Ratio of horizontal spread to vertical spread of the spectral contour
- **Calculation**: `100 * horizontal_spread / vertical_spread`
- **Units**: Percentage
- **Interpretation**: Higher values indicate longer duration relative to frequency bandwidth. Values >100 suggest temporally extended oscillations.

#### `circularity`
- **Description**: Measure of how circular the spectral contour is
- **Calculation**: `100 * (4π * area) / perimeter²`
- **Units**: Percentage
- **Interpretation**: Perfect circle = 100%. Lower values indicate more elongated or irregular shapes. Useful for distinguishing HFOs from artifacts.

#### `area`
- **Description**: Area of the spectral contour in the time-frequency plane
- **Units**: Pixels (scaled to time-frequency units)
- **Interpretation**: Larger areas suggest more prominent spectral energy concentration. Related to the extent of the oscillatory event.

#### `nr_oscillations`
- **Description**: Estimated number of oscillations within the event duration
- **Calculation**: `round(duration_ms / period_ms)` where `period_ms = 1000/frequency_centroid_Hz`
- **Units**: Count
- **Interpretation**: Indicates the rhythmicity of the oscillation. Clinical HFOs typically have ≥4 oscillations.

### 2. Bandpass Signal Features (Temporal Domain)

#### `bp_sig_ampl` (Bandpass Signal Amplitude)
- **Description**: Peak-to-peak amplitude of the bandpass filtered signal
- **Calculation**: `max(bp_signal) - min(bp_signal)`
- **Units**: Microvolts (μV)
- **Interpretation**: Measure of oscillation strength. Higher amplitudes may indicate pathological activity.

#### `bp_sig_pow` (Bandpass Signal Power)
- **Description**: Mean squared amplitude of the bandpass filtered signal
- **Calculation**: `mean(bp_signal²)`
- **Units**: μV²
- **Interpretation**: Energy content of the oscillation. More robust to outliers than amplitude.

#### `bp_sig_std` (Bandpass Signal Standard Deviation)
- **Description**: Standard deviation of the bandpass filtered signal
- **Calculation**: `std(bp_signal)`
- **Units**: μV
- **Interpretation**: Measure of signal variability. Consistent with power but less affected by DC offset.

### 3. Background Signal Features

#### `bkgrnd_sig_pow` (Background Signal Power)
- **Description**: Power of the background signal adjacent to the event
- **Calculation**: `mean(background_bp_signal²)`
- **Units**: μV²
- **Interpretation**: Baseline activity level. Used for signal-to-background ratio calculations.

#### `bkgrnd_sig_std` (Background Signal Standard Deviation)
- **Description**: Standard deviation of the background signal
- **Calculation**: `std(background_bp_signal)`
- **Units**: μV
- **Interpretation**: Background noise level. Lower values indicate cleaner baseline conditions.

### 4. Sinusoidal Correlation Features

#### `max_hfo_sine_corr` (Maximum HFO Sinusoidal Correlation)
- **Description**: Maximum correlation between the bandpass signal and sinusoidal templates
- **Calculation**: Max correlation across frequencies and phase shifts: `max(corrcoef(sine_template, bp_signal))`
- **Range**: [0, 1]
- **Interpretation**: Higher values (>0.7) suggest more sinusoidal, rhythmic oscillations typical of genuine HFOs.

### 5. Peak Analysis Features

#### `all_relevant_peaks_nr` (All Relevant Peaks Number)
- **Description**: Count of all significant peaks in the bandpass signal
- **Calculation**: Number of peaks above prominence threshold (10% of prominent peak threshold)
- **Units**: Count
- **Interpretation**: Indicates oscillatory complexity. Too few peaks suggest non-oscillatory events.

#### `all_relevant_peaks_avg_freq` (All Relevant Peaks Average Frequency)
- **Description**: Average frequency based on intervals between all relevant peaks
- **Calculation**: `mean(sampling_rate / diff(peak_locations))`
- **Units**: Hz
- **Interpretation**: Dominant frequency of the oscillation. Should align with the frequency band of interest.

#### `prom_peaks_nr` (Prominent Peaks Number)
- **Description**: Count of prominent peaks (high amplitude, high prominence)
- **Calculation**: Number of peaks above prominence threshold and above average amplitude
- **Units**: Count
- **Interpretation**: Indicates the number of major oscillatory cycles. Clinical relevance for HFO classification.

#### `prom_peaks_avg_freq` (Prominent Peaks Average Frequency)
- **Description**: Average frequency based on intervals between prominent peaks only
- **Calculation**: `mean(sampling_rate / diff(prominent_peak_locations))`
- **Units**: Hz
- **Interpretation**: More stable frequency estimate focusing on major oscillations.

### 6. Signal-to-Background Ratio Features

#### `EventBkgrndRatio_Power` (Event-to-Background Power Ratio)
- **Description**: Ratio of event power to background power
- **Calculation**: `bp_sig_pow / bkgrnd_sig_pow`
- **Units**: Ratio (dimensionless)
- **Interpretation**: Values >3-5 typically indicate significant events above baseline. Critical for HFO detection algorithms.

### 7. Metadata Features

#### `overlap_ratio` (Temporal Overlap Ratio)
- **Description**: Temporal overlap between ELPI event and best-matching HFO contour
- **Calculation**: `overlap_duration / min(elpi_duration, contour_duration)`
- **Range**: [0, 1]
- **Interpretation**: Values >0.5 indicate good temporal alignment. Higher values suggest better feature reliability.

#### `best_contour_idx` (Best Contour Index)
- **Description**: Index of the HFO contour with highest temporal overlap
- **Units**: Integer index
- **Interpretation**: Traceability information for debugging and validation purposes.

## Feature Quality and Reliability

### High-Reliability Features
- **`overlap_ratio > 0.7`**: Excellent temporal matching
- **`max_hfo_sine_corr > 0.6`**: Strong oscillatory pattern
- **`EventBkgrndRatio_Power > 3.0`**: Clear distinction from background
- **`nr_oscillations ≥ 4`**: Sufficient rhythmicity for HFO classification

### Moderate-Reliability Features
- **`0.3 ≤ overlap_ratio ≤ 0.7`**: Fair temporal matching
- **`0.3 ≤ max_hfo_sine_corr ≤ 0.6`**: Moderate oscillatory pattern
- **`1.5 ≤ EventBkgrndRatio_Power ≤ 3.0`**: Moderate signal strength

### Low-Reliability Features
- **`overlap_ratio < 0.3`**: Poor temporal matching - features may be unreliable
- **`max_hfo_sine_corr < 0.3`**: Non-oscillatory pattern
- **`EventBkgrndRatio_Power < 1.5`**: Low signal-to-noise ratio

## Clinical and Research Applications

### HFO Classification
Primary features for machine learning models:
- `max_hfo_sine_corr`
- `EventBkgrndRatio_Power`
- `nr_oscillations`
- `prom_peaks_avg_freq`
- `circularity`

### Quality Assessment
Features for event validation:
- `overlap_ratio`
- `bp_sig_std` vs `bkgrnd_sig_std`
- `all_relevant_peaks_nr`

### Morphological Analysis
Features for studying oscillation patterns:
- `hvr`
- `area`
- `prom_peaks_nr`
- Peak frequency distributions

## Implementation Notes

- **Bandpass Filtering**: Applied in the HFO frequency range (typically 80-500 Hz)
- **Background Estimation**: Calculated from signal segments adjacent to detected events
- **Peak Detection**: Uses prominence-based thresholding to identify relevant oscillatory peaks
- **Temporal Alignment**: Uses millisecond precision for event-contour matching
- **Missing Values**: Features are set to `NaN` when no suitable HFO contour match is found

## Usage in Analysis Pipelines

1. **Preprocessing**: Ensure proper bandpass filtering and artifact removal
2. **Feature Extraction**: Run characterization module on detected events
3. **Quality Control**: Filter based on `overlap_ratio` and signal-to-background ratios
4. **Analysis**: Use appropriate feature subsets for specific research questions
5. **Validation**: Cross-reference with manual annotations when available

This feature set provides a comprehensive characterization of HFO events suitable for both automated detection algorithms and detailed morphological analysis in clinical and research settings.
