# EEG HFO Characterization and Detection Tool Usage Guide

## Brief Description

`run_eeg_hfo_characterize_detect.py` is the main Python script to run the EEG characterization and the subsequent detection of of High-Frequency Oscillations (HFOs). The tool performs spectral analysis on EEG recordings to extract features and uses machine learning (XGBoost) to classify and detect HFOs. It supports various EEG file formats, montage configurations, and provides flexible analysis parameters for research and clinical applications. The tested montages are EDF (.edf), BrainVisin(.vhdr) and Persyst (laydat).

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Command Line Arguments](#command-line-arguments)
5. [Usage Examples](#usage-examples)
6. [Input/Output Structure](#inputoutput-structure)
7. [Configuration Options](#configuration-options)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Performance Considerations](#performance-considerations)
11. [Technical Details](#technical-details)

## Overview

The script performs the following main operations:
- Reads EEG data from various formats (EDF, BrainVision, Persyst)
- Applies spectral characterization using windowed analysis
- Extracts features for HFO detection
- Uses an XGBoost classifier to detect HFO events
- Generates output files with detection results and features

### Key Features
- **Multi-format support**: EDF, DAT, VHDR file formats
- **Flexible montage options**: Intracranial/scalp, bipolar/referential
- **Parallel processing**: Multi-core CPU utilization
- **Power line filtering**: Automatic detection of power line noise and notch filtering (50/60 Hz)
- **Logging**: Comprehensive logging with timestamps

## Prerequisites

- Python 3.8 or higher
- Required Python packages (see `pyproject.toml`)
- EEG data in supported formats
- Sufficient disk space for output files

## Installation

Ensure you're in the project directory and have the virtual environment activated:

```bash
cd /path/to/EEG_Characterizer_HFO
source .venv/bin/activate  # or activate your virtual environment
```

## Command Line Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--dataset_name` | string | Name identifier for the dataset |
| `--input_folder` | string | Path to directory containing EEG files |
| `--output_folder`| string | Path to output directory |
| `--montage_type` | string | Montage type: 'ib' (intracranial bipolar), 'ir' (intracranial referential), 'sb' (scalp bipolar), 'sr' (scalp referential) |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--rm_vchann` | string | "yes" | Remove Natus virtual channels ("yes"/"no") |
| `--eeg_format` | string | "edf" | EEG file format ("edf", "dat", "vhdr") |
| `--montage_channels` | string | "" | Comma-separated channels to detect. Must match montage_type (e.g., "F3-C3,C3-P3") |
| `--power_line_freq` | int | 60 | Power line frequency (0, 50, or 60 Hz) |
| `--start_sec` | float | 0 | Analysis start time in seconds |
| `--end_sec` | float | -1 | Analysis end time in seconds (-1 for full length) |
| `--wdw_step_s` | float | 0.1 | Window step size in seconds |
| `--force_characterization` | string | "no" | Force recalculation of features ("yes"/"no") |
| `--force_hfo_detection` | string | "yes" | Force HFO detection ("yes"/"no") |
| `--n_jobs` | int | -1 | Number of parallel jobs (-1 uses all CPU cores) |
| `--verbose` | string | "no" | Enable verbose output ("yes"/"no") |

## Usage Examples

### Basic Usage

```bash
python run_eeg_hfo_characterize_detect.py \
    --dataset_name "MyDataset" \
    --input_folder "/path/to/eeg/files" \
    --montage_type "sb"
```

### Advanced Usage with Custom Parameters

```bash
python run_eeg_hfo_characterize_detect.py \
    --dataset_name "Study_2024" \
    --input_folder "/data/eeg_recordings" \
    --output_folder "/results/hfo_analysis" \
    --eeg_format "edf" \
    --montage_type "ib" \
    --montage_channels "F3-C3,C3-P3,F4-C4,C4-P4" \
    --power_line_freq 50 \
    --start_sec 60 \
    --end_sec 300 \
    --wdw_step_s 0.05 \
    --force_characterization "yes" \
    --n_jobs 8 \
    --verbose "yes"
```

### Specific Channel Analysis

```bash
python run_eeg_hfo_characterize_detect.py \
    --dataset_name "Epilepsy_Patients" \
    --input_folder "/data/seizure_recordings" \
    --montage_type "ib" \
    --montage_channels "LA1-LA2,LH1-LH2,RA1-RA2" \
    --power_line_freq 60 \
    --verbose "yes"
```

## Configuration Options

### Montage Types
- **`ib`** - Intracranial Bipolar: For depth electrodes, paired channels
- **`ir`** - Intracranial Referential: Single-ended intracranial recordings
- **`sb`** - Scalp Bipolar: Surface electrode pairs
- **`sr`** - Scalp Referential: Single-ended scalp recordings

### Power Line Filtering
- **`0`** - Disable power line filtering
- **`50`** - European/Asian standard (50 Hz)
- **`60`** - North American standard (60 Hz)

### Window Analysis
- **Step size**: Configurable overlap between analysis windows
- **Start/End times**: Flexible time range selection

## Advanced Features

### Parallel Processing
- Uses joblib for CPU parallelization
- Select amount of CPU cores to use, or `-1` to use all available.

### Caching and Force Options
- **Feature caching**: Avoids recomputation of existing features
- **Force recalculation**: Override caching when needed
- **Selective forcing**: Separate control for characterization and detection

### Natus Virtual Channel Handling
- Automatic detection and removal of Natus virtual channels
- Configurable option for different EEG systems

### Test Mode
The script includes a built-in test mode that activates when:
- Running in a debugger environment
- No command line arguments provided
- Uses predefined test parameters

## Troubleshooting

### Common Issues

**1. Sampling Rate Error**
```
Error: Sampling Rate is XXX Hz, which is under 1000 Hz!
```
*Solution*: HFO analysis requires sampling rates > 1000 Hz. Use higher sampling rate recordings.

**2. No Files Found**
```
Warning: No edf files found in /path/to/folder
```
*Solution*: Check file format parameter matches actual files, verify folder path.

**3. Memory Issues**
```
Error: Out of memory during processing
```
*Solution*: Reduce `n_jobs` parameter or process fewer files simultaneously.

**4. Channel Configuration Error**
```
Error: Invalid montage type
```
*Solution*: Use valid montage types: 'ib', 'ir', 'sb', 'sr'.

### Debugging Tips
- Use `--verbose "yes"` for detailed output
- Check log files in the `logs/` directory
- Verify file permissions for input/output directories

## Performance Considerations

### Optimization Strategies
- **CPU cores**: Use `-1` for automatic detection, or specify optimal number
- **Window step size**: Larger steps = faster processing, less overlap
- **Time ranges**: Process specific time segments for faster analysis

---
