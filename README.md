
# EEG Characterizer HFO

A spectral-based High-Frequency Oscillation (HFO) detection and characterization tool for EEG data analysis.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Dependencies](#dependencies)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Command Line Arguments](#command-line-arguments)
  - [Example Usage](#example-usage)
- [Development](#development)
  - [MLflow Integration](#mlflow-integration)
- [Data Management](#data-management)
  - [Data Transfer to HPC](#data-transfer-to-hpc)
  - [File Management Commands](#file-management-commands)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This tool provides automated detection and characterization of High-Frequency Oscillations (HFOs) in EEG data using spectral analysis. It supports various EEG formats and montage configurations for comprehensive neurophysiological analysis.

## Features

- **Multi-format support**: EDF, BrainVision, and other common EEG formats
- **Flexible montage options**: Scalp bipolar (sb), scalp referential (sr), intracranial bipolar (ib), intracranial referential (ir)
- **Spectral analysis**: Advanced HFO detection using time-frequency analysis
- **Machine learning classification**: XGBoost-based HFO classification
- **Power line noise filtering**: Configurable notch filtering for different regions
- **Batch processing**: Process multiple files and datasets efficiently
- **MLflow integration**: Experiment tracking and model management

## Installation

### Environment Setup

This project uses UV for Python environment management. Follow these steps to set up the environment:

#### 1. Install uv from Astral
```bash
mkdir ~/tmp
cd ~/tmp
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~
```

#### 2. Clone the Repository
```bash
mkdir ~/Projects
cd ~/Projects
git clone https://github.com/Daniel-Lachner-Piza/EEG_Characterizer_HFO.git
cd EEG_Characterizer_HFO
```

#### 3. Install Dependencies
```bash
uv sync
```

#### 4. Activate the Environment
```bash
source .venv/bin/activate
```


## Usage

### Quick Start

Activate the environment and run the HFO detector:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with basic parameters
python run_eeg_characterization.py --name MyAnalysis --inpath /path/to/eeg/data --outpath /path/to/output --format edf --montage sb --plf 60
```

The detector will batch process all the files in the --inpath directory that have the specified --format extension.

### Command Line Arguments

- `--name`: Analysis name/identifier
- `--inpath`: Path to input EEG data directory
- `--outpath`: Path to output directory
- `--format`: EEG file format (edf, bv, etc.)
- `--montage`: Montage type (sb, sr, ib, ir)
- `--plf`: Power line frequency for notch filtering (50 or 60 Hz, anything else will deactivate notch filters)

### Example Usage

```bash
name=Test_DLP
inpath=/home/dlp/Documents/Development/Data/Physio_EEG_Data/
outpath=/home/dlp/Documents/Development/Data/Test-DLP-Output/
fmt=edf
montage=sb
plf=60

python run_eeg_characterization.py \
  --name $name \
  --inpath $inpath \
  --outpath $outpath \
  --format $fmt \
  --montage $montage \
  --plf $plf
```

## Data Management

### Data Transfer to HPC

For transferring data to High-Performance Computing (HPC) systems using rsync:

#### Prerequisites
- **Windows**: Open a WSL (Windows Subsystem for Linux) terminal
- **Linux/MacOS**: rsync should be available from your terminal

#### Copy Entire Folder

```bash
source_path="/mnt/c/Users/HFO/Documents/Postdoc_Calgary/Research/Scalp_HFO_GoldStandard/"
destination_path="daniel.lachnerpiza@arc.ucalgary.ca:/work/jacobs_lab/EEG_Data/Scalp_HFO_GoldStandard/"

echo "----------------------------------------------- Data Transfer Info -----------------------------------------------"
echo "Source Path: $source_path"
echo "Destination Path: $destination_path"
echo "-------------------------------------------------------------------------------------------------------------------"

rsync -a --partial --progress "$source_path" "$destination_path"
```

#### Copy Specific File Types

```bash
groups_ls=(
    "HFOHealthy1monto2yrs" 
    "HFOHealthy3to5yrs" 
    "HFOHealthy6to10yrs" 
    "HFOHealthy11to13yrs" 
    "HFOHealthy14to17yrs"
)

for group in ${groups_ls[@]}; do
    echo "Processing group: $group"
    source_path="/mnt/c/Users/HFO/Documents/Postdoc_Calgary/Research/Tatsuya/PhysioEEGs/Anonymized_EDFs/${group}"
    destination_path="daniel.lachnerpiza@arc.ucalgary.ca:/work/jacobs_lab/EEG_Data/AnonymPhysioEEGs/${group}"
    file_type="*/*.edf"
    
    echo "----------------------------------------------- Data Transfer Info -----------------------------------------------"
    echo "Source Path: $source_path"
    echo "Destination Path: $destination_path"
    echo "File Type: $file_type"
    echo "-------------------------------------------------------------------------------------------------------------------"
    
    rsync --relative --partial --progress ${source_path}${file_type} $destination_path
done
```

### File Management Commands

#### Check folder size and file count on HPC:

```bash
echo "Files: $(find . -type f | wc -l)"; echo "Total size: $(du -sh . | cut -f1)"
```

## Project Structure

```
EEG_Characterizer_HFO/
├── hfo_spectral_detector/
│   ├── dsp/                    # Digital signal processing modules
│   ├── eeg_io/                 # EEG input/output handling
│   ├── elpi/                   # ELPI interface for annotations
│   ├── file_io/                # File handling utilities
│   ├── prediction/             # HFO classification models
│   ├── read_setup_eeg/         # EEG setup and montage creation
│   ├── spectral_analyzer/      # Spectral analysis components
│   └── studies_info/           # Study metadata management
├── hpc_jobs/                   # HPC job submission scripts
├── main.py                     # Main entry point
├── run_eeg_characterization.py # Command-line interface
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```
