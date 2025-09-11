# PowerShell script to activate uv virtual environment and run HFO detection
# Author: GitHub Copilot
# Description: This script activates the uv virtual environment and runs the HFO detection script with predefined parameters

# Exit on any error
$ErrorActionPreference = "Stop"

# Configuration parameters - MODIFY THESE ACCORDING TO YOUR NEEDS
$DATASET_NAME = "test_dataset"
$INPUT_FOLDER = "C:\Users\dalap\Development\EEG_Data\"
$OUTPUT_FOLDER = "C:\Users\dalap\Development\Output_HFO\"
$EEG_FORMAT = "edf"
$MONTAGE_TYPE = "sb"  # Options: ib, ir, sb, sr
$MONTAGE_CHANNELS = ""  # Leave empty to detect all channels, or specify like "F3-C3,C3-P3,F4-C4,C4-P4"
$RM_VCHANN = "yes"  # Remove Natus virtual channels: yes/no
$POWER_LINE_FREQ = 60  # Power line frequency: 50, 60, or 0 to turn off
$START_SEC = 0  # Start analysis from specific second
$END_SEC = 60  # End analysis at specific second (-1 for full length)
$WDW_STEP_S = 0.1  # Window step size in seconds
$FORCE_CHARACTERIZATION = "no"  # Force recalculation of features: yes/no
$FORCE_HFO_DETECTION = "no"  # Force HFO detection: yes/no
$N_JOBS = -1  # Number of parallel jobs (-1 uses all CPU cores)
$VERBOSE = "yes"  # Enable verbose output: yes/no

Write-Host "=== HFO Detection Script ===" -ForegroundColor Green
Write-Host "Activating uv virtual environment..." -ForegroundColor Yellow

# Get the directory where this script is located and navigate to project root
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR
Set-Location $PROJECT_ROOT

# Check if uv is available
try {
    uv --version | Out-Null
    Write-Host "uv is available" -ForegroundColor Green
} catch {
    Write-Error "uv is not installed or not in PATH. Please install uv first."
    exit 1
}

# Check if virtual environment exists in project root
if (-not (Test-Path ".venv")) {
    Write-Host "Virtual environment not found in project root." -ForegroundColor Yellow
}

Write-Host "Running HFO detection with the following parameters:" -ForegroundColor Cyan
Write-Host "  Dataset Name: $DATASET_NAME"
Write-Host "  Input Folder: $INPUT_FOLDER"
Write-Host "  Output Folder: $OUTPUT_FOLDER"
Write-Host "  EEG Format: $EEG_FORMAT"
Write-Host "  Montage Type: $MONTAGE_TYPE"
Write-Host "  Montage Channels: $MONTAGE_CHANNELS"
Write-Host "  Remove Virtual Channels: $RM_VCHANN"
Write-Host "  Power Line Frequency: $POWER_LINE_FREQ"
Write-Host "  Start Second: $START_SEC"
Write-Host "  End Second: $END_SEC"
Write-Host "  Window Step Size: $WDW_STEP_S"
Write-Host "  Force Characterization: $FORCE_CHARACTERIZATION"
Write-Host "  Force HFO Detection: $FORCE_HFO_DETECTION"
Write-Host "  Number of Jobs: $N_JOBS"
Write-Host "  Verbose: $VERBOSE"
Write-Host ""

# Validate required directories exist
if (-not (Test-Path $INPUT_FOLDER)) {
    Write-Error "Input folder does not exist: $INPUT_FOLDER"
    exit 1
}

# Create output folder if it doesn't exist
if (-not (Test-Path $OUTPUT_FOLDER)) {
    Write-Host "Creating output folder: $OUTPUT_FOLDER" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $OUTPUT_FOLDER -Force | Out-Null
}

# Build the command arguments
$python_args = @(
    "run_eeg_hfo_characterize_detect.py",
    "--dataset_name", $DATASET_NAME,
    "--input_folder", $INPUT_FOLDER,
    "--output_folder", $OUTPUT_FOLDER,
    "--eeg_format", $EEG_FORMAT,
    "--montage_type", $MONTAGE_TYPE,
    "--rm_vchann", $RM_VCHANN,
    "--power_line_freq", $POWER_LINE_FREQ,
    "--start_sec", $START_SEC,
    "--end_sec", $END_SEC,
    "--wdw_step_s", $WDW_STEP_S,
    "--force_characterization", $FORCE_CHARACTERIZATION,
    "--force_hfo_detection", $FORCE_HFO_DETECTION,
    "--n_jobs", $N_JOBS,
    "--verbose", $VERBOSE
)

# Add montage_channels if specified
if ($MONTAGE_CHANNELS -ne "") {
    $python_args += @("--montage_channels", $MONTAGE_CHANNELS)
}

Write-Host "Starting HFO detection..." -ForegroundColor Green
Write-Host "Command: uv run python $($python_args -join ' ')" -ForegroundColor Gray
Write-Host ""

try {
    # Run the Python script with uv
    & uv run python @python_args
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "HFO detection completed successfully!" -ForegroundColor Green
        Write-Host "Results saved to: $OUTPUT_FOLDER" -ForegroundColor Cyan
    } else {
        Write-Error "HFO detection failed with exit code: $LASTEXITCODE"
        exit $LASTEXITCODE
    }
} catch {
    Write-Error "Error running HFO detection: $_"
    exit 1
}

Write-Host ""
Write-Host "Script execution completed." -ForegroundColor Green
