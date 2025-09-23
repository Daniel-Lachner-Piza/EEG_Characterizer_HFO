# HFO Detector Instructions

## Installation
### 1. Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

### 2. Download detector
https://github.com/Daniel-Lachner-Piza/EEG_Characterizer_HFO

### 3. Generate virtual environment
uv sync

### 4. Download XGBoost files
https://drive.google.com/drive/folders/1yUeMmSEcIxKHiqPuXfkrhusDih5uNM5g


# How tu run it
### 1. Activate venv
.\.venv\Scripts\activate

### 2. Run the detector:
$dataset_name = "Alex_Test_Run"
$input_folder = "E:/HFO_Detector/EEG_Data"
$output_folder = "E:/HFO_Detector/HFO_Detector_Output/"
$eeg_format = "edf"
$montage_type = "sb"
$montage_channels = " "
$power_line_freq = 50
$force_characterization = "no"
$force_hfo_detection = "no"
$start_sec = 0
$end_sec = -1
$wdw_step_s = 0.1
$n_jobs = -1

python run_eeg_hfo_characterize_detect.py --dataset_name "$dataset_name" --input_folder "$input_folder" --output_folder "$output_folder" --eeg_format "$eeg_format" --montage_type "$montage_type" --montage_channels $montage_channels --rm_vchann "yes" --power_line_freq "$power_line_freq" --start_sec $start_sec  --end_sec $end_sec --wdw_step_s $wdw_step_s --force_characterization "$force_characterization" --force_hfo_detection "$force_hfo_detection" --n_jobs $n_jobs --verbose "yes"

