#!/bin/bash

#SBATCH --job-name=hfo_detection_multifolder_job_%u_%j
#SBATCH --output=~/Slurm_Logs/hfo_detection_multifolder_job_%u_%j.out
#SBATCH --error=~/Slurm_Logs/hfo_detection_multifolder_job_%u_%j.err

####### Reserve computing resources #############
#SBATCH --mail-user=daniel.lachnerpiza@ucalgary.ca
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=1-00:00:00
#SBATCH --mem=80G
#SBATCH --partition=bigmem

####### Set environment variables ###############
source ~/Projects/EEG_Characterizer_HFO/.venv/bin/activate
echo $(which python)

####### Run your script #########################
dataset_name=ACH_Physio_Scalp_Data
input_folder_base="/work/jacobs_lab/EEG_Data/AnonymPhysioEEGs/"
output_folder_base="/work/jacobs_lab/Output/"
eeg_format=edf              # Tested formats: edf, vhdr(Brain Vision), dat(Persyst's laydat format) 
montage_type=sb             # Scal bipolar: sb, Scalp referential: sr, iEEG bipolar: ib, iEEG referential: ir
power_line_freq=60          # 60Hz = North America, 50Hz = Europe
force_characterization="no" # Recalculate features even if files present
force_hfo_detection="no"   # Force HFO detection even if files present
data_groups=(
    "HFOHealthy1monto2yrs" 
    "HFOHealthy3to5yrs" 
    "HFOHealthy6to10yrs" 
    "HFOHealthy11to13yrs" 
    "HFOHealthy14to17yrs"
)



parser.add_argument('--dataset_name', type=str, required=True, 
                       help='Name of the dataset')
    parser.add_argument('--rm_vchann', type=str, default="yes", 
                    choices=['yes', 'no'], help='Remove Natus virtual channels if present')
    parser.add_argument('--input_folder', type=str, required=True, 
                       help='Path to directory containing EEG files')
    parser.add_argument('--output_folder', type=str, 
                       help='Path to the output directory')
    parser.add_argument('--eeg_format', type=str, default="edf", 
                       help='File format of the EEG files')
    parser.add_argument('--montage_type', type=str, required=True, 
                       help='Name of the montage (ib, ir, sb, sr)')
    parser.add_argument('--montage_channels', type=str, default="",
                       help='Comma separated montage channels to detect, empty detects all (e.g., "F3-C3,C3-P3, F4-C4,C4-P4"')
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


for group in ${data_groups[@]}; do
  input_folder="${input_folder_base}${group}/"
  output_folder="${output_folder_base}Output_${dataset_name}/${group}/"
  python ~/Projects/EEG_Characterizer_HFO/run_eeg_characterization.py \
    --dataset_name "$dataset_name" \
    --input_folder "$input_folder" \
    --output_folder "$output_folder" \
    --eeg_format "$eeg_format" \
    --montage_type "$montage_type" \
    --montage_channels "" \
    --power_line_freq "$power_line_freq" \
    --start_sec 0 \
    --end_sec -1 \
    --wdw_step_s 0.1 \
    --force_characterization "$force_characterization" \
    --force_hfo_detection "$force_hfo_detection" \
    --n_jobs -1 \
    --verbose "yes"
done
echo "All processing completed."
