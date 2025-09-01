#!/bin/bash

#SBATCH --job-name=hfo_detection_multifolder_job_%u_%j
#SBATCH --output=~/Slurm_Logs/%x.out
#SBATCH --error=~/Slurm_Logs/%x.err

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

for group in ${data_groups[@]}; do
  input_folder="${input_folder_base}${group}/"
  output_folder="${output_folder_base}Output_${dataset_name}/${group}/"
  python ~/Projects/EEG_Characterizer_HFO/run_eeg_characterization.py \
    --dataset_name "$dataset_name" \
    --input_folder "$input_folder" \
    --output_folder "$output_folder" \
    --eeg_format "$eeg_format" \
    --montage_type "$montage_type" \
    --power_line_freq "$power_line_freq" \
    --force_characterization "$force_characterization" \
    --force_hfo_detection "$force_hfo_detection"
done
echo "All processing completed."
