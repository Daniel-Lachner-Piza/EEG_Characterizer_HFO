#!/bin/bash

#SBATCH --job-name=hfo_detection_unifolder_job_%u_%j
#SBATCH --output=Slurm_Logs/hfo_detection_unifolder_job_%u_%j.out
#SBATCH --error=Slurm_Logs/hfo_detection_unifolder_job_%u_%j.err

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
dataset_name=Clustering_Patho_HFO
input_folder="/work/jacobs_lab/EEG_Data/Clustering_Patho_HFO/"
output_folder="/work/jacobs_lab/Output/${dataset_name}/"
eeg_format=edf              # Tested formats: edf, vhdr(Brain Vision), dat(Persyst's laydat format) 
montage_type=sb             # Scalp bipolar: sb, Scalp referential: sr, iEEG bipolar: ib, iEEG referential: ir
power_line_freq=60          # 60Hz = North America, 50Hz = Europe
force_characterization="no" # Recalculate features even if files present
force_hfo_detection="no"   # Force HFO detection even if files present

python ~/Projects/EEG_Characterizer_HFO/run_eeg_hfo_characterize_detect.py \
    --dataset_name "$dataset_name" \
    --input_folder "$input_folder" \
    --output_folder "$output_folder" \
    --eeg_format "$eeg_format" \
    --montage_type "$montage_type" \
    --montage_channels "" \
    --rm_vchann "yes" \
    --power_line_freq "$power_line_freq" \
    --start_sec 0 \
    --end_sec -1 \
    --wdw_step_s 0.1 \
    --force_characterization "$force_characterization" \
    --force_hfo_detection "$force_hfo_detection" \
    --n_jobs -1 \
    --verbose "yes"

echo "All processing completed."
