#!/bin/bash

#SBATCH --job-name=ieeg_bipolar_hfo_detection_job

####### Reserve computing resources #############
#SBATCH --mail-user=daniel.lachnerpiza@ucalgary.ca
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --partition=bigmem

####### Set environment variables ###############
module load conda
conda activate hfo-detect
echo $(which python)

####### Run your script #########################
python ~/Projects/Scalp_HFO_Spectral_Detector/run_main.py
