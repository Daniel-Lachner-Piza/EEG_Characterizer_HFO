#!/bin/bash

#SBATCH --job-name=bipolar_scalp_hfo_detection_job

####### Reserve computing resources #############
#SBATCH --mail-user=daniel.lachnerpiza@ucalgary.ca
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=7-00:00:00
#SBATCH --mem=20G
#SBATCH --partition=cpu2019

####### Set environment variables ###############
module load conda
conda activate hfo-detect
echo $(which python)

####### Run your script #########################
python ~/Projects/Scalp_HFO_Spectral_Detector/run_physio_patients.py
