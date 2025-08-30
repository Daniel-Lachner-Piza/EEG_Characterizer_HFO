#!/bin/bash

#SBATCH --job-name=frankfurt_clae_hfo_detection_job

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
 . ~/Projects/EEG_Characterizer_HFO/activate.sh
echo $(which python)

####### Run your script #########################
dataset_name=PhysioTest
inpath=/work/jacobs_lab/EEG_Data/AnonymPhysioEEGs/
oupath=/work/jacobs_lab/Output/Output_${dataset_name}/
eegfmt=edf
mtg_name=sb
power_line_frequency=60
python ~/Projects/EEG_Characterizer_HFO/run_eeg_characterization.py --name $dataset_name --inpath $inpath --outpath $oupath --format $eegfmt --montage $mtg_name --plf power_line_frequency
echo "Job finished"