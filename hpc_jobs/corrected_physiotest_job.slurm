#!/bin/bash

#SBATCH --job-name=hfo_detection_physiotest

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
# Load environment - using improved pipeline
cd ~/Projects/EEG_Characterizer_HFO
source .venv/bin/activate
echo "Python executable: $(which python)"
echo "Current directory: $(pwd)"

####### Run your script #########################
dataset_name=PhysioTest
eegfmt=edf
mtg_name=sb
power_line_frequency=60

# Define data groups to process
data_groups=(
    "HFOHealthy1monto2yrs" 
    "HFOHealthy3to5yrs" 
    "HFOHealthy6to10yrs" 
    "HFOHealthy11to13yrs" 
    "HFOHealthy14to17yrs"
)

echo "Starting HFO detection for ${#data_groups[@]} data groups..."

# Process each data group
for group in "${data_groups[@]}"; do
    echo "Processing group: $group"
    
    inpath="/work/jacobs_lab/EEG_Data/AnonymPhysioEEGs/${group}/"
    oupath="/work/jacobs_lab/Output/Output_${dataset_name}/${group}/"
    
    # Check if input directory exists
    if [ ! -d "$inpath" ]; then
        echo "Warning: Input directory does not exist: $inpath"
        continue
    fi
    
    # Create output directory
    mkdir -p "$oupath"
    
    echo "  Input:  $inpath"
    echo "  Output: $oupath"
    
    # Run the improved HFO detection pipeline
    python ~/Projects/EEG_Characterizer_HFO/run_eeg_characterization.py \
        --input_folder "$inpath" \
        --dataset_name "${dataset_name}_${group}" \
        --output_folder "$oupath" \
        --eeg_format "$eegfmt" \
        --montage_type "$mtg_name" \
        --power_line_freq "$power_line_frequency" \
        --max_retries 5 \
        --verbose \
        --log_level INFO
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "Successfully completed processing for group: $group"
    else
        echo "Error: Processing failed for group: $group"
    fi
    
    echo "----------------------------------------"
done

echo "HFO detection job completed for all groups."
echo "Check individual log files in: ~/Projects/EEG_Characterizer_HFO/logs/"
