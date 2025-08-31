### Data Transfer to HPC

# Using rsync to synchronize a local folder with a folder on the HPC
#### Prerequisites
- **Windows**: Open a WSL (Windows Subsystem for Linux) terminal
- **Linux/MacOS**: rsync should be available from your terminal

#### Copy Entire Folder
```bash
source_path="/mnt/c/Users/HFO/Documents/Postdoc_Calgary/Research/Scalp_HFO_GoldStandard/"
destination_path="daniel.lachnerpiza@arc.ucalgary.ca:/work/jacobs_lab/EEG_Data/Scalp_HFO_GoldStandard/"
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
	source_path="/mnt/c/Users/HFO/Documents/Postdoc_Calgary/Research/Tatsuya/PhysioEEGs/Anonymized_EDFs/${group}"
	destination_path="daniel.lachnerpiza@arc.ucalgary.ca:/work/jacobs_lab/EEG_Data/AnonymPhysioEEGs/${group}"
	file_type="*/*.edf"    
    rsync --relative --partial --progress ${source_path}${file_type} $destination_path
done
```

# Transfer a single file using scp command
```bash
scp XGB_Single_Class_2025-04-24_22-09_Kappa86.json daniel.lachnerpiza@arc.ucalgary.ca:~/Projects/EEG_Characterizer_HFO/hfo_spectral_detector/prediction/
```

#### Check folder size and file count on HPC:

```bash
echo "Files: $(find . -type f | wc -l)"; echo "Total size: $(du -sh . | cut -f1)"
```