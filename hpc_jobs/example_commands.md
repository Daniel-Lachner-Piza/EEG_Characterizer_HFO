# ARC

# 1.Create venv to run detector
1.Install uv python from astral
´´´sh
mkdir ~/tmp
cd ~/tmp
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~
´´´

2.Copy repo to Projects folder
´´´sh
mkdir Projects
wget ###
´´´

2.In Projects folder run the following:
´´´sh
uv sync
´´´

3.Activate the environment
´´´sh
. activate
´´´

4. Run EEG characterization 
´´´sh
inpath=/work/jacobs_lab/PhysioEEG/
oupath=/work/jacobs_lab/Test/Output/
eegfmt=edf 
python run_eeg_characterization.py --name FRA_Test --inpath $inpath --outpath $oupath --format $eegfmt
´´´

# Connect to ARC
	ssh arc.ucalgary.ca 

#Job submission
´´´sh
	module load conda
´´´

´´´sh
	conda activate hfo-detect
´´´

´´´sh
	salloc --mem=64G --partition=bigmem -c 80 -N 1 -n 1  -t 03:00:00

	salloc --mem=80G --partition=bigmem -c 80 -N 1 -n 1  -t 01:00:00	
´´´

´´´sh
	python ~/Projects/Scalp_HFO_Spectral_Detector/run_main.py
´´´

# Monitor job
´´´sh
	arc.job-info job_id
´´´
# Repeat job monitoring command
´´´sh
while sleep 2; do arc.job-info 35038925; done
´´´

# Kill Job
´´´sh
	scancel job_id
´´´


# Check Job efficiency
´´´sh
	seff job_id
´´´

# Check all submitted jobs
´´´sh
	squeue -u $USER
´´´

# List all jobs froma certain user
´´´sh
	sacct --starttime 2025-04-01 -u daniel.lachnerpiza
´´´


# Data Transfer

# Single files
Open terminal in local computer
Use secure copy to copy single file:
´´´sh
scp daniel.lachnerpiza@arc.ucalgary.ca:~/Projects/Scalp_HFO_Spectral_Detector/requirements.txt ~/Documents/
´´´

## From local to remote
´´´sh
	source_path="/mnt/c/Users/HFO/Documents/Postdoc_Calgary/Research/Characterized_Spectral_Blobs/1_Characterized_Objects_ACH_27_Multidetect_SOZ_Study/"
	destination_path="daniel.lachnerpiza@arc.ucalgary.ca:/work/jacobs_lab/Detection_Output/Characterized_Spectral_Blobs/1_Characterized_Objects_ACH_27_Multidetect_SOZ_Study/"
	file_type=".parquet"
	echo ""
	echo ""
	echo "----------------------------------------------- Data Transfer Info -----------------------------------------------"
	echo Source Path : $source_path
	echo Destination Path : $destination_path
	echo File Type : $file_type
	echo "-------------------------------------------------------------------------------------------------------------------"
	echo ""
	echo ""

	rsync --partial --progress $source_path*$file_type $destination_path
´´´
## From remote to local
	rsync --partial --progress $destination_path*$file_type $source_path

## Count Number of files in a folder
ls -1q /work/jacobs_lab/PhysioEEG/HFOHealthy3to5yrs/ | wc -l


# Run non interactive josb

	sbatch bp_ieeg_hfo_detection.slurm

# Check Job status
	squeue -u $USER
	arc.job-info job_id

simple_job.sh:

#!/bin/bash
####### Reserve computing resources #############
#SBATCH --mail-user=daniel.lachnerpiza@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --mem=70G
#SBATCH --partition=bigmem

####### Set environment variables ###############
module load conda
conda activate hfo-detect
echo $(which python)

####### Run your script #########################
python ~/Projects/Scalp_HFO_Spectral_Detector/run_main.py





# MARC Cluster Guide
https://rcs.ucalgary.ca/index.php/MARC_Cluster_Guide

# Map SCDS to local drive
https://ucalgary.service-now.com/kb_view.do?sysparm_article=KB0030204

MARC Videos
https://uofc.sharepoint.com/sites/SPO-DEPT-Projects-IT/SCP%20%20Training%20Videos/Forms/AllItems.aspx?RootFolder=%2Fsites%2FSPO%2DDEPT%2DProjects%2DIT%2FSCP%20%20Training%20Videos%2FMARC%20%2D%20Training%20Materials%2FTraining%20Videos&View=%7BF9223443%2D5AEA%2D48A3%2DA73A%2DDB55B14E2F83%7D


# Use VPN to connect to the university network
- Use FortiClient VPN

# Acess the Secure Computing Document Storage Drive created (HFO406905 )

### If you're off campus you can access it via the following
	-Navigate to the ShareFile Sign In page: https://ucalgary.sharefile.com
	- Sign in with company credentials
	


# Access MARC

## Access the SCDS drive 
	smbclient -L scds.uc.ucalgary.ca
	smbclient \\\\scds.uc.ucalgary.ca\\HFO406905 -U daniel.lachnerpiza@ucalgary.casmbclient \\\\scds.uc.ucalgary.ca\\HFO406905 -U daniel.lachnerpiza@ucalgary.ca 
	
### Transfer a single file
	get get mouse_genome.csv
	
### Transfer multiple files
	recurse ON
	
	#Optional,  turns off the interactive mode
	prompt OFF
	
	#Get all files in folder
	mget *
	
	exit
	
# Docker Container to Apptainer
https://rcs.ucalgary.ca/How_to_convert_a_Docker_container_to_an_Apptainer_container 




# TRansfer EEG files
groups_ls=(
	"HFOHealthy1monto2yrs" 
	"HFOHealthy3to5yrs" 
	"HFOHealthy6to10yrs" 
	"HFOHealthy11to13yrs" 
	"HFOHealthy14to17yrs"
)

for group in ${groups_ls[@]}
do
	echo $group
	source_path="/mnt/c/Users/HFO/Documents/Postdoc_Calgary/Research/Tatsuya/PhysioEEGs/Anonymized_EDFs/${group}"
	destination_path="daniel.lachnerpiza@arc.ucalgary.ca:/work/jacobs_lab/EEG_Data/AnonymPhysioEEGs/${group}"
	file_type="*/*.edf"
	echo ""
	echo ""
	echo "----------------------------------------------- Data Transfer Info -----------------------------------------------"
	echo Source Path : $source_path
	echo Destination Path : $destination_path
	echo File Type : $file_type
	echo "-------------------------------------------------------------------------------------------------------------------"
	echo ""
	echo ""
	
	rsync --partial --progress $source_path*$file_type $destination_path
done


# 