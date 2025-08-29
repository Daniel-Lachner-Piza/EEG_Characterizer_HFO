# bash command to call 

uv source \.venv\bin\activate

´´´sh
name=Test_DLP
inpath=/home/dlp/Documents/Development/Data/Physio_EEG_Data/
outpath=/home/dlp/Documents/Development/Data/Test-DLP-Output/
fmt=edf
montage=sb
plf=60
python run_eeg_characterization.py --name $name --inpath $inpath --outpath $outpath --format $fmt --montage $montage --plf $plf
´´´


#create poetry environment using Python 3.11
poetry env use "C:\Python\Python311\python.exe" 

# install all packages to environment
poetry install

#start mlflow ui
cd C:\Users\HFO\Documents\Postdoc_Calgary\Research\Scalp_HFO_Spectral_Detector\hfo_spectral_detector\mlflow
poetry shell
mlflow ui --backend-store-uri sqlite:///mlflow.db