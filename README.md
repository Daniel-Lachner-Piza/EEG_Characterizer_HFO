#create poetry environment using Python 3.11
poetry env use "C:\Python\Python311\python.exe" 

# install all packages to environment
poetry install

#start mlflow ui
cd C:\Users\HFO\Documents\Postdoc_Calgary\Research\Scalp_HFO_Spectral_Detector\hfo_spectral_detector\mlflow
poetry shell
mlflow ui --backend-store-uri sqlite:///mlflow.db