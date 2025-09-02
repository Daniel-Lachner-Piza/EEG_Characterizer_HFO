import sys
import os
#sys.path.append(os.path.dirname(__file__))
import argparse
import socket
import logging

import mne
import numpy as np

from pathlib import Path
from hfo_spectral_detector.spectral_analyzer.characterize_events import characterize_events
from hfo_spectral_detector.eeg_io.eeg_io import EEG_IO
from hfo_spectral_detector.prediction.predict_characterize_hfo import HFO_Detector


def init_logging():
    logger = logging.getLogger(__name__)
    log_fpath = Path(os.path.dirname(__file__))/ "logs" / "hfo_spectral_detector.log"
    os.makedirs(log_fpath.parent, exist_ok=True)
    logging.basicConfig(
        filename= log_fpath,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logger

logger = init_logging()

class Characterization_Config():
    def __init__(self, args):

        self.dataset_name = args.dataset_name
        self.input_folder = Path(args.input_folder)
        self.output_folder = Path(args.output_folder)
        self.eeg_format = args.eeg_format
        self.montage_type = args.montage_type
        self.power_line_freqs = int(args.power_line_freq)
        self.start_sec=args.start_sec
        self.end_sec=args.end_sec
        self.wdw_step_s=args.wdw_step_s
        self.force_characterization=args.force_characterization
        self.force_hfo_detection=args.force_hfo_detection
        self.n_jobs=args.n_jobs


    def display_config(self):
        print(f"Dataset = {str(self.dataset_name)}")
        print(f"Input folder = {str(self.input_folder)}")
        print(f"Output folder = {str(self.output_folder)}")
        print(f"EEG format = {str(self.eeg_format)}")
        print(f"Montage type = {str(self.montage_type)}")
        print(f"Power line frequencies = {str(self.power_line_freqs)}")
        print(f"Force characterization = {str(self.force_characterization)}")
        print(f"Force HFO detection = {str(self.force_hfo_detection)}")

    def log_config(self, logger):
        logger.info(f"Dataset = {str(self.dataset_name)}")
        logger.info(f"Input folder = {str(self.input_folder)}")
        logger.info(f"Output folder = {str(self.output_folder)}")
        logger.info(f"EEG format = {str(self.eeg_format)}")
        logger.info(f"Montage type = {str(self.montage_type)}")
        logger.info(f"Power line frequencies = {str(self.power_line_freqs)}")
        logger.info(f"Force characterization = {str(self.force_characterization)}")
        logger.info(f"Force HFO detection = {str(self.force_hfo_detection)}")

class InputDataExtractor:
    def __init__(self, cfg: Characterization_Config):
        self.cfg = cfg

    def get_files_to_process(self):
        # Get the filepaths from the EEG files in the input folder
        eeg_input_files_ls = list(self.cfg.input_folder.glob(f"**/*.{self.cfg.eeg_format}"))
        return eeg_input_files_ls

    def display_files_to_process(self):
        files = self.get_files_to_process()
        for f in files:
            print(f)
    def log_files_to_process(self, logger):
        files = self.get_files_to_process()
        for f in files:
            logger.info(f"EEG File to process: {f}")

def run_eeg_characterization(cfg: Characterization_Config, files_to_process: list):

    os.makedirs(cfg.output_folder, exist_ok=True)

    # Create the detector object
    detector_results_path = cfg.output_folder / 'Elpi_Detector_Results'
    detector = HFO_Detector(eeg_type=cfg.montage_type, output_path=detector_results_path)
    detector.load_models()

    for eeg_fpath in files_to_process:
        
        # Read EEG Data
        pat_name = eeg_fpath.stem
        try:
        
            eeg_reader = EEG_IO(eeg_filepath=eeg_fpath, mtg_t=cfg.montage_type)
            fs = eeg_reader.fs
            eeg_reader.remove_natus_virtual_channels()
            assert eeg_reader.fs > 1000, "Sampling Rate is under 1000 Hz!"

            # Define time windows to use for spectral analysis
            WDW_LENGTH_SAMPLES = int(1 * fs)
            GO_PARALLEL = True
            SAVE_SPECT_IMAGE = False

            analysis_start_sample = cfg.start_sec*fs
            analysis_end_sample = eeg_reader.n_samples
            if cfg.end_sec>0:
                analysis_end_sample = cfg.end_sec*fs                
            wdw_step_samples = int(np.round(cfg.wdw_step_s * fs))

            logger.info(pat_name)
            logger.info(f"EEG Duration: {eeg_reader.n_samples/fs} seconds")
            logger.info(f"EEG Sampling Rate: {eeg_reader.fs} Hz")
            logger.info(f"EEG Nr. Samples: {eeg_reader.n_samples} samples")
            logger.info(f"Analysis start second: {cfg.start_sec}")
            logger.info(f"Analysis end second: {cfg.end_sec}")
            logger.info(f"Analysis start sample: {analysis_start_sample}")
            logger.info(f"Analysis end sample: {analysis_end_sample}")
            logger.info(f"Nr. Channels: {len(eeg_reader.ch_names)}")
            logger.info(f"{eeg_reader.ch_names}\n")

            an_wdws_dict = {'start':[], 'end':[]}
            an_wdws_dict['start'] = np.array(np.arange(analysis_start_sample, analysis_end_sample-WDW_LENGTH_SAMPLES+1, wdw_step_samples).tolist()).astype(int)
            an_wdws_dict['end'] =  an_wdws_dict['start'] + WDW_LENGTH_SAMPLES
            assert sum( an_wdws_dict['start']<0)==0, "Incorrectly defined analysis window start samples"
            assert sum(an_wdws_dict['end']>analysis_end_sample)==0, "Incorrectly defined analysis window end samples"

            # Obtain the feature characterizing each time window
            params = {
                "pat_name": pat_name,
                "eeg_reader": eeg_reader,
                "an_wdws_dict": an_wdws_dict,
                "out_path": cfg.output_folder,
                "power_line_freqs":cfg.power_line_freqs,
                "n_jobs":cfg.n_jobs,
                "force_recalc":cfg.force_characterization.lower()=="yes",
                "save_spect_img":SAVE_SPECT_IMAGE,
            }

            allch_events_fpath = characterize_events(**params)
            detector.set_fs(fs)
            detector.run_hfo_detection(pat_name, allch_events_fpath, force_recalc=cfg.force_hfo_detection.lower()=="yes")

        except Exception as e:
            logger.error(f"Error processing {pat_name}: {e}")
            continue

    logger.info('Finished')


    pass

if __name__ == "__main__":

    # Automatically enable test mode when running in debugger or no arguments passed
    test_mode = sys.gettrace() is not None or len(sys.argv) == 1

    if test_mode:
        class args():
            def __init__(self):
                self.dataset_name="PhysioTest_DLP"

                self.input_folder="/home/dlp/Documents/Development/Data/Physio_EEG_Data/"
                self.output_folder="/home/dlp/Documents/Development/Data/Test-DLP-Output/"                
                #self.input_folder="/work/jacobs_lab/EEG_Data/AnonymPhysioEEGs/HFOHealthy1monto2yrs/")
                #self.output_folder="/work/jacobs_lab/Output/Output_{dataset_name}/HFOHealthy1monto2yrs/")
                
                self.eeg_format="edf"
                self.montage_type="sb"
                self.power_line_freq=60
                self.force_characterization="yes"
                self.force_hfo_detection="yes"
                self.start_sec=0.0
                self.end_sec=60.0
                self.wdw_step_s=1.0
                self.n_jobs=-1

        args = args()
    else:
        parser = argparse.ArgumentParser(description='Characterize EEG to detect HFO')
        parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
        parser.add_argument('--input_folder', type=str, required=True, help='Path to directory containing EEG files')
        parser.add_argument('--output_folder', type=str, help='Path to the output directory')
        parser.add_argument('--eeg_format', type=str, default="edf", help='File format of the EEG files (default: edf)')
        parser.add_argument('--montage_type', type=str, required=True, help='Name of the montage (ib, ir, sb, sr)')
        parser.add_argument('--power_line_freq', type=int, default=60, help='Frequency of Power Lines')
        parser.add_argument('--start_sec', type=float, default=0, help='Start analysis from a specific second, defailt is 0')
        parser.add_argument('--end_sec', type=float, default=-1, help='End analysis from a specific second, -1 is full length')
        parser.add_argument('--wdw_step_s', type=float, default=0.1, help='Window step size in seconds, default is 0.1')
        parser.add_argument('--force_characterization', type=str, default="no", help='Force recalculation of features')
        parser.add_argument('--force_hfo_detection', type=str, default="yes", help='Force HFO detection')
        parser.add_argument('--n_jobs', type=int, default=-1, help='Number of jobs to run in parallel, -1 uses all CPU cores')

        args = parser.parse_args()

    cfg = Characterization_Config(args)
    data_extractor = InputDataExtractor(cfg)
    files_to_process = data_extractor.get_files_to_process()

    run_mode_str = "Running normal mode"
    if test_mode:
        print(f"**********Running Test Mode!**********")
        run_mode_str = "Running test mode"

    print("socket.gethostname() = ", socket.gethostname())
    print(f"Number of CPUs: {os.cpu_count()}")
    logger.info(run_mode_str)
    logger.info(f"socket.gethostname() = {socket.gethostname()}")
    logger.info(f"Number of CPUs: {os.cpu_count()}")

    cfg.display_config()
    cfg.log_config(logger)

    data_extractor.display_files_to_process()
    data_extractor.log_files_to_process(logger)

    run_eeg_characterization(cfg, files_to_process)
