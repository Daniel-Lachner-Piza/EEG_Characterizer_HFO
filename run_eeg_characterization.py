import sys
import os
#sys.path.append(os.path.dirname(__file__))
import argparse
import socket
import logging

import mne
import numpy as np

from pathlib import Path
from hfo_spectral_detector.spectral_analyzer.characterize_events import characterize_events, collect_chann_spec_events
from hfo_spectral_detector.eeg_io.eeg_io import EEG_IO
from hfo_spectral_detector.prediction.predict_characterize_hfo import HFO_Detector

logger = logging.getLogger(__name__)

def run_eeg_characterization(dataset_name, files_dict, mtg_name, power_line_freqs, out_path):
    
    log_fpath = Path(os.path.dirname(__file__))/ "logs" / "hfo_spectral_detector.log"
    os.makedirs(log_fpath.parent, exist_ok=True)
    logging.basicConfig(
        filename= log_fpath,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger.info('Started')
    logger.info("socket.gethostname() = ", socket.gethostname())
    logger.info(f"Number of CPUs: {os.cpu_count()}")

    logger.info(f"Dataset = {str(dataset_name)}")
    logger.info(f"Output path = {str(out_path)}")

    os.makedirs(out_path, exist_ok=True)

    # Create the detector object
    detector_results_path = out_path / 'Elpi_Detector_Results'
    detector = HFO_Detector(eeg_type=mtg_name, output_path=detector_results_path)
    detector.load_models()

    files_idxs = np.arange(len(files_dict['PatName']))
    #files_idxs = files_idxs[0:6]
    for fi in range(len(files_dict['PatName'])):
        # Read EEG Data
        eeg_fpath = files_dict['Filepath'][fi]
        pat_name = files_dict['PatName'][fi]

        try:
        
            eeg_reader = EEG_IO(eeg_filepath=eeg_fpath, mtg_t=mtg_name)
            eeg_reader.remove_natus_virtual_channels()
            assert eeg_reader.fs > 1000, "Sampling Rate is under 1000 Hz!"

            # Read Elpi annotations
            elpi_annots = []#load_elpi_file(data_path+elpi_data_filename)

            # Define time windows to use for spectral analysis
            fs = eeg_reader.fs
            ANALYSIS_START_S = 1
            ANALYSIS_START_SAMPLE = ANALYSIS_START_S*fs
            wdw_duration_samples = int(1 * fs)
            step_samples = int(np.round(0.1 * fs))
            #step_samples = int(np.round(1 * fs))
            ANALYSIS_END_SAMPLE = eeg_reader.n_samples
            #ANALYSIS_END_SAMPLE = ANALYSIS_START_SAMPLE + 60*fs

            logger.info(pat_name)
            logger.info(f"EEG Duration: {eeg_reader.n_samples/fs} seconds")
            logger.info(f"EEG Sampling Rate: {eeg_reader.fs} Hz")
            logger.info(f"EEG Nr. Samples: {eeg_reader.n_samples} samples")            
            logger.info(f"ANALYSIS_START_SAMPLE Sample: {ANALYSIS_START_SAMPLE}")
            logger.info(f"ANALYSIS_END_SAMPLE Sample: {ANALYSIS_END_SAMPLE}")
            logger.info(f"Nr. Channels: {len(eeg_reader.ch_names)}")
            logger.info(f"{eeg_reader.ch_names}\n")


            #assert len(elpi_annots)>1, "Elpi Annotations are empty!"
            
            if len(elpi_annots)>1:
                ANALYSIS_START_SAMPLE = (np.floor(elpi_annots.StartSec.min())-1)*fs
                if ANALYSIS_START_SAMPLE < fs:
                    ANALYSIS_START_SAMPLE = fs
                ANALYSIS_END_SAMPLE = (np.ceil(elpi_annots.EndSec.max())-1)*fs
                if ANALYSIS_END_SAMPLE > eeg_reader.n_samples-fs:
                    ANALYSIS_END_SAMPLE = eeg_reader.n_samples-fs

            an_wdws_dict = {'start':[], 'end':[]}
            an_wdws_dict['start'] = np.array(np.arange(ANALYSIS_START_SAMPLE, ANALYSIS_END_SAMPLE-wdw_duration_samples+1, step_samples).tolist()).astype(int)
            an_wdws_dict['end'] =  an_wdws_dict['start'] + wdw_duration_samples
            assert sum( an_wdws_dict['start']<0)==0, "Incorrectly defined analysis window start samples"
            assert sum(an_wdws_dict['end']>ANALYSIS_END_SAMPLE)==0, "Incorrectly defined analysis window end samples"

            GO_PARALLEL = True
            SAVE_SPECT_IMAGE = False
            FORCE_RECALC = False

            # Obtain the feature characterizing each time window
            params = {
                "pat_name": pat_name,
                "eeg_reader": eeg_reader,
                "an_wdws_dict": an_wdws_dict,
                "out_path": out_path,
                "power_line_freqs":power_line_freqs,
                "go_parallel":GO_PARALLEL,
                "force_recalc":FORCE_RECALC,
                "save_spect_img":SAVE_SPECT_IMAGE,
            }
            characterize_events(**params)
            contour_objs_df = collect_chann_spec_events(**params)
            
            detector.set_fs(fs)
            detector.run_hfo_detection(contour_objs_df)
            
        except Exception as e:
            logger.error(f"Error processing {pat_name}: {e}")
            continue

    logger.info('Finished')


    pass

if __name__ == "__main__":

    # Automatically enable test mode when running in debugger or no arguments passed
    test_mode = sys.gettrace() is not None or len(sys.argv) == 1

    if test_mode:
        # Define Dataset name and data path
        dataset_name="PhysioTest_DLP"
        input_folder=Path("/home/dlp/Documents/Development/Data/Physio_EEG_Data/")
        output_folder=Path("/home/dlp/Documents/Development/Data/Test-DLP-Output/")

        input_folder=Path("/work/jacobs_lab/EEG_Data/AnonymPhysioEEGs/HFOHealthy1monto2yrs/")
        output_folder=Path(f"/work/jacobs_lab/Output/Output_{dataset_name}/HFOHealthy1monto2yrs/")

        eeg_format="edf"
        montage_type="sb"
        power_line_freqs=60
    else:
        parser = argparse.ArgumentParser(description='Characterize EEG to detect HFO')
        parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
        parser.add_argument('--input_folder', type=str, required=True, help='Path to directory containing EEG files')
        parser.add_argument('--output_folder', type=str, help='Path to the output directory')
        parser.add_argument('--eeg_format', type=str, default="edf", help='File format of the EEG files (default: edf)')
        parser.add_argument('--montage_type', type=str, required=True, help='Name of the montage (ib, ir, sb, sr)')
        parser.add_argument('--power_line_freq', type=int, default=60, help='Frequency of POwer Lines')

        args = parser.parse_args()

        dataset_name = args.dataset_name
        input_folder = Path(args.input_folder)
        output_folder = Path(args.output_folder)
        eeg_format = args.eeg_format
        montage_type = args.montage_type
        power_line_freqs = int(args.power_line_freq)

    print("socket.gethostname() = ", socket.gethostname())
    print(f"Number of CPUs: {os.cpu_count()}")

    print(f"Dataset = {str(dataset_name)}")
    print(f"Input path = {str(input_folder)}")
    print(f"EEG format = {str(eeg_format)}")
    print(f"Output path = {str(output_folder)}")
    if test_mode:
        print(f"Running Test Mode!")

    logger.info(f"Dataset = {str(dataset_name)}")
    logger.info(f"Input path = {str(input_folder)}")
    logger.info(f"EEG format = {str(eeg_format)}")
    logger.info(f"Output path = {str(output_folder)}")
    
    if test_mode:
        logger.info(f"Running test mode!")
    else:
        logger.info(f"Running in normal mode!")

    # Populate dictionary with files to process
    print("Files to process:")
    logger.info("Files to process:")

    files_dict = {'PatName': [], 'Filepath': []}
    for path in input_folder.glob(f"**/*.{eeg_format}"):
        files_dict['Filepath'].append(path)
        files_dict['PatName'].append(path.stem)

        print(f"{path.stem}")
        logger.info(f"{path.stem}")


    run_eeg_characterization(dataset_name, files_dict, montage_type, power_line_freqs, output_folder)
