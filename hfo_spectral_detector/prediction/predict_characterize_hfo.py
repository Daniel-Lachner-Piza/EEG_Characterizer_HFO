import mne
import numpy as np
import pandas as pd
import datetime
import os
import joblib
from pathlib import Path

from hfo_spectral_detector.read_setup_eeg.montage_creator import MontageCreator
from hfo_spectral_detector.elpi.elpi_interface import load_elpi_file, write_elpi_file, get_agreement_between_elpi_files
from hfo_spectral_detector.studies_info.studies_info import StudiesInfo
from hfo_spectral_detector.eeg_io.eeg_io import EEG_IO

from xgboost import XGBClassifier
import xgboost as xgb

class HFO_Detector:
    def __init__(self, eeg_type:str="", eeg_data_path:str=None, eeg_filenames:str=None, characterized_data_path:str=None) -> None:

        assert eeg_type in ['sr', 'sb', 'ir', 'ib'], "EEG type must be one of 'sr', 'sb', 'ir', or 'ib'."

        self.eeg_type = eeg_type
        self.eeg_data_path = eeg_data_path
        self.eeg_filenames = eeg_filenames
        self.characterized_data_path = characterized_data_path
        
        # Define the path to the classifier model and the standard scaler
        models_path = Path(__file__).parent
        self.classifier_model_fpath = models_path / "XGB_Single_Class_2025-04-24_22-09_Kappa86.json"
        self.feat_scaler_fpath = models_path /  "XGB_Single_Class_Standard_Scaler_2025-04-24_22-09_Kappa86.bin"
        self.classifier_model = None
        self.feat_scaler = None

        # Define features to use for the classification  
        self.feature_selection = [
            'hvr', 'circularity', 'area', 'nr_oscillations', 
            "bp_sig_ampl", 'bp_sig_pow', 'bp_sig_std',
            'bkgrnd_sig_pow','bkgrnd_sig_std',
            'max_hfo_sine_corr', 'inverted_max_hfo_sine_corr',
            'all_relevant_peaks_nr', 'all_relevant_peaks_avg_freq', 'all_relevant_peaks_freq_stddev', 
            'all_relevant_peaks_amplitude_stability', 
            'all_relevant_peaks_prominence_stability',
            'prom_peaks_nr', 'prom_peaks_avg_freq', 'prom_peaks_freqs_stddev', 
            'prom_peaks_avg_amplitude_stability', 'prom_peaks_prominence_stability',
            'inverted_all_relevant_peaks_amplitude_stability', 'inverted_all_relevant_peaks_prominence_stability', 
            'inverted_prom_peaks_nr', 'inverted_prom_peaks_avg_freq', 'inverted_prom_peaks_freqs_stddev',
            'inverted_prom_peaks_avg_amplitude_stability', 'inverted_prom_peaks_prominence_stability',
            'EventBkgrndRatio_Power', 
            ]

    
    def load_models(self):
        # Check if the model and scaler files exist
        assert os.path.isfile(self.classifier_model_fpath), f"Classifier model file not found: {self.classifier_model_fpath}"
        assert os.path.isfile(self.feat_scaler_fpath), f"Standard scaler file not found: {self.feat_scaler_fpath}"

        # Load model and scaler
        self.classifier_model = XGBClassifier()
        self.classifier_model.load_model(self.classifier_model_fpath)
        self.feat_scaler = joblib.load(self.feat_scaler_fpath)

        pass


    def contour_objs_to_elpi(self, eeg_fn, eoi_feats_df):

        if len(eoi_feats_df) == 0:
            print(f"No EOI features found for {eeg_fn}. Skipping to next file.")
            return pd.DataFrame()
         
        eoir_feats_df_patient_name = eoi_feats_df.Patient.unique()
        assert len(eoir_feats_df_patient_name) == 1

        mtg_labels = []
        fs = 0
        n_samples = 0
        eeg_dur_s = 0


        # Read EEG
        fs = 0
        eeg_fpath = self.eeg_data_path / eeg_fn
        try:
            eeg_reader = EEG_IO(eeg_filepath=eeg_fpath, mtg_t=self.eeg_type)
            fs = eeg_reader.fs
        except:
            print(f"Erro reading scalp-referential montage from EEG file: {eeg_fpath}")
        
        assert fs> 1000, "Sampling Rate is under 1000 Hz!"

        n_samples = eeg_reader.n_samples
        eeg_dur_s = n_samples/fs
        
        # Pre-allocate dictionaries for the elpi file 
        all_channs_eoi_dict = {
            "Channel": [],
            "Type": [],
            "StartSec": [],
            "EndSec": [],
            "StartSample": [],
            "EndSample": [],
            "Comments": [],
            "ChSpec": [],
            "CreationTime": [],
            "User": [],
        }

        if fs > 0:
            pat_name = eoi_feats_df.Patient.unique()
            assert len(pat_name)==1, "Features DataFrame contains more than one patient."

            mtg_labels = eoi_feats_df.channel.unique()

            mask_step_size_ms = 1
            time_mask = np.arange(0, eeg_dur_s*1000, mask_step_size_ms)
            for idx, mtg in enumerate(mtg_labels):
                
                print(f"Processing channel {mtg} ({idx+1}/{len(mtg_labels)})")
                this_ch_objs_df = eoi_feats_df[eoi_feats_df.channel.str.fullmatch(mtg.lower(), case=False)].reset_index(drop=True)
                if len(this_ch_objs_df)>0:
                    chan_eoi_start_ms = this_ch_objs_df.start_ms.to_numpy()
                    chan_eoi_end_ms = this_ch_objs_df.end_ms.to_numpy()

                    # Create a zeroed-mask for all samples
                    # If an EOI is found for a certain sample range, then place a 1 within this range
                    eoi_mask = np.zeros_like(time_mask)
                    for idx, (eoi_start_ms, eoi_end_ms) in enumerate(zip(chan_eoi_start_ms, chan_eoi_end_ms)):
                        eoi_mask[np.logical_and(time_mask>=eoi_start_ms,time_mask<=eoi_end_ms)] = 1

                    # Get the start and end times of the zeroed-mask by finding the first and last 1 from sequences surrounded by zeros
                    eoi_start_times_ms = time_mask[np.argwhere(np.diff(eoi_mask) == 1)+1]
                    eoi_end_times_ms = time_mask[np.argwhere(np.diff(eoi_mask) == -1)]
                    eoi_start_times_ms = eoi_start_times_ms.flatten()
                    eoi_end_times_ms = eoi_end_times_ms.flatten()

                    assert len(eoi_start_times_ms) == len(eoi_end_times_ms), f"Processing patient {pat_name}, channel {mtg}. Start and end times of EOI do not match!"
                    
                    eoi_durations_ms = eoi_end_times_ms-eoi_start_times_ms
                    eoi_start_samples = np.round(fs*eoi_start_times_ms/1000)
                    eoi_start_samples = eoi_start_samples.astype(np.int64)
                    eoi_end_samples = np.round(fs*eoi_end_times_ms/1000)
                    eoi_end_samples = eoi_end_samples.astype(np.int64)

                    nr_eoi = len(eoi_start_times_ms)

                    # Create a dictionary with the EOI information so that it can be read in Elpi
                    creation_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
                    merged_eoi_dict = {
                        "Channel": [mtg] * nr_eoi,
                        "Type": ["spect_HFO"] * nr_eoi,
                        "StartSec": eoi_start_times_ms/1000,
                        "EndSec": eoi_end_times_ms/1000,
                        "StartSample": eoi_start_samples,
                        "EndSample": eoi_end_samples,
                        "Comments": [pat_name] * nr_eoi,
                        "ChSpec": np.ones(nr_eoi, dtype=bool),
                        "CreationTime": [creation_time] * nr_eoi,
                        "User": ["DLP_Prune_HFO"] * nr_eoi,
                    }

                    # Check that all columns have the same number of elements
                    #print([len(value) for key, value in merged_eoi_dict.items()])

                    if nr_eoi > 0:
                        for key, values in merged_eoi_dict.items():
                            assert (key in all_channs_eoi_dict.keys()), "Key not found in all channels dictionary!"
                            if key in all_channs_eoi_dict.keys():
                                all_channs_eoi_dict[key].extend(values)
                else:
                    print("No EOI in channel: ", mtg)

                #detections_mat_fn = (out_files_dest_path + f"{pat_name}_spectrogram_HFO.mat")
                #savemat(detections_mat_fn, all_channs_eoi_dict)
                pass

        return pd.DataFrame(all_channs_eoi_dict)
    
    def select_obvious_gs_negative_objs(self, gs_objs_df, fs):

        ################################################
        min_nr_bp_osc = 2 # 3
        hvr_nok = gs_objs_df.hvr.to_numpy() < 10 # 10
        spect_osc_nok = gs_objs_df.nr_oscillations.to_numpy()<4  #<3
        nr_prom_peaks_nok = np.logical_and(gs_objs_df.prom_peaks_nr.to_numpy() < min_nr_bp_osc, gs_objs_df.inverted_prom_peaks_nr.to_numpy() < min_nr_bp_osc)
        valid_obj_sel = np.logical_or.reduce((hvr_nok, spect_osc_nok))

        return valid_obj_sel

    def auto_hfo_detection(self, out_files_dest_path, force_recalc:bool=True):

        print("Classifier model: ", self.classifier_model_fpath)
        
        hfo_marks_path = out_files_dest_path
        os.makedirs(hfo_marks_path, exist_ok=True)

        for idx, eeg_fn in enumerate(self.eeg_filenames):

            pat_name = eeg_fn
            print(pat_name)

            elpi_hfo_marks_fn = pat_name.replace('.vhdr', '') + "__hfo_detections.mat"
            elpi_hfo_marks_fpath = out_files_dest_path / elpi_hfo_marks_fn
            os.makedirs(elpi_hfo_marks_fpath.parent, exist_ok=True)

            if os.path.isfile(elpi_hfo_marks_fpath) and not force_recalc:
                print(f"File already exists: {elpi_hfo_marks_fpath}")
                continue

            # Read detections
            contour_objs_df_filepath = self.characterized_data_path / f"{pat_name}All_Ch_Objects.parquet"
            try:
                contour_objs_df = pd.read_parquet(contour_objs_df_filepath)
                contour_objs_df = contour_objs_df.loc[:, ~contour_objs_df.columns.str.contains('Unnamed')]

                assert len(contour_objs_df)>0, "No HFO objects found for this patient"
                contour_objs_df = contour_objs_df.loc[:, ~contour_objs_df.columns.str.contains('Unnamed')]
                pass
            except:
                print(f"Detections file file not found: {contour_objs_df_filepath}")
                continue

            if 'EventBkgrndRatio_Power' in list(contour_objs_df.columns):
                pass
            # Add features describing the ratio between Event and Background
            contour_objs_df['EventBkgrndRatio_Power'] = contour_objs_df['bp_sig_pow']/contour_objs_df['bkgrnd_sig_pow']
            contour_objs_df['EventBkgrndRatio_StdDev'] = contour_objs_df['bp_sig_std']/contour_objs_df['bkgrnd_sig_std']
            contour_objs_df['EventBkgrndRatio_Activity'] = contour_objs_df['bp_sig_activity']/contour_objs_df['bkgrnd_sig_activity']
            contour_objs_df['EventBkgrndRatio_Mobility'] = contour_objs_df['bp_sig_avg_mobility']/contour_objs_df['bkgrnd_sig_avg_mobility']
            contour_objs_df['EventBkgrndRatio_Complexity'] = contour_objs_df['bp_sig_complexity']/contour_objs_df['bkgrnd_sig_complexity']

            print(f"\n\nProcessing {contour_objs_df_filepath}")
          
            # # Detection of HFO contour objects
            very_negative_sel = self.select_obvious_gs_negative_objs(contour_objs_df, 0)
            contours_to_detect_df = contour_objs_df[np.logical_not(very_negative_sel)].copy().reset_index(drop=True)

            scaled_feat_vals = self.feat_scaler.transform(contours_to_detect_df[self.feature_selection].to_numpy())
            y_pred = self.classifier_model.predict(scaled_feat_vals).ravel()

            detected_hfo_contours_df = contours_to_detect_df[y_pred>0].reset_index(drop=True).copy()

            if detected_hfo_contours_df.shape[0]> 0:
                # Save the detected HFO contours to a new file
                elpi_hfo_detections_df = self.contour_objs_to_elpi(Path(eeg_fn), detected_hfo_contours_df)
                elpi_hfo_detections_df.Type = f"spctHFO"

                if len(elpi_hfo_detections_df)>0:
                    write_elpi_file(elpi_hfo_detections_df, elpi_hfo_marks_fpath)
            else:
                print(f"No HFO detections found for {eeg_fn}. Skipping to next file.")
                continue

        return
 
if __name__ == "__main__":
   
    # get dataset name and EEG filepaths
    dataset_name, files_dict = StudiesInfo().ACH_27_Multidetect_SOZ_Study()
    eeg_files_ls = np.flip(files_dict['PatName'])
    eeg_data_path = files_dict['Filepath'][0].parent

    # Define path where the characterized contour objects are found and where to store the generated processed data
    characterized_data_path = Path(f"C:/Users/HFO/Documents/Postdoc_Calgary/Research/Characterized_Spectral_Blobs/1_Characterized_Objects_ACH_27_Multidetect_SOZ_Study/")
    detector_results_path = Path(f"C:/Users/HFO/Documents/Postdoc_Calgary/Research/Spectral_HFO_SOZ_Prediction/ACH/ACH_HFO_Detections/")    
    os.makedirs(detector_results_path, exist_ok=True)

    # Create the detector object
    detector = HFO_Detector(eeg_type='ib', 
                            eeg_data_path=eeg_data_path, 
                            eeg_filenames=eeg_files_ls, 
                            characterized_data_path=characterized_data_path, 
                            )

    force_recalc = False

    # best_performer_hfo_detections_filepath = detector_results_path 
    # detector.load_models()
    # detector.best_performer_hfo_detection(best_performer_hfo_detections_filepath, force_recalc)

    # best_performer_hfo_detections_filepath = detector_results_path 
    # detector.load_models()
    # detector.auto_hfo_detection(best_performer_hfo_detections_filepath, force_recalc)

    # best_performer_hfo_detections_filepath = detector_results_path 
    # detector.load_models()
    # detector.auto_hfo_characterize_channels(best_performer_hfo_detections_filepath, force_recalc)

    pass