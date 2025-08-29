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
    def __init__(self, eeg_type:str="", output_path:str=None) -> None:

        assert eeg_type in ['sr', 'sb', 'ir', 'ib'], "EEG type must be one of 'sr', 'sb', 'ir', or 'ib'."

        self.eeg_type = eeg_type
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        
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

    def select_obvious_gs_negative_objs(self, gs_objs_df, fs):

        ################################################
        min_nr_bp_osc = 2 # 3
        hvr_nok = gs_objs_df.hvr.to_numpy() < 10 # 10
        spect_osc_nok = gs_objs_df.nr_oscillations.to_numpy()<4  #<3
        nr_prom_peaks_nok = np.logical_and(gs_objs_df.prom_peaks_nr.to_numpy() < min_nr_bp_osc, gs_objs_df.inverted_prom_peaks_nr.to_numpy() < min_nr_bp_osc)
        valid_obj_sel = np.logical_or.reduce((hvr_nok, spect_osc_nok))

        return valid_obj_sel

    def auto_hfo_detection(self, contour_objs_df, eeg_n_samples, eeg_fs):

        # Get patient name
        pat_name = contour_objs_df.Patient.unique()
        assert len(pat_name)==1, "Features DataFrame contains more than one patient."

        print("Classifier model: ", self.classifier_model_fpath.stem)
        print(f"Processing {pat_name}")

        # Check structure of data frame with contour objects
        contour_objs_df = contour_objs_df.loc[:, ~contour_objs_df.columns.str.contains('Unnamed')]
        assert len(contour_objs_df)>0, "No HFO objects found for this patient"
        contour_objs_df = contour_objs_df.loc[:, ~contour_objs_df.columns.str.contains('Unnamed')]

        # Add features describing the ratio between Event and Background
        contour_objs_df['EventBkgrndRatio_Power'] = contour_objs_df['bp_sig_pow']/contour_objs_df['bkgrnd_sig_pow']
        contour_objs_df['EventBkgrndRatio_StdDev'] = contour_objs_df['bp_sig_std']/contour_objs_df['bkgrnd_sig_std']
        contour_objs_df['EventBkgrndRatio_Activity'] = contour_objs_df['bp_sig_activity']/contour_objs_df['bkgrnd_sig_activity']
        contour_objs_df['EventBkgrndRatio_Mobility'] = contour_objs_df['bp_sig_avg_mobility']/contour_objs_df['bkgrnd_sig_avg_mobility']
        contour_objs_df['EventBkgrndRatio_Complexity'] = contour_objs_df['bp_sig_complexity']/contour_objs_df['bkgrnd_sig_complexity']
        
        # # Detection of HFO contour objects
        very_negative_sel = self.select_obvious_gs_negative_objs(contour_objs_df, 0)
        contours_to_detect_df = contour_objs_df[np.logical_not(very_negative_sel)].reset_index(drop=True)

        scaled_feat_vals = self.feat_scaler.transform(contours_to_detect_df[self.feature_selection].to_numpy())
        y_pred = self.classifier_model.predict(scaled_feat_vals).ravel()

        detected_hfo_contours_df = contours_to_detect_df[y_pred>0].reset_index(drop=True).copy()

        if detected_hfo_contours_df.shape[0]> 0:
            # Save the detected HFO contours to a new file
            elpi_hfo_marks_fn = f"{pat_name}_hfo_detections.mat"
            elpi_hfo_marks_fpath = self.output_path / elpi_hfo_marks_fn

            elpi_hfo_detections_df = self.contour_objs_to_elpi(detected_hfo_contours_df, eeg_n_samples, eeg_fs)
            elpi_hfo_detections_df.Type = f"spctHFO"

            if len(elpi_hfo_detections_df)>0:
                write_elpi_file(elpi_hfo_detections_df, elpi_hfo_marks_fpath)
        else:
            print(f"No HFO detections found for {pat_name}. Skipping to next file.")

        return
 
    def contour_objs_to_elpi(self, eoi_feats_df, eeg_n_samples, eeg_fs):
        """
        Create the elpi files with the HFO detections
        """
        # Validate input
        pat_names = eoi_feats_df.Patient.unique()
        assert len(pat_names) == 1, "Features DataFrame contains more than one patient."
        pat_name = pat_names[0]
        
        fs = eeg_fs
        assert fs > 1000, "Sampling Rate is under 1000 Hz!"
        
        # Pre-allocate result dictionary
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
        
        # Get unique channels
        channels = eoi_feats_df.channel.unique()
        
        # Generate creation time once
        creation_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        
        for channel_idx, channel in enumerate(channels):
            print(f"Processing channel {channel} ({channel_idx+1}/{len(channels)})")
            
            # Filter data for current channel (case-insensitive)
            channel_data = eoi_feats_df[
                eoi_feats_df.channel.str.lower() == channel.lower()
            ].reset_index(drop=True)
            
            if len(channel_data) == 0:
                print(f"No EOI in channel: {channel}")
                continue
                
            # Extract start and end times
            start_times_ms = channel_data.start_ms.to_numpy()
            end_times_ms = channel_data.end_ms.to_numpy()
            
            # Merge overlapping intervals efficiently
            merged_intervals = self._merge_overlapping_intervals(start_times_ms, end_times_ms)            
            if len(merged_intervals) == 0:
                continue
                
            # Convert to arrays for vectorized operations
            merged_start_ms = np.array([interval[0] for interval in merged_intervals])
            merged_end_ms = np.array([interval[1] for interval in merged_intervals])
            
            # Convert to samples (vectorized)
            merged_start_samples = np.round(fs * merged_start_ms / 1000).astype(np.int64)
            merged_end_samples = np.round(fs * merged_end_ms / 1000).astype(np.int64)
            
            num_intervals = len(merged_intervals)
            
            # Extend all dictionary lists at once (more efficient than individual extends)
            all_channs_eoi_dict["Channel"].extend([channel] * num_intervals)
            all_channs_eoi_dict["Type"].extend(["spect_HFO"] * num_intervals)
            all_channs_eoi_dict["StartSec"].extend(merged_start_ms / 1000)
            all_channs_eoi_dict["EndSec"].extend(merged_end_ms / 1000)
            all_channs_eoi_dict["StartSample"].extend(merged_start_samples)
            all_channs_eoi_dict["EndSample"].extend(merged_end_samples)
            all_channs_eoi_dict["Comments"].extend([pat_name] * num_intervals)
            all_channs_eoi_dict["ChSpec"].extend([True] * num_intervals)
            all_channs_eoi_dict["CreationTime"].extend([creation_time] * num_intervals)
            all_channs_eoi_dict["User"].extend(["DLP_Prune_HFO"] * num_intervals)
        
        return pd.DataFrame(all_channs_eoi_dict)
    
    def _merge_overlapping_intervals(self, start_times, end_times):
        """
        Merge overlapping intervals using a sweep line algorithm.
        
        Args:
            start_times: Array of interval start times
            end_times: Array of interval end times
            
        Returns:
            List of merged intervals as (start, end) tuples
        """
        if len(start_times) == 0:
            return []
        
        # Create intervals and sort by start time
        intervals = list(zip(start_times, end_times))
        intervals.sort(key=lambda x: x[0])
        
        merged = [intervals[0]]
        
        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged[-1]
            
            # Check for overlap (including adjacent intervals)
            if current_start <= last_end:
                # Merge intervals by extending the end time
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                # No overlap, add as new interval
                merged.append((current_start, current_end))
        
        return merged


if __name__ == "__main__":

    pass