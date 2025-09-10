import mne
import numpy as np
import pandas as pd
import datetime
import os
import joblib
import json
import logging
from pathlib import Path

from hfo_spectral_detector.elpi.elpi_interface import load_elpi_file, write_elpi_file, get_agreement_between_elpi_files
from hfo_spectral_detector.studies_info.studies_info import StudiesInfo
from hfo_spectral_detector.eeg_io.eeg_io import EEG_IO
from hfo_spectral_detector.prediction.elpi_events_characterizer import HFO_Characterizer

from xgboost import XGBClassifier
import xgboost as xgb

logger = logging.getLogger(__name__)

class HFO_Detector:
    def __init__(self, output_path:Path=Path(Path(os.path.dirname(__file__)))) -> None:

        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        
        # Initialize the characterizer
        self.characterizer = HFO_Characterizer()
        
        # Define the path to the classifier model and the standard scaler
        models_path = Path(__file__).parent
        self.classifier_model_fpath = models_path / "XGB_2025-09-08_23-42_Kappa90.json"
        self.feat_scaler_fpath = models_path /  "XGB_Standard_Scaler_2025-09-08_23-42_Kappa90.bin"
        self.threshold_fpath = models_path / "XGB_OptimalThreshold_2025-09-08_23-42_90.json"
        self.classifier_model = None
        self.feat_scaler = None
        self.classifier_th = 0.5

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

    def set_fs(self, fs):
        self.fs = fs

    def load_models(self):
        # Check if the model and scaler files exist
        assert os.path.isfile(self.classifier_model_fpath), f"Classifier model file not found: {self.classifier_model_fpath}"
        assert os.path.isfile(self.feat_scaler_fpath), f"Standard scaler file not found: {self.feat_scaler_fpath}"
        assert os.path.isfile(self.threshold_fpath), f"Threshold file not found: {self.threshold_fpath}"

        # Load model and scaler
        self.classifier_model = XGBClassifier()
        self.classifier_model.load_model(self.classifier_model_fpath)
        self.feat_scaler = joblib.load(self.feat_scaler_fpath)
        
        # Load optimal threshold from JSON file
        with open(self.threshold_fpath, 'r') as f:
            threshold_data = json.load(f)
            self.classifier_th = threshold_data['optimal_threshold']

        logger.info(f"Models loaded successfully. Optimal threshold: {self.classifier_th}")

        pass

    def fetch_all_channel_events(self, allch_events_fpath:str=None)->pd.DataFrame:

        logger.info(f"\nFetch_all_chann_events")
        print(f"\nFetch_all_chann_events")

        try:
            all_ch_contours_df = pd.read_parquet(allch_events_fpath)
            return all_ch_contours_df
        except Exception as e:
            logger.error(f"Error reading {allch_events_fpath}: {e}")
            return None


    def select_obvious_gs_negative_objs(self, gs_objs_df, fs):

        ################################################
        min_nr_bp_osc = 2 # 3
        hvr_nok = gs_objs_df.hvr.to_numpy() < 10 # 10
        spect_osc_nok = gs_objs_df.nr_oscillations.to_numpy()<4  #<3
        nr_prom_peaks_nok = np.logical_and(gs_objs_df.prom_peaks_nr.to_numpy() < min_nr_bp_osc, gs_objs_df.inverted_prom_peaks_nr.to_numpy() < min_nr_bp_osc)
        valid_obj_sel = np.logical_or.reduce((hvr_nok, spect_osc_nok))

        return valid_obj_sel

    def run_hfo_detection(self, pat_name, allch_events_fpath:str=None, force_recalc:bool=False)->None:
        """        
        Args:
            allch_events_fpath: Path to the all-channel events file

        Returns:
            tuple: (detected_hfo_contours_df, elpi_detections_df, output_file_path)
                   Returns None values if no detections found
        """

        print("Running HFO detection...")
        logger.info("Running HFO detection...")

        # Define output of elpi compatible file containing automatic HFO detections
        elpi_fn = f"{pat_name}_hfo_detections.mat"
        elpi_hfo_marks_fpath = self.output_path / elpi_fn
        if os.path.isfile(elpi_hfo_marks_fpath) and not force_recalc:
            print(f"ELPI HFO marks file already exists: {elpi_hfo_marks_fpath}")
            logger.info(f"ELPI HFO marks file already exists: {elpi_hfo_marks_fpath}")
            return
        
        contour_objs_df = self.fetch_all_channel_events(allch_events_fpath)

        pat_names = contour_objs_df.Patient.unique()
        assert len(pat_names) == 1, "DataFrame must contain exactly one patient"

        # Input validation
        if contour_objs_df is None or len(contour_objs_df) == 0:
            raise ValueError("Input DataFrame is empty or None")
        
        # Get and validate patient name
        pat_names = contour_objs_df.Patient.unique()
        if len(pat_names) != 1:
            raise ValueError(f"DataFrame must contain exactly one patient, found {len(pat_names)}")
        pat_name = pat_names[0]
        
        print(f"Classifier model: {self.classifier_model_fpath.stem}")
        print(f"Processing patient: {pat_name}")

        logger.info(f"Classifier model: {self.classifier_model_fpath.stem}")
        logger.info(f"Processing patient: {pat_name}")

        # Clean DataFrame - remove unnamed columns efficiently
        unnamed_cols = contour_objs_df.columns[contour_objs_df.columns.str.contains('Unnamed')]
        if len(unnamed_cols) > 0:
            contour_objs_df = contour_objs_df.drop(columns=unnamed_cols)
        
        if len(contour_objs_df) == 0:
            raise ValueError("No valid HFO objects found after cleaning")
        
        # Validate required columns for feature engineering
        required_cols = ['bp_sig_pow', 'bkgrnd_sig_pow', 'bp_sig_std', 'bkgrnd_sig_std']
        missing_cols = [col for col in required_cols if col not in contour_objs_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add engineered features with error handling for division by zero
        contour_objs_df = self._add_ratio_features(contour_objs_df)
        
        # Validate that feature selection columns exist
        missing_features = [feat for feat in self.feature_selection if feat not in contour_objs_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features for classification: {missing_features}")
        
        # Pre-filter obviously negative objects
        very_negative_sel = self.select_obvious_gs_negative_objs(contour_objs_df, 0)
        contours_to_detect_df = contour_objs_df[~very_negative_sel].reset_index(drop=True)
        
        if len(contours_to_detect_df) == 0:
            print(f"No valid contours remaining after pre-filtering for {pat_name}")
            return None, None, None
        
        print(f"Processing {len(contours_to_detect_df)} contours after pre-filtering "
              f"(filtered out {np.sum(very_negative_sel)} obvious negatives)")
        logger.info(f"Processing {len(contours_to_detect_df)} contours after pre-filtering "
                    f"(filtered out {np.sum(very_negative_sel)} obvious negatives)")

        # Feature scaling and prediction
        try:
            feature_matrix = contours_to_detect_df[self.feature_selection].to_numpy()
            X_Data = self.feat_scaler.transform(feature_matrix)
            booster = self.classifier_model.get_booster()
            y_pred = booster.predict(xgb.DMatrix(X_Data)).ravel()
            y_pred = y_pred >= self.classifier_th
        except Exception as e:
            raise RuntimeError(f"Error during classification: {str(e)}")
        
        # Get positive detections
        positive_mask = y_pred > 0
        detected_hfo_contours_df = contours_to_detect_df[positive_mask].reset_index(drop=True).copy()
        
        print(f"Detected {len(detected_hfo_contours_df)} HFO events from {len(contours_to_detect_df)} candidates")
        logger.info(f"Detected {len(detected_hfo_contours_df)} HFO events from {len(contours_to_detect_df)} candidates")
        
        if len(detected_hfo_contours_df) == 0:
            print(f"No HFO detections found for {pat_name}")
            logger.info(f"No HFO detections found for {pat_name}")
            return detected_hfo_contours_df, None, None
        
        # Convert to ELPI format
        try:
            elpi_hfo_detections_df = self.contour_objs_to_elpi(detected_hfo_contours_df)
            elpi_hfo_detections_df.loc[:, 'Type'] = "spctHFO"
            # Event type is spctHFO-ntd if the signal was notch filtered to remove power line noise
            elpi_hfo_detections_df.loc[detected_hfo_contours_df.notch_filtered, 'Type'] = "spctHFO-ntd"
        except Exception as e:
            raise RuntimeError(f"Error converting to ELPI format: {str(e)}")
        
        if len(elpi_hfo_detections_df) == 0:
            print(f"No valid ELPI detections generated for {pat_name}")
            logger.info(f"No valid ELPI detections generated for {pat_name}")
            return detected_hfo_contours_df, elpi_hfo_detections_df, None
                
        # Save HFO detections in Elpi format
        #try:
        #    write_elpi_file(elpi_hfo_detections_df, elpi_hfo_marks_fpath)
        #    print(f"Saved {len(elpi_hfo_detections_df)} HFO detections to {elpi_hfo_marks_fpath}")
        #    logger.info(f"Saved {len(elpi_hfo_detections_df)} HFO detections to {elpi_hfo_marks_fpath}")
        #except Exception as e:
        #    raise RuntimeError(f"Error saving ELPI file: {str(e)}")
        
        # Chracterize each elpi HFO event
        try:
            chrtrzd_elpi_marks_fpath = elpi_hfo_marks_fpath.parent / f"{elpi_hfo_marks_fpath.stem}_characterized.mat"
            self.characterizer.characterize_elpi_events(chrtrzd_elpi_marks_fpath, detected_hfo_contours_df, elpi_hfo_detections_df)
        except Exception as e:
            raise RuntimeError(f"Error during ELPI event characterization: {str(e)}")

        return detected_hfo_contours_df, elpi_hfo_detections_df, elpi_hfo_marks_fpath
    
    def _add_ratio_features(self, df):
        """
        Add ratio features between event and background signals with robust error handling.
        
        Args:
            df: DataFrame with signal features
            
        Returns:
            DataFrame with added ratio features
        """
        df = df.copy()  # Avoid modifying original DataFrame
        
        # Define ratio feature mappings
        ratio_features = {
            'EventBkgrndRatio_Power': ('bp_sig_pow', 'bkgrnd_sig_pow'),
            'EventBkgrndRatio_StdDev': ('bp_sig_std', 'bkgrnd_sig_std'),
            'EventBkgrndRatio_Activity': ('bp_sig_activity', 'bkgrnd_sig_activity'),
            'EventBkgrndRatio_Mobility': ('bp_sig_avg_mobility', 'bkgrnd_sig_avg_mobility'),
            'EventBkgrndRatio_Complexity': ('bp_sig_complexity', 'bkgrnd_sig_complexity')
        }
        
        for ratio_name, (numerator_col, denominator_col) in ratio_features.items():
            if numerator_col in df.columns and denominator_col in df.columns:
                # Handle division by zero and invalid values
                denominator = df[denominator_col].to_numpy()
                numerator = df[numerator_col].to_numpy()
                
                # Replace zeros and very small values in denominator to avoid division issues
                safe_denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
                ratio = numerator / safe_denominator
                
                # Handle infinite and NaN values
                ratio = np.where(np.isfinite(ratio), ratio, np.nan)
                df[ratio_name] = ratio
            else:
                print(f"Warning: Cannot create {ratio_name}, missing columns: {numerator_col} or {denominator_col}")
        
        return df

    def contour_objs_to_elpi(self, eoi_feats_df):
        """
        Create the elpi files with the HFO detections
        """
        # Validate input
        pat_names = eoi_feats_df.Patient.unique()
        assert len(pat_names) == 1, "Features DataFrame contains more than one patient."
        pat_name = pat_names[0]
        
        assert self.fs > 1000, "Sampling Rate is under 1000 Hz!"
        
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
        
        # Generate creation time
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
            merged_start_samples = np.round(self.fs * merged_start_ms / 1000).astype(np.int64)
            merged_end_samples = np.round(self.fs * merged_end_ms / 1000).astype(np.int64)
            
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

    def _add_ratio_features(self, df):
        """
        Add ratio features between event and background signals with robust error handling.
        
        Args:
            df: DataFrame with signal features
            
        Returns:
            DataFrame with added ratio features
        """
        df = df.copy()  # Avoid modifying original DataFrame
        
        # Define ratio feature mappings
        ratio_features = {
            'EventBkgrndRatio_Power': ('bp_sig_pow', 'bkgrnd_sig_pow'),
            'EventBkgrndRatio_StdDev': ('bp_sig_std', 'bkgrnd_sig_std'),
            'EventBkgrndRatio_Activity': ('bp_sig_activity', 'bkgrnd_sig_activity'),
            'EventBkgrndRatio_Mobility': ('bp_sig_avg_mobility', 'bkgrnd_sig_avg_mobility'),
            'EventBkgrndRatio_Complexity': ('bp_sig_complexity', 'bkgrnd_sig_complexity')
        }
        
        for ratio_name, (numerator_col, denominator_col) in ratio_features.items():
            if numerator_col in df.columns and denominator_col in df.columns:
                # Handle division by zero and invalid values
                denominator = df[denominator_col].to_numpy()
                numerator = df[numerator_col].to_numpy()
                
                # Replace zeros and very small values in denominator to avoid division issues
                safe_denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
                ratio = numerator / safe_denominator
                
                # Handle infinite and NaN values
                ratio = np.where(np.isfinite(ratio), ratio, np.nan)
                df[ratio_name] = ratio
            else:
                print(f"Warning: Cannot create {ratio_name}, missing columns: {numerator_col} or {denominator_col}")
        
        return df


if __name__ == "__main__":

    pass