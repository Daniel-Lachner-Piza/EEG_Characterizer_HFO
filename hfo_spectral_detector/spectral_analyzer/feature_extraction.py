"""
Feature extraction utilities for HFO characterization.

This module contains functions for extracting various features from HFO events,
including spectral features, bandpass signal features, and background features.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Any, List
from sklearn.preprocessing import minmax_scale

from hfo_spectral_detector.spectral_analyzer.get_bp_features import get_bp_features

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Class for extracting features from HFO events."""
    
    def __init__(self, fs: float):
        """
        Initialize feature extractor.
        
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
    
    def extract_bandpass_features(self, bp_signal: np.ndarray, 
                                 hfo_freqs: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Extract features from bandpass filtered signal.
        
        Args:
            bp_signal: Bandpass filtered signal
            hfo_freqs: Tuple of (min_freq, center_freq, max_freq)
            
        Returns:
            Dictionary of extracted features
        """
        return get_bp_features(fs=self.fs, bp_signal=bp_signal, hfo_freqs=hfo_freqs)
    
    def extract_background_features(self, background_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract features from background signal.
        
        Args:
            background_signal: Background signal (signal excluding the event)
            
        Returns:
            Dictionary of background features
        """
        if len(background_signal) == 0:
            return {
                'bkgrnd_sig_ampl': 0.0,
                'bkgrnd_sig_avg_ampl': 0.0,
                'bkgrnd_sig_std': 0.0,
                'bkgrnd_sig_pow': 0.0,
                'bkgrnd_sig_activity': 0.0,
                'bkgrnd_sig_avg_mobility': 0.0,
                'bkgrnd_sig_complexity': 0.0
            }
        
        background_diff = np.diff(background_signal)
        
        # Calculate Hjorth parameters
        variance = np.var(background_signal)
        mobility = np.sqrt(np.var(background_diff) / variance) if variance > 0 else 0.0
        
        complexity = 0.0
        if len(background_diff) > 1:
            variance_diff2 = np.var(np.diff(background_diff))
            variance_diff = np.var(background_diff)
            if variance_diff > 0 and mobility > 0:
                complexity = np.sqrt(variance_diff2 / variance_diff) / mobility
        
        return {
            'bkgrnd_sig_ampl': np.max(background_signal) - np.min(background_signal),
            'bkgrnd_sig_avg_ampl': np.mean(background_signal),
            'bkgrnd_sig_std': np.std(background_signal),
            'bkgrnd_sig_pow': np.mean(np.power(background_signal, 2)),
            'bkgrnd_sig_activity': variance,
            'bkgrnd_sig_avg_mobility': mobility,
            'bkgrnd_sig_complexity': complexity
        }
    
    def calculate_relative_features(self, event_features: Dict[str, float], 
                                   background_features: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate relative features between event and background.
        
        Args:
            event_features: Features extracted from the event
            background_features: Features extracted from the background
            
        Returns:
            Dictionary of relative features
        """
        relative_features = {}
        
        feature_pairs = [
            ('bp_sig_pow', 'bkgrnd_sig_pow', 'EventBkgrndRatio_Power'),
            ('bp_sig_std', 'bkgrnd_sig_std', 'EventBkgrndRatio_StdDev'),
            ('bp_sig_activity', 'bkgrnd_sig_activity', 'EventBkgrndRatio_Activity'),
            ('bp_sig_avg_mobility', 'bkgrnd_sig_avg_mobility', 'EventBkgrndRatio_Mobility'),
            ('bp_sig_complexity', 'bkgrnd_sig_complexity', 'EventBkgrndRatio_Complexity')
        ]
        
        for event_key, background_key, ratio_key in feature_pairs:
            background_val = background_features.get(background_key, 0.0)
            event_val = event_features.get(event_key, 0.0)
            
            if background_val != 0:
                relative_features[ratio_key] = event_val / background_val
            else:
                relative_features[ratio_key] = np.inf if event_val > 0 else 0.0
                
        return relative_features
    
    def calculate_overlap_features(self, event_start: float, event_end: float,
                                 all_starts: np.ndarray, all_ends: np.ndarray) -> Dict[str, int]:
        """
        Calculate overlap features with other events.
        
        Args:
            event_start: Start time of current event
            event_end: End time of current event
            all_starts: Array of all event start times
            all_ends: Array of all event end times
            
        Returns:
            Dictionary with overlap features
        """
        # Check different types of overlap
        overlap_a = np.logical_and(all_ends >= event_start, all_ends <= event_end)
        overlap_b = np.logical_and(all_starts >= event_start, all_starts <= event_end)
        overlap_c = np.logical_and(event_start >= all_starts, event_end <= all_ends)
        
        # Count overlapping events (excluding self)
        total_overlaps = np.sum(np.logical_or(np.logical_or(overlap_a, overlap_b), overlap_c)) - 1
        
        return {'nr_overlapping_objs': total_overlaps}


class ContourFeatureProcessor:
    """Class for processing contour features and adding metadata."""
    
    def __init__(self):
        """Initialize contour feature processor."""
        pass
    
    def initialize_feature_columns(self, df: pd.DataFrame, n_objects: int) -> pd.DataFrame:
        """
        Initialize all feature columns in the dataframe with default values.
        
        Args:
            df: Input dataframe
            n_objects: Number of objects/events
            
        Returns:
            Dataframe with initialized feature columns
        """
        # Basic validation features
        df['bp_ok'] = [False] * n_objects
        df['visual_valid'] = [False] * n_objects
        
        # Bandpass signal features
        bp_features = [
            'bp_sig_ampl', 'bp_sig_avg_ampl', 'bp_sig_std', 'bp_sig_pow',
            'bp_sig_activity', 'bp_sig_avg_mobility', 'bp_sig_complexity'
        ]
        for feature in bp_features:
            df[feature] = [0.0] * n_objects
        
        # Background signal features
        bg_features = [
            'bkgrnd_sig_ampl', 'bkgrnd_sig_avg_ampl', 'bkgrnd_sig_std', 'bkgrnd_sig_pow',
            'bkgrnd_sig_activity', 'bkgrnd_sig_avg_mobility', 'bkgrnd_sig_complexity'
        ]
        for feature in bg_features:
            df[feature] = [0.0] * n_objects
        
        # HFO-specific features
        hfo_features = [
            'max_hfo_sine_corr', 'all_relevant_peaks_nr', 'all_relevant_peaks_avg_freq',
            'all_relevant_peaks_freq_stddev', 'all_relevant_peaks_amplitude_stability',
            'all_relevant_peaks_prominence_stability', 'prom_peaks_nr', 'prom_peaks_avg_freq',
            'prom_peaks_freqs_stddev', 'prom_peaks_avg_amplitude_stability',
            'prom_peaks_prominence_stability'
        ]
        for feature in hfo_features:
            df[feature] = [0.0] * n_objects
        
        # Inverted signal features
        inverted_features = [
            'inverted_max_hfo_sine_corr', 'inverted_all_relevant_peaks_amplitude_stability',
            'inverted_all_relevant_peaks_prominence_stability', 'inverted_prom_peaks_nr',
            'inverted_prom_peaks_avg_freq', 'inverted_prom_peaks_freqs_stddev',
            'inverted_prom_peaks_avg_amplitude_stability', 'inverted_prom_peaks_prominence_stability'
        ]
        for feature in inverted_features:
            df[feature] = [0.0] * n_objects
        
        # Time-frequency features
        tf_features = ['TF_Complexity', 'NrSpectrumPeaks', 'SumFreqPeakWidths', 'NI']
        for feature in tf_features:
            df[feature] = [0.0] * n_objects
        
        # Overlap features
        df['nr_overlapping_objs'] = [0] * n_objects
        
        return df
    
    def add_metadata_columns(self, df: pd.DataFrame, pat_name: str, channel: str,
                           start_times: List[float], n_objects: int,
                           notch_applied: bool) -> pd.DataFrame:
        """
        Add metadata columns to the dataframe.
        
        Args:
            df: Input dataframe
            pat_name: Patient name
            channel: Channel name
            start_times: List of event start times
            n_objects: Number of objects
            notch_applied: Whether notch filter was applied
            
        Returns:
            Dataframe with metadata columns added
        """
        df.insert(0, "notch_filtered", [notch_applied] * n_objects, False)
        df.insert(0, "an_start_ms", start_times, False)
        df.insert(0, "channel", [str(channel)] * n_objects, False)
        df.insert(0, "Patient", [str(pat_name)] * n_objects, False)
        
        return df
    
    def update_global_times(self, df: pd.DataFrame, global_start_ms: float) -> pd.DataFrame:
        """
        Convert relative times to global times.
        
        Args:
            df: Input dataframe with relative times
            global_start_ms: Global start time in milliseconds
            
        Returns:
            Dataframe with global times
        """
        time_columns = ['center_ms', 'start_ms', 'end_ms']
        for col in time_columns:
            if col in df.columns:
                df[col] = df[col] + global_start_ms
        
        return df
    
    def add_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add relative features comparing event to background.
        
        Args:
            df: Input dataframe with event and background features
            
        Returns:
            Dataframe with relative features added
        """
        if df.shape[0] == 0:
            return df
        
        # Calculate relative features
        df['EventBkgrndRatio_Power'] = df['bp_sig_pow'] / df['bkgrnd_sig_pow']
        df['EventBkgrndRatio_StdDev'] = df['bp_sig_std'] / df['bkgrnd_sig_std']
        df['EventBkgrndRatio_Activity'] = df['bp_sig_activity'] / df['bkgrnd_sig_activity']
        df['EventBkgrndRatio_Mobility'] = df['bp_sig_avg_mobility'] / df['bkgrnd_sig_avg_mobility']
        df['EventBkgrndRatio_Complexity'] = df['bp_sig_complexity'] / df['bkgrnd_sig_complexity']
        
        return df
    
    def validate_and_clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Validate and clean the dataframe by removing NaN values.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (cleaned_dataframe, statistics_dict)
        """
        rows_before = df.shape[0]
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        rows_after = df_clean.shape[0]
        
        # Verify no NaN values remain
        for col_name in df_clean.columns:
            nr_nans = df_clean[col_name].isna().sum()
            if nr_nans > 0:
                logger.warning(f"Column {col_name} still contains {nr_nans} NaN values after cleaning")
        
        stats = {
            'rows_before_cleaning': rows_before,
            'rows_after_cleaning': rows_after,
            'rows_removed': rows_before - rows_after
        }
        
        return df_clean, stats


def safe_memory_intensive_concatenation(dataframe_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Safely concatenate list of dataframes with memory management and error handling.
    
    Args:
        dataframe_list: List of dataframes to concatenate
        
    Returns:
        Concatenated dataframe
    """
    assembled_events = []
    
    for i, df in enumerate(dataframe_list):
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            try:
                # Check for NaN values
                nr_nans = df.isna().sum().sum()
                
                if nr_nans == 0:
                    assembled_events.append(df)
                else:
                    logger.warning(f"Found {nr_nans} NaNs in dataframe {i}, skipping")
                    
            except Exception as e:
                logger.error(f"Error processing dataframe {i}: {e}")
                continue
    
    if len(assembled_events) > 0:
        logger.info(f"Successfully assembled {len(assembled_events)} event dataframes")
        return pd.concat(assembled_events, ignore_index=True)
    else:
        logger.warning("No valid dataframes to concatenate")
        return pd.DataFrame()
