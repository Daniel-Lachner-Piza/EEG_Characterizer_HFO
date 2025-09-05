"""
Characterization module for HFO prediction pipeline.

This module contains the HFO_Characterizer class responsible for characterizing
ELPI events based on HFO contour features with temporal overlap analysis.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

from hfo_spectral_detector.elpi.elpi_interface import write_elpi_file

logger = logging.getLogger(__name__)


class HFO_Characterizer:
    """
    Class for characterizing ELPI events based on HFO contour features.
    
    This class handles the temporal overlap analysis between ELPI events and HFO contours,
    and transfers characterization features from the best-matching HFO contours to ELPI events.
    """
    
    def __init__(self):
        """Initialize the HFO_Characterizer."""
        pass
    
    def _get_characterization_features(self):
        """
        Get the list of characterization features to extract from HFO contours.
        
        Returns:
            list: List of feature names for characterization
        """
        return [
            'hvr', 'circularity', 'area', 'nr_oscillations',
            'bp_sig_ampl', 'bp_sig_pow', 'bp_sig_std',
            'bkgrnd_sig_pow', 'bkgrnd_sig_std',
            'max_hfo_sine_corr', 'all_relevant_peaks_nr', 
            'all_relevant_peaks_avg_freq', 'prom_peaks_nr', 
            'prom_peaks_avg_freq', 'EventBkgrndRatio_Power'
        ]
    
    def _initialize_characterization_dataframe(self, elpi_events_df, hfo_contours_df):
        """
        Initialize the characterization DataFrame with NaN values for feature columns.
        
        Args:
            elpi_events_df: DataFrame containing ELPI event annotations
            hfo_contours_df: DataFrame containing HFO contour annotations
            
        Returns:
            pd.DataFrame: Initialized DataFrame with characterization columns
        """
        characterized_elpi_df = elpi_events_df.copy()
        characterization_features = self._get_characterization_features()
        
        # Initialize characterization columns with NaN
        for feature in characterization_features:
            if feature in hfo_contours_df.columns:
                characterized_elpi_df[feature] = np.nan
        
        # Add metadata columns
        characterized_elpi_df['overlap_ratio'] = np.nan
        characterized_elpi_df['best_contour_idx'] = -1
        
        return characterized_elpi_df
    
    def _calculate_temporal_overlap(self, elpi_start_ms, elpi_end_ms, contour_start_ms, contour_end_ms):
        """
        Calculate temporal overlap ratio between an ELPI event and HFO contour.
        
        Args:
            elpi_start_ms: ELPI event start time in milliseconds
            elpi_end_ms: ELPI event end time in milliseconds
            contour_start_ms: HFO contour start time in milliseconds
            contour_end_ms: HFO contour end time in milliseconds
            
        Returns:
            float: Overlap ratio (0-1), where 1 means complete overlap
        """
        # Calculate overlap boundaries
        overlap_start = max(elpi_start_ms, contour_start_ms)
        overlap_end = min(elpi_end_ms, contour_end_ms)
        
        if overlap_end <= overlap_start:  # No overlap
            return 0.0
        
        # Calculate durations
        overlap_duration = overlap_end - overlap_start
        elpi_duration = elpi_end_ms - elpi_start_ms
        contour_duration = contour_end_ms - contour_start_ms
        
        # Calculate overlap ratio as fraction of smaller event
        min_duration = min(elpi_duration, contour_duration)
        return overlap_duration / min_duration if min_duration > 0 else 0.0
    
    def _find_best_matching_contour(self, elpi_row, hfo_contours_df):
        """
        Find the HFO contour with the highest temporal overlap for a given ELPI event.
        
        Args:
            elpi_row: Single row from ELPI events DataFrame
            hfo_contours_df: DataFrame containing all HFO contour annotations
            
        Returns:
            tuple: (best_overlap_ratio, best_contour_row) or (0, None) if no overlap found
        """
        channel = elpi_row['Channel'].lower()
        elpi_start_ms = elpi_row['StartSec'] * 1000
        elpi_end_ms = elpi_row['EndSec'] * 1000
        
        # Find matching HFO contours in the same channel
        channel_contours = hfo_contours_df[
            hfo_contours_df['channel'].str.lower() == channel
        ].reset_index(drop=True)
        
        if len(channel_contours) == 0:
            return 0.0, None
            
        best_overlap_ratio = 0.0
        best_contour_row = None
        
        # Calculate overlap with each contour in the channel
        for _, contour_row in channel_contours.iterrows():
            overlap_ratio = self._calculate_temporal_overlap(
                elpi_start_ms, elpi_end_ms,
                contour_row['start_ms'], contour_row['end_ms']
            )
            
            if overlap_ratio > best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
                best_contour_row = contour_row
        
        return best_overlap_ratio, best_contour_row
    
    def _copy_characterization_features(self, target_df, target_idx, source_row, overlap_ratio, contour_idx):
        """
        Copy characterization features from HFO contour to ELPI event.
        
        Args:
            target_df: Target DataFrame to modify
            target_idx: Index in target DataFrame
            source_row: Source row containing features to copy
            overlap_ratio: Calculated overlap ratio
            contour_idx: Index of the best matching contour
        """
        characterization_features = self._get_characterization_features()
        
        # Set metadata
        target_df.loc[target_idx, 'overlap_ratio'] = overlap_ratio
        target_df.loc[target_idx, 'best_contour_idx'] = contour_idx
        
        # Copy characterization features
        for feature in characterization_features:
            if feature in source_row.index and pd.notna(source_row[feature]):
                target_df.loc[target_idx, feature] = source_row[feature]
    
    def _save_characterized_events(self, characterized_df, characterization_fpath):
        """
        Save characterized ELPI events to file with error handling.
        
        Args:
            characterized_df: DataFrame with characterized events
            characterization_fpath: Path to save the characterized events
            
        Raises:
            RuntimeError: If saving fails
        """
        try:
            write_elpi_file(characterized_df, characterization_fpath)
            print(f"Saved characterized ELPI events to {characterization_fpath}")
            logger.info(f"Saved characterized ELPI events to {characterization_fpath}")
        except Exception as e:
            error_msg = f"Error saving characterized ELPI file: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def characterize_elpi_events(self, characterization_fpath, hfo_contours_df, elpi_events_df):
        """
        Characterize ELPI events based on contour objects classified as HFO.
        The HFO contour with the highest overlap with each ELPI event is used for characterization.

        Args:
            characterization_fpath: Path to the output characterization file
            hfo_contours_df: DataFrame containing HFO contour annotations
            elpi_events_df: DataFrame containing ELPI event annotations
            
        Returns:
            pd.DataFrame: Characterized ELPI events DataFrame
        """
        # Input validation
        if len(hfo_contours_df) == 0 or len(elpi_events_df) == 0:
            print("Warning: No HFO contours or ELPI events to characterize")
            logger.warning("No HFO contours or ELPI events to characterize")
            return elpi_events_df.copy()
        
        print(f"Characterizing {len(elpi_events_df)} ELPI events using {len(hfo_contours_df)} HFO contours")
        logger.info(f"Characterizing {len(elpi_events_df)} ELPI events using {len(hfo_contours_df)} HFO contours")
        
        # Initialize output DataFrame
        characterized_elpi_df = self._initialize_characterization_dataframe(elpi_events_df, hfo_contours_df)
        
        # Process each ELPI event
        for elpi_idx, elpi_row in characterized_elpi_df.iterrows():
            best_overlap_ratio, best_contour_row = self._find_best_matching_contour(elpi_row, hfo_contours_df)
            
            # If we found a good overlap, add characterization features
            if best_overlap_ratio > 0 and best_contour_row is not None:
                self._copy_characterization_features(
                    characterized_elpi_df, elpi_idx, best_contour_row, 
                    best_overlap_ratio, best_contour_row.name
                )
        
        # Report results
        characterized_count = characterized_elpi_df['overlap_ratio'].notna().sum()
        print(f"Successfully characterized {characterized_count}/{len(elpi_events_df)} ELPI events")
        logger.info(f"Successfully characterized {characterized_count}/{len(elpi_events_df)} ELPI events")
        
        # Save results
        self._save_characterized_events(characterized_elpi_df, characterization_fpath)
        
        return characterized_elpi_df
