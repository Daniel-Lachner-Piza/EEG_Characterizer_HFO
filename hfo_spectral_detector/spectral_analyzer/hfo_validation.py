"""
HFO validation utilities for event classification.

This module contains functions and classes for validating High Frequency 
Oscillations (HFOs) based on various criteria including spectral, temporal,
and morphological features.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HFOValidationCriteria:
    """Configuration class for HFO validation criteria."""
    
    # Sinusoidal correlation thresholds
    min_sine_correlation: float = 0.75
    high_sine_correlation: float = 0.80
    
    # Peak-related thresholds
    min_prominent_peaks: int = 4
    min_relevant_peaks: int = 4
    max_freq_deviation_ratio: float = 0.2
    max_freq_stddev: float = 15.0
    min_amplitude_stability: float = 0.2
    high_amplitude_stability: float = 0.70
    min_prominence_stability: float = 0.4
    
    # Spectral criteria
    min_hvr: float = 10.0  # Height-to-width ratio
    min_circularity: float = 30.0
    
    # Peak ratio criteria
    max_inverted_peak_diff: int = 2
    max_relevant_to_prominent_ratio: float = 0.5


class HFOValidator:
    """Class for validating HFO events based on multiple criteria."""
    
    def __init__(self, criteria: Optional[HFOValidationCriteria] = None):
        """
        Initialize HFO validator.
        
        Args:
            criteria: Validation criteria configuration
        """
        self.criteria = criteria or HFOValidationCriteria()
    
    def validate_sinusoidal_properties(self, features: Dict[str, Any]) -> bool:
        """
        Validate sinusoidal properties of the HFO.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            bool: True if sinusoidal criteria are met
        """
        sine_corr = features.get('max_hfo_sine_corr', 0.0)
        return sine_corr > self.criteria.min_sine_correlation
    
    def validate_peak_properties(self, features: Dict[str, Any], 
                                hfo_freqs: Tuple[float, float, float],
                                fs: float) -> bool:
        """
        Validate peak-related properties of the HFO.
        
        Args:
            features: Dictionary of extracted features
            hfo_freqs: Tuple of (min_freq, center_freq, max_freq)
            fs: Sampling frequency
            
        Returns:
            bool: True if peak criteria are met
        """
        min_freq, center_freq, max_freq = hfo_freqs
        
        # Check prominent peaks frequency
        prom_peaks_avg_freq = features.get('prom_peaks_avg_freq', 0.0)
        freq_deviation = abs((prom_peaks_avg_freq - center_freq) / center_freq)
        
        prom_peaks_freq_ok = (
            freq_deviation <= self.criteria.max_freq_deviation_ratio and
            prom_peaks_avg_freq >= min_freq and
            prom_peaks_avg_freq <= max_freq and
            prom_peaks_avg_freq <= fs / 3
        )
        
        # Check frequency standard deviation
        prom_peaks_freq_stddev = features.get('prom_peaks_freqs_stddev', float('inf'))
        prom_peaks_freq_stddev_ok = prom_peaks_freq_stddev <= (max_freq - min_freq)
        
        # Check amplitude stability
        prom_peaks_amplitude_stability = features.get('prom_peaks_avg_amplitude_stability', 0.0)
        prom_peaks_amplitude_ok = prom_peaks_amplitude_stability >= self.criteria.min_amplitude_stability
        
        # Check prominence stability
        prom_peaks_prominence_stability = features.get('prom_peaks_prominence_stability', 0.0)
        prom_peaks_stab_ok = prom_peaks_prominence_stability > self.criteria.min_prominence_stability
        
        # Check number of prominent peaks
        nr_prom_peaks = features.get('prom_peaks_nr', 0)
        nr_prom_peaks_ok = nr_prom_peaks >= self.criteria.min_prominent_peaks
        
        return all([
            prom_peaks_freq_ok,
            prom_peaks_freq_stddev_ok,
            prom_peaks_amplitude_ok,
            prom_peaks_stab_ok,
            nr_prom_peaks_ok
        ])
    
    def validate_relevant_peaks(self, features: Dict[str, Any], fs: float) -> bool:
        """
        Validate relevant peaks properties.
        
        Args:
            features: Dictionary of extracted features
            fs: Sampling frequency
            
        Returns:
            bool: True if relevant peaks criteria are met
        """
        all_relevant_peaks_avg_freq = features.get('all_relevant_peaks_avg_freq', 0.0)
        prom_peaks_avg_freq = features.get('prom_peaks_avg_freq', 0.0)
        
        all_relevant_peaks_freq_ok = (
            all_relevant_peaks_avg_freq <= prom_peaks_avg_freq * 2.5 and
            all_relevant_peaks_avg_freq <= fs / 3
        )
        
        return all_relevant_peaks_freq_ok
    
    def validate_spectral_properties(self, spectral_features: Dict[str, Any]) -> bool:
        """
        Validate spectral properties from contour analysis.
        
        Args:
            spectral_features: Dictionary of spectral features
            
        Returns:
            bool: True if spectral criteria are met
        """
        hvr = spectral_features.get('hvr', 0.0)
        circularity = spectral_features.get('circularity', 0.0)
        
        return hvr > self.criteria.min_hvr and circularity > self.criteria.min_circularity
    
    def validate_inverted_signal_consistency(self, normal_features: Dict[str, Any],
                                           inverted_features: Dict[str, Any]) -> bool:
        """
        Validate consistency between normal and inverted signal analysis.
        
        Args:
            normal_features: Features from normal signal
            inverted_features: Features from inverted signal
            
        Returns:
            bool: True if consistency criteria are met
        """
        normal_peaks = normal_features.get('prom_peaks_nr', 0)
        inverted_peaks = inverted_features.get('prom_peaks_nr', 0)
        
        peak_diff = abs(normal_peaks - inverted_peaks)
        return peak_diff <= self.criteria.max_inverted_peak_diff
    
    def validate_comprehensive_hfo(self, features: Dict[str, Any],
                                  spectral_features: Dict[str, Any],
                                  inverted_features: Dict[str, Any],
                                  hfo_freqs: Tuple[float, float, float],
                                  fs: float) -> Tuple[bool, Dict[str, bool]]:
        """
        Perform comprehensive HFO validation using all criteria.
        
        Args:
            features: Bandpass signal features
            spectral_features: Spectral analysis features
            inverted_features: Inverted signal features
            hfo_freqs: Tuple of (min_freq, center_freq, max_freq)
            fs: Sampling frequency
            
        Returns:
            Tuple of (overall_validity, individual_validations)
        """
        # Individual validation checks
        validations = {
            'sinusoidal_valid': self.validate_sinusoidal_properties(features),
            'peak_properties_valid': self.validate_peak_properties(features, hfo_freqs, fs),
            'relevant_peaks_valid': self.validate_relevant_peaks(features, fs),
            'spectral_properties_valid': self.validate_spectral_properties(spectral_features),
            'inverted_consistency_valid': self.validate_inverted_signal_consistency(features, inverted_features)
        }
        
        # Overall validation (all criteria must be met)
        overall_valid = all(validations.values())
        
        return overall_valid, validations
    
    def validate_high_confidence_hfo(self, features: Dict[str, Any],
                                   spectral_features: Dict[str, Any],
                                   inverted_features: Dict[str, Any]) -> bool:
        """
        Validate HFO using high-confidence criteria.
        
        Args:
            features: Bandpass signal features
            spectral_features: Spectral analysis features
            inverted_features: Inverted signal features
            
        Returns:
            bool: True if high-confidence criteria are met
        """
        # High-confidence criteria
        high_sine_corr = features.get('max_hfo_sine_corr', 0.0) > self.criteria.high_sine_correlation
        good_spectral = (
            spectral_features.get('hvr', 0.0) > self.criteria.min_hvr and
            spectral_features.get('circularity', 0.0) > self.criteria.min_circularity
        )
        sufficient_peaks = features.get('prom_peaks_nr', 0) >= self.criteria.min_prominent_peaks
        good_freq_stability = features.get('prom_peaks_freqs_stddev', float('inf')) <= self.criteria.max_freq_stddev
        high_amplitude_stability = (
            features.get('prom_peaks_avg_amplitude_stability', 0.0) >= self.criteria.high_amplitude_stability
        )
        
        # Peak ratio check
        all_relevant_peaks = features.get('all_relevant_peaks_nr', 0)
        prom_peaks = features.get('prom_peaks_nr', 1)  # Avoid division by zero
        peak_ratio_ok = (all_relevant_peaks - prom_peaks) < int(prom_peaks * self.criteria.max_relevant_to_prominent_ratio)
        
        # Inverted signal consistency
        inverted_consistent = self.validate_inverted_signal_consistency(features, inverted_features)
        
        return all([
            high_sine_corr,
            good_spectral,
            sufficient_peaks,
            good_freq_stability,
            high_amplitude_stability,
            peak_ratio_ok,
            inverted_consistent
        ])


class HFOClassifier:
    """Class for classifying HFO events based on validation results."""
    
    def __init__(self, validator: Optional[HFOValidator] = None):
        """
        Initialize HFO classifier.
        
        Args:
            validator: HFO validator instance
        """
        self.validator = validator or HFOValidator()
    
    def classify_event(self, features: Dict[str, Any],
                      spectral_features: Dict[str, Any],
                      inverted_features: Dict[str, Any],
                      hfo_freqs: Tuple[float, float, float],
                      fs: float) -> Dict[str, Any]:
        """
        Classify a single HFO event.
        
        Args:
            features: Bandpass signal features
            spectral_features: Spectral analysis features
            inverted_features: Inverted signal features
            hfo_freqs: Tuple of (min_freq, center_freq, max_freq)
            fs: Sampling frequency
            
        Returns:
            Dictionary with classification results
        """
        # Comprehensive validation
        comprehensive_valid, individual_validations = self.validator.validate_comprehensive_hfo(
            features, spectral_features, inverted_features, hfo_freqs, fs
        )
        
        # High-confidence validation
        high_confidence_valid = self.validator.validate_high_confidence_hfo(
            features, spectral_features, inverted_features
        )
        
        # Spectral validity (from original spectral analysis)
        spectral_valid = spectral_features.get('spect_ok', False)
        
        # Final classification combining multiple criteria
        final_classification = high_confidence_valid and spectral_valid
        
        return {
            'bp_ok': final_classification,
            'comprehensive_valid': comprehensive_valid,
            'high_confidence_valid': high_confidence_valid,
            'spectral_valid': spectral_valid,
            **individual_validations
        }
    
    def classify_events_batch(self, df: pd.DataFrame, fs: float) -> pd.DataFrame:
        """
        Classify multiple HFO events in batch.
        
        Args:
            df: DataFrame containing event features
            fs: Sampling frequency
            
        Returns:
            DataFrame with classification results added
        """
        n_events = len(df)
        
        if n_events == 0:
            return df
        
        # Initialize classification columns
        df['bp_ok'] = [False] * n_events
        df['comprehensive_valid'] = [False] * n_events
        df['high_confidence_valid'] = [False] * n_events
        df['spectral_valid'] = [False] * n_events
        
        # Classify each event
        for idx in range(n_events):
            # Extract features for current event
            features = {col: df.at[idx, col] for col in df.columns if col.startswith(('max_hfo', 'all_relevant', 'prom_peaks'))}
            spectral_features = {col: df.at[idx, col] for col in df.columns if col in ['hvr', 'circularity', 'spect_ok']}
            inverted_features = {col: df.at[idx, col] for col in df.columns if col.startswith('inverted_')}
            
            # Get HFO frequencies
            hfo_freqs = (
                df.at[idx, 'freq_min_Hz'],
                df.at[idx, 'freq_centroid_Hz'],
                df.at[idx, 'freq_max_Hz']
            )
            
            # Classify event
            classification = self.classify_event(features, spectral_features, inverted_features, hfo_freqs, fs)
            
            # Update dataframe
            for key, value in classification.items():
                if key in df.columns:
                    df.at[idx, key] = value
        
        return df


def calculate_classification_metrics(df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate classification metrics from validation results.
    
    Args:
        df: DataFrame with classification results
        
    Returns:
        Dictionary with classification metrics
    """
    if len(df) == 0:
        return {'tp_cnt': 0, 'fp_cnt': 0, 'tn_cnt': 0, 'fn_cnt': 0}
    
    # Assuming 'visual_valid' column exists for ground truth
    predicted_positive = df.get('bp_ok', pd.Series([False] * len(df))) & df.get('spect_ok', pd.Series([False] * len(df)))
    actual_positive = df.get('visual_valid', pd.Series([False] * len(df)))
    
    tp_cnt = sum(predicted_positive & actual_positive)
    fp_cnt = sum(predicted_positive & ~actual_positive)
    tn_cnt = sum(~predicted_positive & ~actual_positive)
    fn_cnt = sum(~predicted_positive & actual_positive)
    
    return {
        'tp_cnt': tp_cnt,
        'fp_cnt': fp_cnt,
        'tn_cnt': tn_cnt,
        'fn_cnt': fn_cnt
    }
