"""
Signal processing utilities for HFO characterization.

This module contains functions for filtering, preprocessing, and signal 
transformations used in High Frequency Oscillation (HFO) analysis.
"""

import numpy as np
import logging
from scipy.signal import firwin
from scipy.ndimage import convolve1d
from sklearn.preprocessing import minmax_scale
from typing import Tuple, List, Optional

from hfo_spectral_detector.dsp.cwt import dcmwt

logger = logging.getLogger(__name__)


class SignalProcessor:
    """Class for handling signal processing operations in HFO analysis."""
    
    def __init__(self, fs: float):
        """
        Initialize signal processor.
        
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
    
    def detect_power_line_noise(self, signal: np.ndarray, power_line_freq: float = 60) -> bool:
        """
        Detect presence of power line noise in the signal.
        
        Args:
            signal: Input signal
            power_line_freq: Expected power line frequency (50 or 60 Hz)
            
        Returns:
            bool: True if power line noise is detected
        """
        # Define frequencies to analyze for power line noise
        cmwt_freqs_emi, dcwt_emi = dcmwt(signal, self.fs, list(range(30, 120, 10)), nr_cycles=6)
        
        # Check if the maximum power is at the power line frequency
        max_power_freq = cmwt_freqs_emi[np.argmax(np.mean(dcwt_emi, axis=1))]
        return max_power_freq == power_line_freq
    
    def apply_notch_filter(self, signal: np.ndarray, notch_freqs: List[float], 
                          notch_width: float = 10, ntaps: int = 513) -> np.ndarray:
        """
        Apply notch filter to remove specific frequency components.
        
        Args:
            signal: Input signal
            notch_freqs: List of frequencies to notch out
            notch_width: Width of the notch in Hz
            ntaps: Number of filter taps
            
        Returns:
            Filtered signal
        """
        filtered_signal = signal.copy()
        
        logger.info(f"Applying Notch Filters: {notch_freqs} Hz")
        
        for nf in notch_freqs:
            # Create notch filter coefficients
            notch_coeffs = firwin(
                ntaps, 
                [int(nf - notch_width), int(nf + notch_width)], 
                width=None, 
                window='hamming', 
                pass_zero='bandstop', 
                fs=self.fs
            )
            
            # Apply filter forward and backward to eliminate phase distortion
            filtered_signal = convolve1d(np.flip(filtered_signal), notch_coeffs)
            filtered_signal = convolve1d(np.flip(filtered_signal), notch_coeffs)
            
        return filtered_signal
    
    def apply_bandpass_filter(self, signal: np.ndarray, 
                             low_freq: float = 80, 
                             high_freq: Optional[float] = None,
                             ntaps: int = 256) -> np.ndarray:
        """
        Apply bandpass filter to signal.
        
        Args:
            signal: Input signal
            low_freq: Low cutoff frequency in Hz
            high_freq: High cutoff frequency in Hz (defaults to fs/3)
            ntaps: Number of filter taps
            
        Returns:
            Bandpass filtered signal
        """
        if high_freq is None:
            high_freq = int(np.round(self.fs / 3))
            
        # Create bandpass filter coefficients
        bp_filter_coeffs = firwin(
            ntaps, 
            [low_freq, high_freq], 
            width=None, 
            window='hamming', 
            pass_zero='bandpass', 
            fs=self.fs
        )
        
        # Apply filter forward and backward to eliminate phase distortion
        bp_signal = convolve1d(np.flip(signal), bp_filter_coeffs)
        bp_signal = convolve1d(np.flip(bp_signal), bp_filter_coeffs)
        
        return bp_signal
    
    def preprocess_signal(self, signal: np.ndarray, 
                         power_line_freq: float = 60,
                         apply_notch: bool = True,
                         bp_low: float = 80,
                         bp_high: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Complete signal preprocessing pipeline.
        
        Args:
            signal: Input raw signal
            power_line_freq: Power line frequency to filter out
            apply_notch: Whether to apply notch filtering
            bp_low: Bandpass low cutoff frequency
            bp_high: Bandpass high cutoff frequency
            
        Returns:
            Tuple of (processed_signal, bandpass_signal, notch_applied)
        """
        processed_signal = signal.copy()
        notch_applied = False
        
        # Detect and filter power line noise if requested
        if apply_notch and power_line_freq in [50, 60]:
            has_noise = self.detect_power_line_noise(signal, power_line_freq)
            if has_noise:
                processed_signal = self.apply_notch_filter(processed_signal, [power_line_freq])
                notch_applied = True
        
        # Apply bandpass filter
        bp_signal = self.apply_bandpass_filter(processed_signal, bp_low, bp_high)
        
        return processed_signal, bp_signal, notch_applied
    
    def compute_wavelet_transform(self, signal: np.ndarray, 
                                 freq_range: Tuple[int, int] = (80, 500),
                                 freq_step: int = 5,
                                 nr_cycles: int = 6) -> Tuple[List[float], np.ndarray]:
        """
        Compute continuous wavelet transform of the signal.
        
        Args:
            signal: Input signal
            freq_range: Frequency range tuple (min_freq, max_freq)
            freq_step: Step size for frequency sampling
            nr_cycles: Number of wavelet cycles
            
        Returns:
            Tuple of (frequencies, wavelet_coefficients)
        """
        # Define frequencies to analyze with wavelets
        dcmwt_freqs = list(range(freq_range[0], freq_range[1] + 1, freq_step))
        dcmwt_freqs = [float(f) for f in dcmwt_freqs]
        
        # Compute wavelet transform
        cmwt_freqs, dcwt = dcmwt(signal, self.fs, dcmwt_freqs, nr_cycles=nr_cycles)
        
        return cmwt_freqs, dcwt
    
    def normalize_signal(self, signal: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize signal using specified method.
        
        Args:
            signal: Input signal
            method: Normalization method ('minmax', 'zscore')
            
        Returns:
            Normalized signal
        """
        if method == 'minmax':
            return minmax_scale(signal)
        elif method == 'zscore':
            return (signal - np.mean(signal)) / np.std(signal)
        else:
            raise ValueError(f"Unknown normalization method: {method}")


def create_spectrogram_image(dcwt: np.ndarray) -> np.ndarray:
    """
    Convert wavelet coefficients to BGR image format for computer vision analysis.
    
    Args:
        dcwt: Wavelet coefficients array
        
    Returns:
        BGR image array
    """
    import matplotlib as mpl
    
    # Convert to RGBA using matplotlib colormap
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=np.min(dcwt), vmax=np.max(dcwt), clip=False)
    colors = cmap(norm(dcwt))
    
    # Convert to uint8 and extract RGB channels
    spect_int = (colors * 255).astype('uint8')[:, :, 0:3]
    
    # Convert RGB to BGR for OpenCV compatibility
    spect_bgr = spect_int[:, :, 0:3].copy()
    spect_bgr[:, :, 0] = spect_int[:, :, 2]  # B
    spect_bgr[:, :, 1] = spect_int[:, :, 1]  # G
    spect_bgr[:, :, 2] = spect_int[:, :, 0]  # R
    
    # Flip vertically to match frequency axis orientation
    spect_bgr = np.flipud(spect_bgr)
    
    return spect_bgr
