"""
Visualization utilities for HFO analysis.

This module contains functions for plotting and visualizing HFO events,
spectrograms, signals, and analysis results.
"""

import os
import time
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

logger = logging.getLogger(__name__)


class HFOVisualizer:
    """Class for visualizing HFO analysis results."""
    
    def __init__(self, screen_size: Tuple[float, float] = (20.0, 11.0)):
        """
        Initialize HFO visualizer.
        
        Args:
            screen_size: Figure size as (width, height)
        """
        self.screen_size = screen_size
        self.signal_linewidth = 1
        
    def plot_hfo_analysis(self, 
                         pat_name: str,
                         mtg: str,
                         an_time: np.ndarray,
                         an_raw_signal: np.ndarray,
                         an_bp_signal: np.ndarray,
                         objects_image: np.ndarray,
                         dcmwt_freqs: List[float],
                         contours_df: pd.DataFrame,
                         fs: float,
                         out_path: Path,
                         fig_title: str,
                         all_relevant_peaks_data: List[Dict],
                         contour_plotting_data: List[Dict],
                         classification_metrics: Dict[str, int]) -> str:
        """
        Create comprehensive HFO analysis plot.
        
        Args:
            pat_name: Patient name
            mtg: Channel/montage name
            an_time: Time array for the analysis window
            an_raw_signal: Raw signal
            an_bp_signal: Bandpass filtered signal
            objects_image: Detected objects image
            dcmwt_freqs: Wavelet frequency array
            contours_df: DataFrame with contour features
            fs: Sampling frequency
            out_path: Output directory path
            fig_title: Figure title
            all_relevant_peaks_data: Peak detection data
            contour_plotting_data: Contour plotting data
            classification_metrics: TP/FP/TN/FN counts
            
        Returns:
            Path to saved figure
        """
        start_time = time.time()
        
        # Create figure with subplots
        fig = plt.figure(2)
        fig.set_figwidth(self.screen_size[0])
        fig.set_figheight(self.screen_size[1])
        
        # Resize objects image for visualization
        new_size = (1000, 250)
        new_obj = cv2.resize(cv2.cvtColor(objects_image, cv2.COLOR_BGR2RGB), 
                           new_size, interpolation=cv2.INTER_LINEAR)
        
        # Create subplot grid
        ax1 = plt.subplot2grid(shape=(6, 10), loc=(0, 3), colspan=7, rowspan=1)
        ax2 = plt.subplot2grid(shape=(6, 10), loc=(1, 3), colspan=7, rowspan=1)
        ax3 = plt.subplot2grid(shape=(6, 10), loc=(2, 3), colspan=7, rowspan=1)
        ax4 = plt.subplot2grid(shape=(6, 10), loc=(3, 3), colspan=7, rowspan=3)
        ax5 = plt.subplot2grid(shape=(6, 10), loc=(0, 0), colspan=3, rowspan=6)
        
        # Plot raw signal
        self._plot_raw_signal(ax1, an_time, an_raw_signal, contours_df)
        
        # Plot bandpass signal
        self._plot_bandpass_signal(ax2, an_time, an_bp_signal)
        
        # Plot detailed bandpass signal with peaks
        self._plot_detailed_bandpass_signal(ax3, an_time, an_bp_signal, 
                                          all_relevant_peaks_data, contour_plotting_data)
        
        # Plot spectrogram with objects
        self._plot_spectrogram_objects(ax4, new_obj, new_size, dcmwt_freqs, an_time, fs)
        
        # Create legend
        legend_strings, legend_colors = self._create_legend_data(contours_df)
        self._plot_legend(ax5, an_time, an_bp_signal, legend_strings, legend_colors)
        
        # Set title
        plt.suptitle(f"{pat_name}\n{mtg}")
        
        # Save figure
        fig_filepath = self._save_figure(out_path, pat_name, fig_title, classification_metrics)
        
        plt.close(2)
        
        logger.debug(f"HFO Objects Plot total_time={time.time() - start_time}")
        
        return str(fig_filepath)
    
    def _plot_raw_signal(self, ax: plt.Axes, time: np.ndarray, 
                        signal: np.ndarray, contours_df: pd.DataFrame):
        """Plot raw signal with visual validation markers."""
        ax.plot(time, signal, '-k', linewidth=self.signal_linewidth)
        ax.set_xlim((np.min(time), np.max(time)))
        ax.set_title('Raw')
        
        # Add visual validation rectangles
        for idx in range(len(contours_df)):
            if contours_df.at[idx, 'visual_valid']:
                start_time = time[0]  # This would need proper mapping
                end_time = time[-1]   # This would need proper mapping
                rect = patches.Rectangle(
                    (start_time, np.min(signal)), 
                    end_time - start_time, 
                    np.max(signal) - np.min(signal),
                    linewidth=1, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
    
    def _plot_bandpass_signal(self, ax: plt.Axes, time: np.ndarray, signal: np.ndarray):
        """Plot bandpass filtered signal."""
        ax.plot(time, signal, '-k', linewidth=self.signal_linewidth)
        ax.set_xlim((np.min(time), np.max(time)))
        
        # Set y-limits with some padding
        signal_range = np.max(signal) - np.min(signal)
        y_center = np.mean(signal)
        ax.set_ylim((y_center - signal_range, y_center + signal_range))
        ax.set_title('Band-passed')
    
    def _plot_detailed_bandpass_signal(self, ax: plt.Axes, time: np.ndarray, 
                                     signal: np.ndarray, peaks_data: List[Dict],
                                     contour_data: List[Dict]):
        """Plot detailed bandpass signal with peak annotations."""
        ax.plot(time, signal, '-k', linewidth=self.signal_linewidth)
        ax.set_xlim((np.min(time), np.max(time)))
        
        # Set y-limits
        signal_range = np.max(signal) - np.min(signal)
        y_center = np.mean(signal)
        ax.set_ylim((y_center - signal_range, y_center + signal_range))
        
        # Plot peak annotations for each contour
        for i, (peaks, contour) in enumerate(zip(peaks_data, contour_data)):
            contour_time = contour['time']
            contour_signal = contour['signal']
            
            # Plot all relevant peaks
            if 'all_relevant_peaks_loc' in peaks:
                all_peaks_loc = peaks['all_relevant_peaks_loc']
                ax.plot(contour_time[all_peaks_loc], contour_signal[all_peaks_loc], 
                       'o', markerfacecolor="None", markeredgecolor='cyan', color="cyan")
                
                # Plot threshold for all peaks
                avg_ampl = peaks.get('contour_obj_avg_ampl', 0)
                ax.plot(contour_time, np.zeros_like(contour_signal) + avg_ampl, 
                       linestyle='--', color="cyan")
            
            # Plot prominent peaks
            if 'prom_peaks_loc' in peaks:
                prom_peaks_loc = peaks['prom_peaks_loc']
                ax.plot(contour_time[prom_peaks_loc], contour_signal[prom_peaks_loc], 
                       "x", color="orange")
                
                # Plot threshold for prominent peaks
                height_th = peaks.get('prom_peaks_height_th', 0)
                ax.plot(contour_time, np.zeros_like(contour_signal) + height_th, 
                       "--", color="orange")
            
            # Plot corrected contour signal with color coding
            corrected_time = contour.get('corrected_time', contour_time)
            corrected_signal = contour.get('corrected_signal', contour_signal)
            
            # Determine color based on validation
            color = self._get_contour_color(contour.get('bp_ok', False), 
                                          contour.get('spect_ok', False))
            
            ax.plot(corrected_time, corrected_signal, '-', 
                   color=color, linewidth=self.signal_linewidth)
    
    def _plot_spectrogram_objects(self, ax: plt.Axes, objects_image: np.ndarray,
                                image_size: Tuple[int, int], freqs: List[float],
                                time: np.ndarray, fs: float):
        """Plot spectrogram with detected objects."""
        ax.imshow(objects_image)
        ax.set_title('Objects')
        
        # Set frequency ticks
        y_ticks_val = np.linspace(0, image_size[1], num=10, endpoint=True, dtype=int)
        y_ticks_labels = np.flip(np.linspace(freqs[0], freqs[-1], num=10, 
                                           endpoint=True, dtype=int))
        ax.set_yticks(y_ticks_val, y_ticks_labels)
        
        # Set time ticks
        x_ticks_vals = np.linspace(0, image_size[0], num=11, endpoint=True, dtype=int)
        x_ticks_labels = np.round(
            np.linspace(0, image_size[0], num=11, endpoint=True, dtype=int) / fs + np.min(time), 1
        )
        ax.set_xticks(x_ticks_vals, x_ticks_labels)
    
    def _plot_legend(self, ax: plt.Axes, time: np.ndarray, signal: np.ndarray,
                    legend_strings: List[str], legend_colors: List[str]):
        """Plot legend with feature information."""
        # Create placeholder plots for legend
        ax.plot(time, signal, '-', color='w', linewidth=self.signal_linewidth)
        for i in range(len(legend_strings) - 1):  # Skip header
            ax.plot(time, signal, '-', color='w', linewidth=self.signal_linewidth)
        
        ax.legend(legend_strings, loc="upper left", frameon=True, labelcolor=legend_colors)
        ax.axis('off')
    
    def _get_contour_color(self, bp_ok: bool, spect_ok: bool) -> str:
        """Get color for contour based on validation status."""
        if bp_ok:
            return "lime" if spect_ok else "dodgerblue"
        return "lightcoral"
    
    def _create_legend_data(self, contours_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Create legend strings and colors for each contour object."""
        legend_strings = []
        legend_colors = []
        
        # Header
        header = "idx (sin) (rp-nr, rp-f) (pp-nr, pp-f) (pp-fstd, pp-stab) (tf-clx, tf-pnr, tf-wsm) (nr_ovo, hclx)"
        legend_strings.append(header)
        legend_colors.append('k')
        
        # Individual objects
        for idx in range(len(contours_df)):
            # Get features for legend
            sine_corr = contours_df.at[idx, 'max_hfo_sine_corr']
            all_peaks_nr = contours_df.at[idx, 'all_relevant_peaks_nr']
            all_peaks_freq = contours_df.at[idx, 'all_relevant_peaks_avg_freq']
            prom_peaks_nr = contours_df.at[idx, 'prom_peaks_nr']
            prom_peaks_freq = contours_df.at[idx, 'prom_peaks_avg_freq']
            prom_freq_std = contours_df.at[idx, 'prom_peaks_freqs_stddev']
            prom_stab = contours_df.at[idx, 'prom_peaks_avg_amplitude_stability']
            tf_complexity = contours_df.at[idx, 'TF_Complexity']
            nr_spectrum_peaks = contours_df.at[idx, 'NrSpectrumPeaks']
            sum_freq_widths = contours_df.at[idx, 'SumFreqPeakWidths']
            nr_overlapping = contours_df.at[idx, 'nr_overlapping_objs']
            complexity = contours_df.at[idx, 'bp_sig_complexity']
            
            # Create legend string
            legend_str = (
                f"{idx}. ({sine_corr:.2f})"
                f"({all_peaks_nr}, {all_peaks_freq:.2f})"
                f"({prom_peaks_nr:.0f}, {prom_peaks_freq:.2f})"
                f"({prom_freq_std:.2f}, {prom_stab:.2f})"
                f"({tf_complexity:.2f}, {nr_spectrum_peaks:.0f}, {sum_freq_widths:.2f})"
                f"({nr_overlapping:.0f}, {complexity:.2f})"
            )
            
            legend_strings.append(legend_str)
            
            # Determine color
            bp_ok = contours_df.at[idx, 'bp_ok']
            spect_ok = contours_df.at[idx, 'spect_ok'] if 'spect_ok' in contours_df.columns else False
            color = self._get_contour_color(bp_ok, spect_ok)
            legend_colors.append(color if bp_ok else 'k')
        
        return legend_strings, legend_colors
    
    def _save_figure(self, out_path: Path, pat_name: str, fig_title: str,
                    metrics: Dict[str, int]) -> Path:
        """Save figure to appropriate directory based on classification results."""
        tp_cnt = metrics.get('tp_cnt', 0)
        fp_cnt = metrics.get('fp_cnt', 0)
        fn_cnt = metrics.get('fn_cnt', 0)
        bp_ok_present = metrics.get('bp_ok_present', False)
        
        # Determine output directory
        temp_pat_name = pat_name[0:int(len(pat_name) / 3)]
        
        if tp_cnt > 0 or fn_cnt > 0 or bp_ok_present:
            save_path = out_path / 'TP_FP_FN'
            os.makedirs(save_path, exist_ok=True)
            fig_filepath = save_path / f"{temp_pat_name}--{fig_title}.png"
        else:
            fig_filepath = out_path / f"{fig_title}.png"
        
        plt.savefig(fig_filepath, bbox_inches='tight')
        
        return fig_filepath


class SignalPlotter:
    """Utility class for plotting various signal types."""
    
    @staticmethod
    def plot_signal_comparison(signals: Dict[str, np.ndarray], 
                             time: np.ndarray,
                             title: str = "Signal Comparison",
                             figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """
        Plot multiple signals for comparison.
        
        Args:
            signals: Dictionary of signal_name: signal_array
            time: Time array
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        n_signals = len(signals)
        fig, axes = plt.subplots(n_signals, 1, figsize=figsize, sharex=True)
        
        if n_signals == 1:
            axes = [axes]
        
        for i, (name, signal) in enumerate(signals.items()):
            axes[i].plot(time, signal, 'b-', linewidth=1)
            axes[i].set_ylabel(name)
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_spectrogram(frequencies: np.ndarray, 
                        time: np.ndarray,
                        spectrogram: np.ndarray,
                        title: str = "Spectrogram",
                        figsize: Tuple[float, float] = (12, 6)) -> plt.Figure:
        """
        Plot time-frequency spectrogram.
        
        Args:
            frequencies: Frequency array
            time: Time array
            spectrogram: 2D spectrogram array
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                      extent=[time[0], time[-1], frequencies[0], frequencies[-1]])
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title)
        
        plt.colorbar(im, ax=ax, label='Magnitude')
        plt.tight_layout()
        
        return fig


def setup_matplotlib_backend():
    """Setup matplotlib backend for non-interactive plotting."""
    mpl.use("Agg")  # Non-interactive backend for file output only


def create_output_directories(base_path: Path) -> Dict[str, Path]:
    """
    Create output directories for different types of plots.
    
    Args:
        base_path: Base output directory path
        
    Returns:
        Dictionary of directory purposes and paths
    """
    directories = {
        'tp_fp_fn': base_path / 'TP_FP_FN',
        'tn': base_path / 'TN',
        'analysis': base_path / 'Analysis',
        'spectrograms': base_path / 'Spectrograms'
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories
