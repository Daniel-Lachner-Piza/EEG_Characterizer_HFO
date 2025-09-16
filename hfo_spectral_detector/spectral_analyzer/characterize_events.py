import os
import time
import numpy as np
import pandas as pd
import logging
import gc
from pathlib import Path
from typing import List, Optional

from joblib import Parallel, delayed
from sklearn.preprocessing import minmax_scale

from hfo_spectral_detector.spectral_analyzer.hfo_spectral_analyzer import hfo_spectral_analysis
from hfo_spectral_detector.eeg_io.eeg_io import EEG_IO
from hfo_spectral_detector.dsp.signal_processing import (
    SignalProcessor, create_spectrogram_image
)
from hfo_spectral_detector.spectral_analyzer.feature_extraction import (
    FeatureExtractor, ContourFeatureProcessor, safe_memory_intensive_concatenation
)
from hfo_spectral_detector.spectral_analyzer.hfo_validation import (
    HFOValidator, HFOClassifier, calculate_classification_metrics
)
from hfo_spectral_detector.spectral_analyzer.visualization import (
    HFOVisualizer, setup_matplotlib_backend
)
from hfo_spectral_detector.spectral_analyzer.config import (
    PipelineConfig, DefaultValues, AnalysisConstants
)

# Setup matplotlib backend for non-interactive plotting
setup_matplotlib_backend()

logger = logging.getLogger(__name__)
CPU_COUNT = os.cpu_count()

def save_all_channel_events(pat_name: str, mtg_labels:list, out_path:str=None, all_ch_df_filepath:str=None, verbose:bool=False)->str:

    logger.info(f"{pat_name}\nSave_all_chann_spec_events")
    if verbose:
        print(f"{pat_name}\nSave_all_chann_spec_events")

    ch_objs_ls = []
    for i, mtg in enumerate(mtg_labels):

        ch_df_filepath = out_path / pat_name / mtg / f"{pat_name}_{mtg}_objects.parquet"
        ch_contours_df = pd.read_parquet(ch_df_filepath)
        if len(ch_contours_df)>0:
            ch_objs_ls.append(ch_contours_df)
        else:
            pass
        logger.info(f"Saving {pat_name} --- {mtg} --- Progress: {i}/{len(mtg_labels)} --- {(i+1)/len(mtg_labels)*100:.2f}%")
        if verbose:
            print(f"Saving {pat_name} --- {mtg} --- Progress: {i}/{len(mtg_labels)} --- {(i+1)/len(mtg_labels)*100:.2f}%")

    if len(ch_objs_ls)>0:
        logger.info(f"Saving {all_ch_df_filepath}")
        if verbose:
            print(f"Saving {all_ch_df_filepath}")
        all_ch_contours_df = pd.concat(ch_objs_ls, ignore_index=True)
        all_ch_contours_df.to_parquet(all_ch_df_filepath, index=False)
    
    return all_ch_df_filepath

def characterize_events(pat_name: str, eeg_reader: EEG_IO, mtgs_to_detect: List[Path], 
                       an_wdws_dict: dict, out_path: Path = None, 
                       power_line_freqs: float = 60, n_jobs: int = -1, 
                       force_recalc: bool = False, save_spect_img: bool = False, 
                       verbose: bool = False, config: Optional[PipelineConfig] = None) -> str:
    """
    Characterize HFO events across multiple channels with improved modularity.
    
    Args:
        pat_name: Patient name identifier
        eeg_reader: EEG data reader object
        mtgs_to_detect: List of channels/montages to process
        an_wdws_dict: Dictionary with analysis windows ('start', 'end')
        out_path: Output directory path
        power_line_freqs: Power line frequency for filtering
        n_jobs: Number of parallel jobs
        force_recalc: Force recalculation even if results exist
        save_spect_img: Whether to save spectrogram images
        verbose: Enable verbose logging
        config: Pipeline configuration (uses default if None)
        
    Returns:
        Path to the saved combined results file
    """
    logger.info(f"{pat_name}\nCharacterize Events")
    if verbose:
        print(f"{pat_name}\nCharacterize Events")
    
    assert power_line_freqs is not None, "Power line frequency is not defined!"
    
    # Use default config if not provided
    if config is None:
        config = PipelineConfig.create_default()
    
    # Create output directory
    new_out_path = out_path / pat_name
    os.makedirs(new_out_path, exist_ok=True)
    if verbose:
        print(f"Created output directory: {new_out_path}")
    
    all_ch_df_filepath = new_out_path / config.file_io.all_channels_filename
    
    # Check if results already exist
    if os.path.isfile(all_ch_df_filepath) and not force_recalc:
        return str(all_ch_df_filepath)
    
    # Disable spectrogram saving for parallel processing
    if n_jobs > 1:
        save_spect_img = False
    
    # Get channel data
    mtg_labels = eeg_reader.ch_names
    mtg_signals = eeg_reader.get_data()
    
    # Process each channel
    for i, mtg in enumerate(mtg_labels):
        try:
            start_time = time.time()
            
            # Skip channels not in detection list
            if mtgs_to_detect is not None and len(mtgs_to_detect) > 0:
                if mtg.lower() not in mtgs_to_detect:
                    logger.info(f"Skipping channel {mtg} as it is not in the list of channels to detect")
                    if verbose:
                        print(f"Skipping channel {mtg} as it is not in the list of channels to detect")
                    continue
            
            # Get channel data
            ch_data_idx = np.argwhere([mtg == this_ls_mtg for this_ls_mtg in mtg_labels])[0][0]
            assert mtg_labels[ch_data_idx] == mtg, "Incorrect channel data index"
            
            # Process channel
            channel_specific_characterization(
                pat_name=pat_name,
                fs=eeg_reader.fs,
                mtg_signal=mtg_signals[ch_data_idx],
                mtg=mtg,
                an_wdws_dict=an_wdws_dict,
                out_path=new_out_path,
                power_line_freqs=power_line_freqs,
                n_jobs=n_jobs,
                force_recalc=force_recalc,
                save_spect_img=save_spect_img,
                verbose=verbose,
                config=config
            )
            
            processing_time = time.time() - start_time
            progress = (i + 1) / len(mtg_labels) * 100
            
            logger.info(f"{pat_name} --- {mtg} --- ProcessingTime: {processing_time:.2f} --- "
                       f"Progress: {i + 1}/{len(mtg_labels)} --- {progress:.2f}%")
            if verbose:
                print(f"{pat_name} --- {mtg} --- ProcessingTime: {processing_time:.2f} --- "
                     f"Progress: {i + 1}/{len(mtg_labels)} --- {progress:.2f}%")
                
        except Exception as e:
            logger.error(f"Error in channel {i}.{mtg}: {e}")
            if verbose:
                print(f"Error in channel {mtg}: {e}")
    
    # Save combined results
    save_all_channel_events(pat_name, mtg_labels, out_path, all_ch_df_filepath, verbose=verbose)
    
    return str(all_ch_df_filepath)

def channel_specific_characterization(pat_name: str, fs: float, mtg_signal: np.ndarray, 
                                    mtg: str, an_wdws_dict: dict, out_path: str, 
                                    power_line_freqs: float = 60, n_jobs: int = -1, 
                                    force_recalc: bool = False, save_spect_img: bool = False, 
                                    verbose: bool = False, config: Optional[PipelineConfig] = None):
    """
    Characterize HFO events for a specific channel using modular components.
    
    Args:
        pat_name: Patient name
        fs: Sampling frequency
        mtg_signal: Signal data for the channel
        mtg: Channel/montage name
        an_wdws_dict: Analysis windows dictionary
        out_path: Output directory path
        power_line_freqs: Power line frequency
        n_jobs: Number of parallel jobs
        force_recalc: Force recalculation
        save_spect_img: Save spectrogram images
        verbose: Verbose logging
        config: Pipeline configuration
    """
    logger.info(f"channel_specific_characterization: {mtg}")
    if verbose:
        print(f"channel_specific_characterization: {mtg}")
    
    # Use default config if not provided
    if config is None:
        config = PipelineConfig.create_default()
    
    # Create channel output directory
    new_out_path = out_path / mtg
    os.makedirs(new_out_path, exist_ok=True)
    
    # Define output file path
    ch_df_filepath = new_out_path / config.file_io.objects_filename_template.format(
        patient=pat_name, channel=mtg
    )
    
    # Check if file exists and skip if not forcing recalculation
    if os.path.isfile(ch_df_filepath) and not force_recalc:
        return
    
    # Initialize processors
    signal_processor = SignalProcessor(fs)
    contour_processor = ContourFeatureProcessor()
    
    try:
        # Signal preprocessing
        processed_signal, bp_signal, notch_applied = signal_processor.preprocess_signal(
            mtg_signal, 
            power_line_freqs, 
            apply_notch=True,
            bp_low=config.signal_processing.bandpass_low_freq
        )
        
        # Compute wavelet transform
        cmwt_freqs, dcwt = signal_processor.compute_wavelet_transform(
            bp_signal,
            freq_range=(config.signal_processing.wavelet_freq_min, 
                       config.signal_processing.wavelet_freq_max),
            freq_step=config.signal_processing.wavelet_freq_step,
            nr_cycles=config.signal_processing.wavelet_nr_cycles
        )
        
        logger.info(f"DLP DCWT completed for {mtg}")
        
        # Normalize bandpass signal
        bp_signal_normalized = signal_processor.normalize_signal(bp_signal, 'minmax')
        
        # Select events from the current channel
        chann_events_start = an_wdws_dict['start']
        chann_events_end = an_wdws_dict['end']
        nr_events_in_ch = len(chann_events_start)
        
        # Set up parallel processing
        if n_jobs < 1:
            n_jobs = int(CPU_COUNT * config.processing.default_n_jobs if config.processing.default_n_jobs > 0 else CPU_COUNT)
        
        logger.info(f"Using {n_jobs} parallel jobs")
        if verbose:
            print(f"Using {n_jobs} parallel jobs")
        
        # Process events in parallel
        parallel = Parallel(n_jobs=n_jobs, return_as="list")
        wdw_objects_feats_ls = parallel(
            delayed(hfo_spectro_bp_wdw_analysis)(
                pat_name=pat_name,
                fs=fs,
                mtg=mtg,
                dcmwt_freqs=cmwt_freqs,
                an_start_ms=chann_events_start[evnt_idx] / fs * 1000,
                an_raw_signal=mtg_signal[chann_events_start[evnt_idx]:chann_events_end[evnt_idx]],
                an_bp_signal=bp_signal_normalized[chann_events_start[evnt_idx]:chann_events_end[evnt_idx]],
                an_dcwt=dcwt[:, chann_events_start[evnt_idx]:chann_events_end[evnt_idx]].copy(),
                out_path=new_out_path,
                save_spect_img=save_spect_img,
                notch_applied=notch_applied,
                config=config
            )
            for evnt_idx in range(nr_events_in_ch)
        )
        
        tot_nr_contour_objs = len(wdw_objects_feats_ls)
        logger.info(f"Total nr. contour objects: {tot_nr_contour_objs}")
        if verbose:
            print(f"Total nr. contour objects: {tot_nr_contour_objs}")
        
        # Combine results
        ch_contours_df = pd.DataFrame()
        if len(wdw_objects_feats_ls) > 0:
            try:
                ch_contours_df = pd.concat(wdw_objects_feats_ls, ignore_index=True)
            except Exception as e:
                logger.error(f"Error in channel {mtg}: {e}")
                logger.error(f"Attempting memory-intensive fallback")
                
                # Clean up memory
                del mtg_signal, bp_signal, dcwt
                gc.collect()
                
                # Try memory-intensive concatenation
                ch_contours_df = safe_memory_intensive_concatenation(wdw_objects_feats_ls)
        
        # Process and save results
        if ch_contours_df.shape[0] > 0:
            # Add relative features
            ch_contours_df = contour_processor.add_relative_features(ch_contours_df)
            
            # Validate and clean
            ch_contours_df, stats = contour_processor.validate_and_clean_dataframe(ch_contours_df)
            
            logger.info(f"Rows before cleaning: {stats['rows_before_cleaning']}, "
                       f"After cleaning: {stats['rows_after_cleaning']}")
            if verbose:
                print(f"Rows before cleaning: {stats['rows_before_cleaning']}, "
                     f"After cleaning: {stats['rows_after_cleaning']}")
            
            # Save to file
            logger.info(f"Saving {ch_df_filepath}")
            if verbose:
                print(f"Saving {ch_df_filepath}")
            
            ch_contours_df.to_parquet(ch_df_filepath, index=False)
        else:
            # Create empty file indicator
            logger.info(f"No objects detected in {mtg}")
            if verbose:
                print(f"No objects detected in {mtg}")
            
            with open(ch_df_filepath, "w") as file:
                file.write(config.file_io.empty_file_content)
                
    except Exception as e:
        logger.error(f"Error processing channel {mtg}: {e}")
        if verbose:
            print(f"Error processing channel {mtg}: {e}")
        raise
            


def hfo_spectro_bp_wdw_analysis(
    pat_name: str,
    fs: float,
    mtg: str,
    dcmwt_freqs: List[float],
    an_start_ms: float,
    an_raw_signal: np.ndarray,
    an_bp_signal: np.ndarray,
    an_dcwt: np.ndarray,
    out_path: Path,
    save_spect_img: bool,
    notch_applied: bool,
    config: PipelineConfig
) -> pd.DataFrame:
    """
    Analyze HFO events in a single analysis window using modular components.
    
    Args:
        pat_name: Patient name
        fs: Sampling frequency
        mtg: Channel/montage name
        dcmwt_freqs: Wavelet frequencies
        an_start_ms: Analysis window start time in ms
        an_raw_signal: Raw signal for analysis window
        an_bp_signal: Bandpass signal for analysis window
        an_dcwt: Wavelet coefficients for analysis window
        out_path: Output path
        save_spect_img: Whether to save spectrogram images
        notch_applied: Whether notch filter was applied
        config: Pipeline configuration
        
    Returns:
        DataFrame with characterized events
    """
    an_duration_ms = len(an_raw_signal) / fs * 1000
    an_time = np.arange(len(an_raw_signal)) / fs + an_start_ms / 1000
    
    # Initialize processors
    signal_processor = SignalProcessor(fs)
    feature_extractor = FeatureExtractor(fs)
    contour_processor = ContourFeatureProcessor()
    hfo_classifier = HFOClassifier()
    
    # Normalize bandpass signal
    an_bp_signal = signal_processor.normalize_signal(an_bp_signal, 'minmax')
    
    # Create spectrogram image for computer vision analysis
    spect_bgr = create_spectrogram_image(an_dcwt)
    
    # Perform computer vision analysis of the spectrogram
    fig_title = f"{mtg}--{np.min(an_time):.1f}-{np.max(an_time):.1f}s"
    cwt_range_Hz = (int(dcmwt_freqs[0]), int(dcmwt_freqs[-1]))
    objects, wdw_contours_df = hfo_spectral_analysis(
        spect_bgr, 
        int(fs), 
        wdw_duration_ms=int(an_duration_ms), 
        cwt_range_Hz=cwt_range_Hz, 
        plot_ok=save_spect_img, 
        fig_title=fig_title, 
        out_path=out_path
    )
    
    nr_wdw_objects = len(wdw_contours_df)
    if nr_wdw_objects == 0:
        return pd.DataFrame()
    
    # Initialize feature columns
    wdw_contours_df = contour_processor.initialize_feature_columns(wdw_contours_df, nr_wdw_objects)
    
    # Add metadata columns
    start_times = [an_start_ms] * nr_wdw_objects
    wdw_contours_df = contour_processor.add_metadata_columns(
        wdw_contours_df, pat_name, mtg, start_times, nr_wdw_objects, notch_applied
    )
    
    # Update global times
    wdw_contours_df = contour_processor.update_global_times(wdw_contours_df, an_start_ms)
    
    # Prepare data for visualization if needed
    plotting_data = []
    peaks_data = []
    classification_metrics = {'tp_cnt': 0, 'fp_cnt': 0, 'tn_cnt': 0, 'fn_cnt': 0, 'bp_ok_present': False}
    
    # Process each contour object
    for idx in range(nr_wdw_objects):
        # Extract HFO properties
        this_hfo_freq = wdw_contours_df.at[idx, 'freq_centroid_Hz']
        this_hfo_max_freq = wdw_contours_df.at[idx, 'freq_max_Hz']
        this_hfo_min_freq = wdw_contours_df.at[idx, 'freq_min_Hz']
        
        this_hfo_start_ms = wdw_contours_df.at[idx, 'start_ms']
        this_hfo_end_ms = wdw_contours_df.at[idx, 'end_ms']
        this_hfo_center_ms = wdw_contours_df.at[idx, 'center_ms']
        
        # Calculate overlap features
        overlap_features = feature_extractor.calculate_overlap_features(
            this_hfo_start_ms, this_hfo_end_ms,
            wdw_contours_df['start_ms'].values, wdw_contours_df['end_ms'].values
        )
        wdw_contours_df.at[idx, 'nr_overlapping_objs'] = overlap_features['nr_overlapping_objs']
        
        # Get signal segments
        contour_ss = int(fs * (this_hfo_start_ms - an_start_ms) / 1000)
        contour_se = int(fs * (this_hfo_end_ms - an_start_ms) / 1000)
        
        # Handle edge cases
        if contour_ss == 0 and contour_se == len(an_raw_signal):
            continue  # Skip if contour spans entire window (likely noise)
        
        contour_ss = max(0, contour_ss)
        contour_se = min(len(an_raw_signal), contour_se)
        
        if contour_ss >= contour_se:
            continue  # Skip invalid segments
        
        # Extract signal segments
        contour_raw_sig = an_raw_signal[contour_ss:contour_se]
        contour_bp_sig = an_bp_signal[contour_ss:contour_se]
        contour_time = an_time[contour_ss:contour_se]
        
        # Extract bandpass features
        from hfo_spectral_detector.spectral_analyzer.get_bp_features import get_bp_features
        
        bp_feats, start_correction, end_correction, all_peaks_loc, prom_peaks_loc, avg_ampl, height_th = \
            get_bp_features(fs=fs, bp_signal=contour_bp_sig, 
                          hfo_freqs=(this_hfo_min_freq, this_hfo_freq, this_hfo_max_freq))
        
        # Extract inverted signal features
        inverted_bp_feats, _, _, _, _, _, _ = get_bp_features(
            fs=fs, bp_signal=minmax_scale(contour_bp_sig * -1),
            hfo_freqs=(this_hfo_min_freq, this_hfo_freq, this_hfo_max_freq)
        )
        
        # Update contour boundaries based on peak detection
        contour_ss += start_correction
        contour_se -= end_correction
        
        # Validate boundaries
        if contour_ss >= len(an_raw_signal) or contour_se <= 0 or contour_ss >= contour_se:
            continue
        
        # Extract background signal
        background_indices = np.r_[0:contour_ss, contour_se:len(an_bp_signal)]
        background_signal = an_bp_signal[background_indices]
        
        # Extract background features
        background_features = feature_extractor.extract_background_features(background_signal)
        
        # Update timing features
        wdw_contours_df.at[idx, 'start_ms'] = an_start_ms + 1000 * (contour_ss / fs)
        wdw_contours_df.at[idx, 'end_ms'] = an_start_ms + 1000 * (contour_se / fs)
        wdw_contours_df.at[idx, 'center_ms'] = np.mean([
            wdw_contours_df.at[idx, 'start_ms'], 
            wdw_contours_df.at[idx, 'end_ms']
        ])
        wdw_contours_df.at[idx, 'dur_ms'] = (
            wdw_contours_df.at[idx, 'end_ms'] - wdw_contours_df.at[idx, 'start_ms']
        )
        
        # Store features
        # Bandpass signal features
        for key, value in bp_feats.items():
            if key in wdw_contours_df.columns:
                wdw_contours_df.at[idx, key] = value
        
        # Background features
        for key, value in background_features.items():
            if key in wdw_contours_df.columns:
                wdw_contours_df.at[idx, key] = value
        
        # Inverted signal features
        for key, value in inverted_bp_feats.items():
            inverted_key = f"inverted_{key}"
            if inverted_key in wdw_contours_df.columns:
                wdw_contours_df.at[idx, inverted_key] = value
        
        # Classify event
        spectral_features = {
            'hvr': wdw_contours_df.at[idx, 'hvr'],
            'circularity': wdw_contours_df.at[idx, 'circularity'],
            'spect_ok': wdw_contours_df.at[idx, 'spect_ok'] if 'spect_ok' in wdw_contours_df.columns else False
        }
        
        classification = hfo_classifier.classify_event(
            bp_feats, spectral_features, inverted_bp_feats,
            (this_hfo_min_freq, this_hfo_freq, this_hfo_max_freq), fs
        )
        
        # Update classification columns
        for key, value in classification.items():
            if key in wdw_contours_df.columns:
                wdw_contours_df.at[idx, key] = value
        
        # Update metrics for visualization
        if classification.get('bp_ok', False):
            classification_metrics['bp_ok_present'] = True
        
        # Collect plotting data if needed
        if save_spect_img:
            peaks_data.append({
                'all_relevant_peaks_loc': all_peaks_loc,
                'contour_obj_avg_ampl': avg_ampl,
                'prom_peaks_loc': prom_peaks_loc,
                'prom_peaks_height_th': height_th
            })
            
            plotting_data.append({
                'time': contour_time,
                'signal': contour_bp_sig,
                'corrected_time': an_time[contour_ss:contour_se],
                'corrected_signal': an_bp_signal[contour_ss:contour_se],
                'bp_ok': classification.get('bp_ok', False),
                'spect_ok': spectral_features.get('spect_ok', False)
            })
            
            # Update classification metrics
            predicted_positive = classification.get('bp_ok', False) and spectral_features.get('spect_ok', False)
            actual_positive = wdw_contours_df.at[idx, 'visual_valid']
            
            if predicted_positive and actual_positive:
                classification_metrics['tp_cnt'] += 1
            elif predicted_positive and not actual_positive:
                classification_metrics['fp_cnt'] += 1
            elif not predicted_positive and actual_positive:
                classification_metrics['fn_cnt'] += 1
            else:
                classification_metrics['tn_cnt'] += 1
    
    # Create visualization if requested
    if save_spect_img and nr_wdw_objects > 0:
        visualizer = HFOVisualizer(config.visualization.default_screen_size)
        
        try:
            visualizer.plot_hfo_analysis(
                pat_name=pat_name,
                mtg=mtg,
                an_time=an_time,
                an_raw_signal=an_raw_signal,
                an_bp_signal=an_bp_signal,
                objects_image=objects,
                dcmwt_freqs=dcmwt_freqs,
                contours_df=wdw_contours_df,
                fs=fs,
                out_path=out_path,
                fig_title=fig_title,
                all_relevant_peaks_data=peaks_data,
                contour_plotting_data=plotting_data,
                classification_metrics=classification_metrics
            )
        except Exception as e:
            logger.warning(f"Failed to create visualization: {e}")
    
    return wdw_contours_df

if __name__ == "__main__":
    pass