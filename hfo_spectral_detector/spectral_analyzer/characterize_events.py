import os
import time
import numpy as np
import pandas as pd
import logging
#import psutil
import gc
#import warnings

import matplotlib as mpl
#mpl.use("TkAgg")
mpl.use("Agg")
# To avoid UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail
# Agg, is a non-interactive backend that can only write to files. 
# #For more information and other ways of solving it see https://matplotlib.org/stable/users/explain/backends.html
import matplotlib
import matplotlib.colors as mplib_colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import cm

#matplotlib.use('agg')

from collections import defaultdict
from joblib import Parallel, delayed
from scipy.signal import firwin
from scipy.ndimage import convolve1d

import cv2
import tkinter

#from silx.math.colormap import apply_colormap
from hfo_spectral_detector.spectral_analyzer.HFO_Spectral_Analyzer import hfo_spectral_analysis

from hfo_spectral_detector.dsp.cwt import dcmwt
from hfo_spectral_detector.spectral_analyzer.get_bp_features import get_bp_features
from hfo_spectral_detector.spectral_analyzer.get_other_spectral_features import get_other_spectral_features
from hfo_spectral_detector.eeg_io.eeg_io import EEG_IO


logger = logging.getLogger(__name__)

COLORMAP = mpl.cm.viridis

CPU_COUNT = os.cpu_count()


# scale numpy array between 0 and 1
def scale_array(arr):
    return (arr-min(arr)) / (np.max(arr)-np.min(arr))

def collect_chann_spec_events(pat_name: str, eeg_reader: EEG_IO, an_wdws_dict: dict, out_path:str=None, power_line_freqs:float=None, go_parallel:bool=True, force_recalc:bool=False, save_spect_img:bool=False)->pd.DataFrame:

    logger.info(f"{pat_name}\ncollect_chann_spec_events")
    print(f"{pat_name}\ncollect_chann_spec_events")

    all_ch_df_filepath = out_path / f"{pat_name}_All_Ch_Objects.parquet"
    if os.path.isfile(all_ch_df_filepath) and  not force_recalc:
        all_ch_contours_df = pd.read_parquet(all_ch_df_filepath)
        return all_ch_contours_df

    new_out_path = out_path / pat_name
    os.makedirs(new_out_path, exist_ok=True)
    mtg_labels = eeg_reader.ch_names
    ch_objs_ls = []
    for i, mtg in enumerate(mtg_labels):

        ch_df_filepath = new_out_path / mtg / f"{pat_name}_{mtg}_objects.parquet"
        ch_contours_df = pd.read_parquet(ch_df_filepath)
        if len(ch_contours_df)>0:
            ch_objs_ls.append(ch_contours_df)
        else:
            pass
        logger.info(f"Collecting {pat_name} --- {mtg} --- Progress: {i}/{len(mtg_labels)} --- {(i+1)/len(mtg_labels)*100:.2f}%")
        print(f"Collecting {pat_name} --- {mtg} --- Progress: {i}/{len(mtg_labels)} --- {(i+1)/len(mtg_labels)*100:.2f}%")
        
    if len(ch_objs_ls)>0:
        logger.info(f"Saving {all_ch_df_filepath}")
        print(f"Saving {all_ch_df_filepath}")
        all_ch_contours_df = pd.concat(ch_objs_ls, ignore_index=True)
        all_ch_contours_df.to_parquet(all_ch_df_filepath, index=False)
    
    return all_ch_contours_df

def characterize_events(pat_name: str, eeg_reader: EEG_IO, an_wdws_dict: dict, out_path:str=None, power_line_freqs:float=60, go_parallel:bool=True, force_recalc:bool=False, save_spect_img:bool=False):

    print(f"{pat_name}\nCharacterize Events")
    assert power_line_freqs is not None, "Power line frequency is not defined!"

    new_out_path = out_path / pat_name
    os.makedirs(new_out_path, exist_ok=True)

    all_ch_df_filepath = new_out_path / "All_Ch_Objects.parquet"

    if os.path.isfile(all_ch_df_filepath) and  not force_recalc:
        return


    if go_parallel:
        save_spect_img = False
    screen_size = (20.0, 11.0)
    mtg_labels = eeg_reader.ch_names
    mtg_signals = eeg_reader.get_data()
    ch_objs_ls = []
    for i, mtg in enumerate(mtg_labels):   
        try:
            start_time = time.time()
        
            #ch_data_idx = np.argwhere(mtg == mtg_eeg_data['mtg_labels']).squeeze()
            ch_data_idx = np.argwhere([mtg == this_ls_mtg for this_ls_mtg in mtg_labels])[0][0]
            assert mtg_labels[ch_data_idx] == mtg, "Incorrect channel data index"            
            channel_specific_characterization(\
                pat_name=pat_name, \
                fs=eeg_reader.fs, \
                screen_size = screen_size, \
                mtg_signal=mtg_signals[ch_data_idx], \
                mtg=mtg, an_wdws_dict=an_wdws_dict, \
                out_path=new_out_path, \
                power_line_freqs=power_line_freqs, \
                go_parallel=go_parallel, force_recalc=force_recalc, save_spect_img=save_spect_img\
                )

            logger.info(f"{pat_name} --- {mtg} --- ProcessingTime: {time.time()-start_time} --- Progress: {i+1}/{len(mtg_labels)} --- {(i+1)/len(mtg_labels)*100:.2f}%")
            print(f"{pat_name} --- {mtg} --- ProcessingTime: {time.time()-start_time} --- Progress: {i+1}/{len(mtg_labels)} --- {(i+1)/len(mtg_labels)*100:.2f}%")
        except Exception as e:
            logger.error(f"Error in channel {i}.{mtg}: {e}")
            print(f"Error in channel {mtg}: {e}")
            #return


def channel_specific_characterization(pat_name: str, fs: float, screen_size:tuple, mtg_signal: np.ndarray, mtg: str, an_wdws_dict: dict, out_path:str, power_line_freqs:float=60, go_parallel:bool=True, force_recalc:bool=False, save_spect_img:bool=False):

    logger.info(f"channel_specific_characterization:{mtg}")
    print(f"channel_specific_characterization:{mtg}")

    new_out_path = out_path / mtg
    os.makedirs(new_out_path, exist_ok=True)
    ch_df_filepath = new_out_path / f"{pat_name}_{mtg}_objects.parquet"
    ch_contours_df = pd.DataFrame()
    if not os.path.isfile(ch_df_filepath) or force_recalc:

        # Define frequencies to analyze with wavelets
        dcmwt_freqs = list(range(80, 500+1, 5))
        dcmwt_freqs = [float(f) for f in dcmwt_freqs]

        # Detect Power Line Noise
        cmwt_freqs_emi, dcwt_emi = dcmwt(mtg_signal, fs, list(range(30, 120, 10)), nr_cycles=6)
        apply_notch_filter =  cmwt_freqs_emi[np.argmax(np.mean(dcwt_emi, axis=1))] == power_line_freqs

        # Notch filter
        if apply_notch_filter:
            notch_freqs = [power_line_freqs]
            notch_width = 10
            ntaps = 513
            logger.info(f"Applying Notch Filters: {power_line_freqs} Hz")
            for nf in notch_freqs:
                notch_coeffs = firwin(ntaps, [int(nf-notch_width), int(nf+notch_width)], width=None, window='hamming', pass_zero='bandstop', fs=fs)
                mtg_signal = convolve1d(np.flip(mtg_signal), notch_coeffs)
                mtg_signal = convolve1d(np.flip(mtg_signal), notch_coeffs)
        
        # Bandpass filter
        ntaps = 256
        bp_cutoff_l = 80
        bp_cutoff_h = int(np.round(fs/3))
        bp_filter_coeffs = firwin(ntaps, [bp_cutoff_l, bp_cutoff_h], width=None, window='hamming', pass_zero='bandpass', fs=fs)
        bp_signal = convolve1d(np.flip(mtg_signal), bp_filter_coeffs)
        bp_signal = convolve1d(np.flip(bp_signal), bp_filter_coeffs)

        # Obtain the Morlet Wavelet Transform from the raw signal
        wvlt_nr_cycles = 6
        start_time = time.time()
        cmwt_freqs, dcwt = dcmwt(
            bp_signal, fs, dcmwt_freqs, nr_cycles=wvlt_nr_cycles)
        #dcwt = dcwt*1000
        logger.info(f"DLP DCWT total_time={time.time()-start_time}")

        # Select events from the current channel
        chann_events_start = an_wdws_dict['start']
        chann_events_end = an_wdws_dict['end']
        nr_events_in_ch = len(chann_events_start)

        #seconds_to_plot = np.random.randint(1, high=len(mtg_signal)/fs, size=10, dtype=int)
        #seconds_to_plot = np.append(seconds_to_plot, [1])

        wdw_objects_feats_ls = []
        tot_nr_contour_objs = 0
        n_jobs = int(CPU_COUNT*1.0)
        if not go_parallel:
            n_jobs = 1

        logger.info(f"Using {n_jobs} parallel jobs")
        print(f"Using {n_jobs} parallel jobs")
        parallel = Parallel(n_jobs=n_jobs, return_as="list")
        wdw_objects_feats_ls = parallel(
                            delayed(hfo_spectro_bp_wdw_analysis)
                            (
                                pat_name=pat_name, \
                                fs=fs, \
                                mtg=mtg, \
                                dcmwt_freqs=dcmwt_freqs, \
                                screen_size=screen_size, \
                                an_start_ms = chann_events_start[evnt_idx]/fs*1000, \
                                an_raw_signal = mtg_signal[chann_events_start[evnt_idx]:chann_events_end[evnt_idx]], \
                                an_bp_signal = bp_signal[chann_events_start[evnt_idx]:chann_events_end[evnt_idx]], \
                                an_dcwt = dcwt[:,chann_events_start[evnt_idx]:chann_events_end[evnt_idx]].copy(), \
                                out_path=new_out_path, \
                                save_spect_img=save_spect_img, \
                                apply_notch_filter=apply_notch_filter,
                            )
                            for evnt_idx in range(nr_events_in_ch)
                            )

        tot_nr_contour_objs += len(wdw_objects_feats_ls)
        logger.info(f"Total nr. contour objects: {tot_nr_contour_objs}")
        print(f"Total nr. contour objects: {tot_nr_contour_objs}")
        
        # while len(wdw_objects_feats_ls) != 0:
        #     wdw_contours_df = wdw_objects_feats_ls.pop(0)
        #     if len(wdw_contours_df)>0:
        #         ch_contours_df = pd.concat([ch_contours_df, wdw_contours_df], ignore_index=True)
        if len(wdw_objects_feats_ls) > 0:
            try:
                ch_contours_df = pd.concat(wdw_objects_feats_ls, ignore_index=True)
            except Exception as e:
                #ram_usage = psutil.virtual_memory()[3]/1000000000
                logger.error(f"Error in channel {mtg}: {e}")
                logger.error(f"Delete mtg_signal, mtg_signal, mtg_signal to free RAM")
                #logger.info('Before, RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
                del mtg_signal
                del bp_signal
                del dcwt
                # Force garbage collection
                gc.collect()
                #logger.info('After, RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

                try:
                    logger.info('Attempt memory intensive saving of characterized objects')
                    logger.info(f'wdw_objects_feats_ls, type: {type(wdw_objects_feats_ls)}, len: {len(wdw_objects_feats_ls)}')
                    assembled_events_ls = []
                    for aei in range(len(wdw_objects_feats_ls)):
                        wdw_contours_df = wdw_objects_feats_ls[aei]

                        #if not isinstance(wdw_contours_df, pd.DataFrame):

                        if isinstance(wdw_contours_df, pd.DataFrame) and len(wdw_contours_df)>0:
                            try:
                                nr_nans = 0
                                for col_name in wdw_contours_df.columns:
                                    nr_nans += wdw_contours_df[col_name].isna().sum()

                                if nr_nans == 0:
                                    assembled_events_ls.append(wdw_contours_df)
                                    pass
                                else:
                                    logger.info(f"{nr_nans} NaNs found at wdw index {aei} in {mtg}")

                            except Exception as eapp:
                                logger.error(f"Error when appending wdw_contours_df{aei}: {eapp}")
                                print(f"Error when appending wdw_contours_df{aei}: {eapp}")
                                logger.info(f"wdw_contours_df type: {type(wdw_contours_df)}")
                                pass

                    logger.info(f'Assembled {len(assembled_events_ls)} events')
                    if len(assembled_events_ls) > 0:
                        ch_contours_df = pd.concat(assembled_events_ls, ignore_index=True)

                except Exception as et:
                    print(f"Error when using memory intensive saving of characterized objects, channel {mtg}: {et}")
                    logger.error(f"Error when using memory intensive saving of characterized objects, channel {mtg}: {et}")

                    return     
            
        # Add relative features and save to parquet file
        if ch_contours_df.shape[0] > 0:
            # Add relative featues
            ch_contours_df['EventBkgrndRatio_Power'] = ch_contours_df['bp_sig_pow']/ch_contours_df['bkgrnd_sig_pow']
            ch_contours_df['EventBkgrndRatio_StdDev'] = ch_contours_df['bp_sig_std']/ch_contours_df['bkgrnd_sig_std']
            ch_contours_df['EventBkgrndRatio_Activity'] = ch_contours_df['bp_sig_activity']/ch_contours_df['bkgrnd_sig_activity']
            ch_contours_df['EventBkgrndRatio_Mobility'] = ch_contours_df['bp_sig_avg_mobility']/ch_contours_df['bkgrnd_sig_avg_mobility']
            ch_contours_df['EventBkgrndRatio_Complexity'] = ch_contours_df['bp_sig_complexity']/ch_contours_df['bkgrnd_sig_complexity']

            rows_before_dropna = ch_contours_df.shape[0]
            ch_contours_df = ch_contours_df.dropna()
            logger.info(f"Rows before dropna: {rows_before_dropna}, Rows after dropna: {ch_contours_df.shape[0]}")
            print(f"Rows before dropna: {rows_before_dropna}, Rows after dropna: {ch_contours_df.shape[0]}")
            for col_name in ch_contours_df.columns:
                nr_nans = ch_contours_df[col_name].isna().sum()
                assert nr_nans == 0, f"Column {col_name} contains {nr_nans} NaN values"
                
            logger.info(f"Saving {ch_df_filepath}")
            print(f"Saving {ch_df_filepath}")
            ch_contours_df.to_parquet(ch_df_filepath, index=False)
        else:
            # If no objects were detected, create an empty dataframe
            logger.info(f"No objects detected in {mtg}")
            print(f"No objects detected in {mtg}")
            with open(ch_df_filepath, "w") as file:
                file.write("NoEvents")
            


def hfo_spectro_bp_wdw_analysis(\
    pat_name, \
    fs, \
    mtg, \
    dcmwt_freqs, \
    screen_size, \
    an_start_ms, \
    an_raw_signal, \
    an_bp_signal, \
    an_dcwt, \
    out_path, \
    save_spect_img, \
    apply_notch_filter, \
    ):

    an_duration_ms = len(an_raw_signal)/fs*1000
    an_time = np.arange(len(an_raw_signal))/fs+an_start_ms/1000

    # scale bandpassed signal between 0 and 1
    an_bp_signal = scale_array(an_bp_signal)
    
    #print(f"\n{pat_name}  {mtg}\nAnalysis Wdw: {str(an_time[0])}")

    wdw_contours_df = pd.DataFrame()

    #ToDo: remove this
    th_idx = 0
    cwt_th_val_str = str(th_idx)

    #start_time = time.time()
    #convert figure to an RGBA array using matplotlib
    use_matplotlib = True
    if use_matplotlib:
        cmap = mpl.cm.jet
        norm = mpl.colors.Normalize(vmin=np.min(an_dcwt), vmax=np.max(an_dcwt), clip=False)
        colors = cmap(norm(an_dcwt))
        spect_int = (colors*255).astype('uint8')[:,:,0:3]
        spect_bgr = spect_int[:,:,0:3].copy()
        spect_bgr[:,:,0] = spect_int[:,:,0:3][:,:,2]
        spect_bgr[:,:,1] = spect_int[:,:,0:3][:,:,1]
        spect_bgr[:,:,2] = spect_int[:,:,0:3][:,:,0]
        spect_bgr = np.flipud(spect_bgr)
        pass
    else:
        # #convert figure to an RGBA array using silx
        # spect = apply_colormap(
        #     data=an_dcwt,
        #     colormap='temperature', #"viridis", "cividis", "magma", "inferno", "plasma", "temperature"
        #     norm = "linear", # linear, gamma, log, sqrt, arcsinh
        #     vmin=np.min(an_dcwt), 
        #     vmax=np.percentile(an_dcwt, 99.85),
        #     #autoscale="minmax" # vmin=np.min(an_dcwt), vmax=np.percentile(an_dcwt, 99.9),
        # )       
        # spect_bgr = spect[:,:,0:3].copy()
        # spect_bgr[:,:,0] = spect[:,:,0:3][:,:,2]
        # spect_bgr[:,:,1] = spect[:,:,0:3][:,:,1]
        # spect_bgr[:,:,2] = spect[:,:,0:3][:,:,0]
        # spect_bgr = np.flipud(spect_bgr)
        pass

    #print(f"Spectrograms_Generation total_time={time.time()-start_time}")

    # Perform Computer Vision analysis of the spectrogram

    #start_time = time.time()
    fig_title = f"{mtg}--{np.min(an_time):.1f}-{np.max(an_time):.1f}s" +  "_th" + cwt_th_val_str
    cwt_range_Hz = (int(dcmwt_freqs[0]), int(dcmwt_freqs[-1]))
    objects, wdw_contours_df = hfo_spectral_analysis(spect_bgr, int(fs), wdw_duration_ms=int(an_duration_ms), cwt_range_Hz=cwt_range_Hz, plot_ok=save_spect_img, fig_title=fig_title, out_path=out_path)
    if len(wdw_contours_df) > 0:
        wdw_contours_df['th_idx'] = [0]*len(wdw_contours_df)

    nr_wdw_objects = len(wdw_contours_df)
    if nr_wdw_objects == 0:
        return pd.DataFrame()

    #print(f"Nr.Objects: {nr_wdw_objects}")
    #print(f"hfo_spectral_analysis total_time={time.time()-start_time}")

    # Edit the features from the contour objects
    wdw_contours_df.insert(0, "notch_filtered", [apply_notch_filter]*nr_wdw_objects, False)
    wdw_contours_df.insert(0, "an_start_ms", [an_start_ms]*nr_wdw_objects, False)
    wdw_contours_df.insert(0, "channel", [str(mtg)]*nr_wdw_objects, False)
    wdw_contours_df.insert(0, "Patient", [str(pat_name)]*nr_wdw_objects, False)
    # convert start and end times to global time
    wdw_contours_df['center_ms']  = wdw_contours_df['center_ms'] + an_start_ms
    wdw_contours_df['start_ms']  = wdw_contours_df['start_ms'] + an_start_ms
    wdw_contours_df['end_ms']  = wdw_contours_df['end_ms'] + an_start_ms
    
    ########################################################################################
    # make space for additional features
    wdw_contours_df['bp_ok'] = [False]*nr_wdw_objects
    wdw_contours_df['visual_valid'] = [False]*nr_wdw_objects
        
    wdw_contours_df['bp_sig_ampl'] = [0.0]*nr_wdw_objects
    wdw_contours_df['bp_sig_avg_ampl'] = [0.0]*nr_wdw_objects
    wdw_contours_df['bp_sig_std'] = [0.0]*nr_wdw_objects
    wdw_contours_df['bp_sig_pow'] = [0.0]*nr_wdw_objects
    wdw_contours_df['bp_sig_activity'] = [0.0]*nr_wdw_objects
    wdw_contours_df['bp_sig_avg_mobility'] = [0.0]*nr_wdw_objects
    wdw_contours_df['bp_sig_complexity'] = [0.0]*nr_wdw_objects

    wdw_contours_df['bkgrnd_sig_ampl'] = [0.0]*nr_wdw_objects
    wdw_contours_df['bkgrnd_sig_avg_ampl'] = [0.0]*nr_wdw_objects
    wdw_contours_df['bkgrnd_sig_std'] = [0.0]*nr_wdw_objects
    wdw_contours_df['bkgrnd_sig_pow'] = [0.0]*nr_wdw_objects
    wdw_contours_df['bkgrnd_sig_activity'] = [0.0]*nr_wdw_objects
    wdw_contours_df['bkgrnd_sig_avg_mobility'] = [0.0]*nr_wdw_objects
    wdw_contours_df['bkgrnd_sig_complexity'] = [0.0]*nr_wdw_objects

    wdw_contours_df['max_hfo_sine_corr'] = [0.0]*nr_wdw_objects
    wdw_contours_df['all_relevant_peaks_nr'] = [0.0]*nr_wdw_objects
    wdw_contours_df['all_relevant_peaks_avg_freq'] = [0.0]*nr_wdw_objects
    wdw_contours_df['all_relevant_peaks_freq_stddev'] = [0.0]*nr_wdw_objects
    wdw_contours_df['all_relevant_peaks_amplitude_stability'] = [0.0]*nr_wdw_objects
    wdw_contours_df['all_relevant_peaks_prominence_stability'] = [0.0]*nr_wdw_objects
    
    wdw_contours_df['prom_peaks_nr'] = [0.0]*nr_wdw_objects
    wdw_contours_df['prom_peaks_avg_freq'] = [0.0]*nr_wdw_objects
    wdw_contours_df['prom_peaks_freqs_stddev'] = [0.0]*nr_wdw_objects
    wdw_contours_df['prom_peaks_avg_amplitude_stability'] = [0.0]*nr_wdw_objects
    wdw_contours_df['prom_peaks_prominence_stability'] = [0.0]*nr_wdw_objects

    wdw_contours_df['inverted_max_hfo_sine_corr'] = [0.0]*nr_wdw_objects
    wdw_contours_df['inverted_all_relevant_peaks_amplitude_stability'] = [0.0]*nr_wdw_objects
    wdw_contours_df['inverted_all_relevant_peaks_prominence_stability'] = [0.0]*nr_wdw_objects
    wdw_contours_df['inverted_prom_peaks_nr'] = [0.0]*nr_wdw_objects
    wdw_contours_df['inverted_prom_peaks_avg_freq'] = [0.0]*nr_wdw_objects
    wdw_contours_df['inverted_prom_peaks_freqs_stddev'] = [0.0]*nr_wdw_objects
    wdw_contours_df['inverted_prom_peaks_avg_amplitude_stability'] = [0.0]*nr_wdw_objects
    wdw_contours_df['inverted_prom_peaks_prominence_stability'] = [0.0]*nr_wdw_objects

    wdw_contours_df['TF_Complexity'] = [0.0]*nr_wdw_objects
    wdw_contours_df['NrSpectrumPeaks'] = [0.0]*nr_wdw_objects
    wdw_contours_df['SumFreqPeakWidths'] = [0.0]*nr_wdw_objects
    wdw_contours_df['NI'] = [0.0]*nr_wdw_objects
    wdw_contours_df['nr_overlapping_objs'] = [0]*nr_wdw_objects
    ########################################################################################

    # For plottinf
    all_relevant_peaks_locs_ls = []
    contour_obj_avg_ampl_ls = []
    prom_peaks_locs_ls = []
    prom_peaks_height_th_ls = []
    contour_raw_sig_ls = []
    contour_bp_sig_ls = []
    contour_time_ls = []
    peak_corrected_contour_bp_sig_ls = []
    peak_corrected_contour_time_ls = []

    # Iterate through all contours in analysis window
    # Get the features from each contour based on the band-passed signal
    #bp_loop_st = time.time()
    get_bp_features_time_cum = 0
    get_other_spectral_features_time_cum = 0
    tp_cnt = 0
    fp_cnt = 0
    tn_cnt = 0
    fn_cnt = 0
    bp_ok_present = False 
    for idx in np.arange(nr_wdw_objects):
        this_hfo_freq = wdw_contours_df.at[idx, 'freq_centroid_Hz']
        this_hfo_max_freq = wdw_contours_df.at[idx, 'freq_max_Hz']
        this_hfo_min_freq = wdw_contours_df.at[idx, 'freq_min_Hz']
        this_hfo_dur_ms = wdw_contours_df.at[idx, 'dur_ms']

        this_hfo_start_ms = wdw_contours_df.at[idx, 'start_ms']
        this_hfo_end_ms = wdw_contours_df.at[idx, 'end_ms']
        this_hfo_center_ms = wdw_contours_df.at[idx, 'center_ms']

        overlap_a = np.logical_and(wdw_contours_df['end_ms'] >= this_hfo_start_ms, wdw_contours_df['end_ms'] <= this_hfo_end_ms)
        overlap_b = np.logical_and(wdw_contours_df['start_ms'] >= this_hfo_start_ms, wdw_contours_df['start_ms'] <= this_hfo_end_ms)
        overlap_c = np.logical_and(this_hfo_start_ms >= wdw_contours_df['start_ms'], this_hfo_end_ms <= wdw_contours_df['end_ms'])
        nr_overlapping_objs = np.sum(np.logical_or(np.logical_or(overlap_a, overlap_b), overlap_c))-1
        wdw_contours_df.at[idx, 'nr_overlapping_objs'] = nr_overlapping_objs

        contour_ss = int(fs*(this_hfo_start_ms-an_start_ms)/1000)
        contour_se = int(fs*(this_hfo_end_ms-an_start_ms)/1000)
        contour_sc = int(fs*(this_hfo_center_ms)/1000)

        # If there is strong power line noise only 1 huge blob will be detected and it will have the same duration as teh whole analysis window
        if contour_ss==0 and contour_se==len(an_raw_signal):
            return []
    
        # Check if contour object is within a visual mark
        obj_visual_valid = False

        if contour_ss < 0:
            contour_ss = 0
        if contour_se > len(an_raw_signal):
            contour_se = len(an_raw_signal)

        contour_dcmwt =  an_dcwt[:, contour_ss:contour_se]
        contour_raw_sig = an_raw_signal[contour_ss:contour_se]
        contour_bp_sig = an_bp_signal[contour_ss:contour_se]
        contour_time = an_time[contour_ss:contour_se]


        # get band-passed EEG features
        #bp_feats_st = time.time()
        bp_feats, start_sample_correction, end_sample_correction, all_relevant_peaks_loc, prom_peaks_loc, contour_obj_avg_ampl, prom_peaks_height_th = get_bp_features(fs=fs, bp_signal = contour_bp_sig, hfo_freqs=(this_hfo_min_freq, this_hfo_freq, this_hfo_max_freq))
        inverted_bp_feats, _, _, _, _, _, _ = get_bp_features(fs=fs, bp_signal=scale_array(contour_bp_sig*-1), hfo_freqs=(this_hfo_min_freq, this_hfo_freq, this_hfo_max_freq))
        contour_ss = contour_ss+start_sample_correction
        contour_se = contour_se-end_sample_correction
        assert contour_ss < len(an_raw_signal), "Incorrect contour_ss"
        assert contour_se > 0, "Incorrect contour_se"

        bkgrnd_sel = np.r_[0:contour_ss, contour_se:len(an_bp_signal)]
        bkgrnd_bp_sig = an_bp_signal[bkgrnd_sel]
        bkgrnd_bp_sig_diff = np.diff(bkgrnd_bp_sig)

        # For the sake of peak detection and sinus correlation, the start and end times were extendedn by a 3 periods of the freq_centroid_Hz in hfo_spectral_analysis
        # Now the start, end and duration are corrected by taking the first and last prominent peaks as start and end of the EOI
        wdw_contours_df.at[idx,'start_ms'] = an_start_ms + 1000*(contour_ss/fs)
        wdw_contours_df.at[idx,'end_ms'] = an_start_ms + 1000*(contour_se/fs)
        wdw_contours_df.at[idx,'center_ms'] = np.mean([wdw_contours_df.at[idx,'start_ms'], wdw_contours_df.at[idx,'end_ms']])
        wdw_contours_df.at[idx,'dur_ms'] = wdw_contours_df.at[idx,'end_ms']-wdw_contours_df.at[idx,'start_ms']

        #get_bp_features_time_cum += (time.time()-bp_feats_st)

        # Get other features based on the time-frequency transform
        #bp_other_st = time.time()
        #other_spectral_feats = get_other_spectral_features(fs=fs, dcmwt_freqs=dcmwt_freqs, dcmwt_matrix=contour_dcmwt)
        #get_other_spectral_features_time_cum += (time.time()-bp_other_st)

        bp_ok = False
        if save_spect_img:
            sinusoidal_ok = bp_feats['max_hfo_sine_corr'] > 0.75
            prom_peaks_freq_ok = np.abs((bp_feats['prom_peaks_avg_freq']-this_hfo_freq)/this_hfo_freq)<=0.2 and \
                                bp_feats['prom_peaks_avg_freq'] >= this_hfo_min_freq and bp_feats['prom_peaks_avg_freq'] <= this_hfo_max_freq and \
                                bp_feats['prom_peaks_avg_freq'] <= fs/3
            prom_peaks_freq_stddev_ok = bp_feats['prom_peaks_freqs_stddev'] <= (this_hfo_max_freq-this_hfo_min_freq) #np.max(np.diff((this_hfo_min_freq, this_hfo_freq, this_hfo_max_freq)))
            prom_peaks_amplitude_ok = bp_feats['prom_peaks_avg_amplitude_stability'] >= 0.2
            prom_peaks_stab_ok = bp_feats['prom_peaks_avg_amplitude_stability'] > 0.4
            nr_prom_peaks_ok = bp_feats['prom_peaks_nr']>=4
            prom_peaks_ok = prom_peaks_freq_ok and prom_peaks_freq_stddev_ok and prom_peaks_amplitude_ok and prom_peaks_stab_ok and nr_prom_peaks_ok

            all_relevant_peaks_freq_ok = bp_feats['all_relevant_peaks_avg_freq'] <= bp_feats['prom_peaks_avg_freq']*2.5 and bp_feats['all_relevant_peaks_avg_freq'] <= fs/3
            nr_relevant_peaks_ok = bp_feats['all_relevant_peaks_nr'] <= 2*bp_feats['prom_peaks_nr']-1
            relevant_peaks_ok = all_relevant_peaks_freq_ok #all_relevant_peaks_freq_ok and nr_relevant_peaks_ok
            tf_ok = other_spectral_feats['NrSpectrumPeaks'] <= 1 and other_spectral_feats['TF_Complexity'] < 0.6
            
            bp_ok = sinusoidal_ok and prom_peaks_ok and relevant_peaks_ok and tf_ok

            bp_ok = wdw_contours_df.at[idx,'hvr'] > 10 and wdw_contours_df.at[idx,'circularity'] > 30 and \
                    bp_feats['prom_peaks_nr']>=4 and bp_feats['max_hfo_sine_corr']>0.80 and \
                    (bp_feats['all_relevant_peaks_nr']-bp_feats['prom_peaks_nr'])<int(bp_feats['prom_peaks_nr']/2) and \
                    bp_feats['prom_peaks_freqs_stddev']<=15 and bp_feats['prom_peaks_avg_amplitude_stability']>= 0.70
            
            inverted_bp_ok = np.abs(bp_feats['prom_peaks_nr']-inverted_bp_feats['prom_peaks_nr'])<=2
            
            bp_ok = bp_ok and inverted_bp_ok
                    
            # if bp_feats['max_hfo_sine_corr']>=0.99 and tf_ok and prom_peaks_freq_ok and bp_feats['prom_peaks_nr']>=3:
            #     bp_ok = True
                    
        wdw_contours_df.at[idx,'bp_ok'] = bp_ok
        wdw_contours_df.at[idx,'visual_valid'] = obj_visual_valid

        wdw_contours_df.at[idx,'bp_sig_ampl'] = bp_feats['bp_sig_ampl']
        wdw_contours_df.at[idx,'bp_sig_avg_ampl'] = bp_feats['bp_sig_avg_ampl']
        wdw_contours_df.at[idx,'bp_sig_std'] = bp_feats['bp_sig_std']
        wdw_contours_df.at[idx,'bp_sig_pow'] = bp_feats['bp_sig_pow']
        wdw_contours_df.at[idx,'bp_sig_activity'] = bp_feats['bp_sig_activity']
        wdw_contours_df.at[idx,'bp_sig_avg_mobility'] = bp_feats['bp_sig_avg_mobility']
        wdw_contours_df.at[idx,'bp_sig_complexity'] = bp_feats['bp_sig_complexity']

        wdw_contours_df.at[idx,'bkgrnd_sig_ampl'] = np.max(bkgrnd_bp_sig)-np.min(bkgrnd_bp_sig)
        wdw_contours_df.at[idx,'bkgrnd_sig_avg_ampl'] = np.mean(bkgrnd_bp_sig)
        wdw_contours_df.at[idx,'bkgrnd_sig_std'] = np.std(bkgrnd_bp_sig)
        wdw_contours_df.at[idx,'bkgrnd_sig_pow'] = np.mean(np.power(bkgrnd_bp_sig,2))
        wdw_contours_df.at[idx,'bkgrnd_sig_activity'] = np.var(bkgrnd_bp_sig)
        wdw_contours_df.at[idx,'bkgrnd_sig_avg_mobility'] = np.sqrt(np.var(bkgrnd_bp_sig_diff)/np.var(bkgrnd_bp_sig))
        wdw_contours_df.at[idx,'bkgrnd_sig_complexity'] = np.sqrt(np.var(np.diff(bkgrnd_bp_sig_diff))/np.var(bkgrnd_bp_sig_diff))/wdw_contours_df.at[idx,'bkgrnd_sig_avg_mobility']

        wdw_contours_df.at[idx,'max_hfo_sine_corr'] = bp_feats['max_hfo_sine_corr']
        wdw_contours_df.at[idx,'all_relevant_peaks_nr'] = bp_feats['all_relevant_peaks_nr']
        wdw_contours_df.at[idx,'all_relevant_peaks_avg_freq'] = bp_feats['all_relevant_peaks_avg_freq']
        wdw_contours_df.at[idx,'all_relevant_peaks_freq_stddev'] = bp_feats['all_relevant_peaks_freq_stddev']
        wdw_contours_df.at[idx,'all_relevant_peaks_amplitude_stability'] = bp_feats['all_relevant_peaks_amplitude_stability']
        wdw_contours_df.at[idx,'all_relevant_peaks_prominence_stability'] = bp_feats['all_relevant_peaks_prominence_stability']
        wdw_contours_df.at[idx,'prom_peaks_nr'] = bp_feats['prom_peaks_nr']
        wdw_contours_df.at[idx,'prom_peaks_avg_freq'] = bp_feats['prom_peaks_avg_freq']
        wdw_contours_df.at[idx,'prom_peaks_freqs_stddev'] = bp_feats['prom_peaks_freqs_stddev']
        wdw_contours_df.at[idx,'prom_peaks_avg_amplitude_stability'] = bp_feats['prom_peaks_avg_amplitude_stability']
        wdw_contours_df.at[idx,'prom_peaks_prominence_stability'] = bp_feats['prom_peaks_prominence_stability']

        wdw_contours_df.at[idx,'inverted_max_hfo_sine_corr'] = inverted_bp_feats['max_hfo_sine_corr']
        wdw_contours_df.at[idx,'inverted_all_relevant_peaks_amplitude_stability'] = inverted_bp_feats['all_relevant_peaks_amplitude_stability']
        wdw_contours_df.at[idx,'inverted_all_relevant_peaks_prominence_stability'] = inverted_bp_feats['all_relevant_peaks_prominence_stability']
        wdw_contours_df.at[idx,'inverted_prom_peaks_nr'] = inverted_bp_feats['prom_peaks_nr']
        wdw_contours_df.at[idx,'inverted_prom_peaks_avg_freq'] = inverted_bp_feats['prom_peaks_avg_freq']
        wdw_contours_df.at[idx,'inverted_prom_peaks_freqs_stddev'] = inverted_bp_feats['prom_peaks_freqs_stddev']
        wdw_contours_df.at[idx,'inverted_prom_peaks_avg_amplitude_stability'] = inverted_bp_feats['prom_peaks_avg_amplitude_stability']
        wdw_contours_df.at[idx,'inverted_prom_peaks_prominence_stability'] = inverted_bp_feats['prom_peaks_prominence_stability']

        # wdw_contours_df.at[idx,'TF_Complexity'] = other_spectral_feats['TF_Complexity']
        # wdw_contours_df.at[idx,'NrSpectrumPeaks'] = other_spectral_feats['NrSpectrumPeaks']
        # wdw_contours_df.at[idx,'SumFreqPeakWidths'] = other_spectral_feats['SumFreqPeakWidths']
        # wdw_contours_df.at[idx,'NI'] = other_spectral_feats['NI']

        if bp_ok:
            bp_ok_present = True

        if save_spect_img:
            pred_label = wdw_contours_df.at[idx,'bp_ok'] and wdw_contours_df.at[idx, 'spect_ok']
            tp_cnt += (pred_label and  wdw_contours_df.at[idx,'visual_valid'])
            fp_cnt +=  (pred_label and not wdw_contours_df.at[idx,'visual_valid'])
            tn_cnt += (not pred_label and not wdw_contours_df.at[idx,'visual_valid'])
            fn_cnt += (not pred_label and wdw_contours_df.at[idx,'visual_valid'])

            peak_corrected_contour_bp_sig = an_bp_signal[contour_ss:contour_se]
            peak_corrected_contour_time = an_time[contour_ss:contour_se]

            all_relevant_peaks_locs_ls.append(all_relevant_peaks_loc)
            contour_obj_avg_ampl_ls.append(contour_obj_avg_ampl)
            prom_peaks_locs_ls.append(prom_peaks_loc)
            prom_peaks_height_th_ls.append(prom_peaks_height_th)
            contour_raw_sig_ls.append(contour_raw_sig)
            contour_bp_sig_ls.append(contour_bp_sig)
            contour_time_ls.append(contour_time)
            peak_corrected_contour_bp_sig_ls.append(peak_corrected_contour_bp_sig)
            peak_corrected_contour_time_ls.append(peak_corrected_contour_time)
        pass

    #print(f"get_bp_features total_time={get_bp_features_time_cum}")
    #print(f"get_other_spectral_features total_time={get_other_spectral_features_time_cum}")
    #print(f"hfo_bp_analysis total_time={time.time()-bp_loop_st}")

    #Test_DLP
    #save_spect_img = tp_cnt>0 or fp_cnt>0 or fn_cnt > 0
    # Draw contours on band-passed signal
    # = save_spect_img and (tp_cnt>0 or fn_cnt > 0 or bp_ok_present)
    if save_spect_img and nr_wdw_objects>0:
        start_time = time.time()
        # creating grid for subplots
        fig = plt.figure(2)
        fig.set_figwidth(screen_size[0])
        fig.set_figheight(screen_size[1])

        new_size = (1000, 250)
        new_obj = cv2.resize(cv2.cvtColor(objects, cv2.COLOR_BGR2RGB), new_size, interpolation=cv2.INTER_LINEAR)

        ax1 = plt.subplot2grid(shape=(6, 10), loc=(0, 3), colspan=7,rowspan=1)
        ax2 = plt.subplot2grid(shape=(6, 10), loc=(1, 3), colspan=7,rowspan=1)
        ax3 = plt.subplot2grid(shape=(6, 10), loc=(2, 3), colspan=7,rowspan=1)
        ax4 = plt.subplot2grid(shape=(6, 10), loc=(3, 3), colspan=7,rowspan=3)
        ax5 = plt.subplot2grid(shape=(6, 10), loc=(0, 0), colspan=3,rowspan=6)

        sigs_lw = 1
        ax1.plot(an_time, an_raw_signal, '-k', linewidth=sigs_lw)
        ax1.set_xlim((np.min(an_time), np.max(an_time)))
        ax1.set_title('Raw')

        ax2.plot(an_time, an_bp_signal, '-k', linewidth=sigs_lw)
        ax2.set_xlim((np.min(an_time), np.max(an_time)))
        bp_ylim_min = np.mean(an_bp_signal) - 1*(np.max(an_bp_signal)-np.min(an_bp_signal))
        bp_ylim_max = np.mean(an_bp_signal) + 1*(np.max(an_bp_signal)-np.min(an_bp_signal))
        ax2.set_ylim((bp_ylim_min, bp_ylim_max))
        ax2.set_title('Band-passed')

        ax3.plot(an_time, an_bp_signal, '-k', linewidth=sigs_lw)
        ax3.set_xlim((np.min(an_time), np.max(an_time)))
        bp_ylim_min = np.mean(an_bp_signal) - 1*(np.max(an_bp_signal)-np.min(an_bp_signal))
        bp_ylim_max = np.mean(an_bp_signal) + 1*(np.max(an_bp_signal)-np.min(an_bp_signal))
        ax3.set_ylim((bp_ylim_min, bp_ylim_max))
        #ax3.set_title('Band-passed')
        
        ax4.imshow(new_obj)
        ax4.set_title('Objects')
        y_ticks_val = np.linspace(0, new_size[1], num=10, endpoint=True, dtype=int).tolist()
        y_ticks_labels = np.flip(np.linspace(dcmwt_freqs[0], dcmwt_freqs[-1], num=10, endpoint=True, dtype=int)).tolist()
        ax4.set_yticks(y_ticks_val,y_ticks_labels)
        x_ticks_vals = np.linspace(0, new_size[0], num=11, endpoint=True, dtype=int)
        x_ticks_labels = np.round(np.linspace(0, new_size[0], num=11, endpoint=True, dtype=int)/fs+np.min(an_time),1)
        ax4.set_xticks(x_ticks_vals,x_ticks_labels)

        ax5.plot(an_time, an_bp_signal, '-w', linewidth=sigs_lw)
        ax5.set_xlim((np.min(an_time), np.max(an_time)))

        # Iterate through contour objects and plot them
        all_objs_legend_str = []
        all_objs_legend_colors = []

        #Header           
        all_objs_legend_str.append("idx (sin) (rp-nr, rp-f) (pp-nr, pp-f) (pp-fstd, pp-stab) (tf-clx, tf-pnr, tf-wsm) (nr_ovo, hclx)")
        all_objs_legend_colors.append('k') 
        for idx in np.arange(nr_wdw_objects):
            all_relevant_peaks_loc = all_relevant_peaks_locs_ls[idx]
            contour_obj_avg_ampl = contour_obj_avg_ampl_ls[idx]
            prom_peaks_loc = prom_peaks_locs_ls[idx]
            prom_peaks_height_th = prom_peaks_height_th_ls[idx]
            contour_bp_sig = contour_bp_sig_ls[idx]
            contour_time = contour_time_ls[idx] 
            peak_corrected_contour_bp_sig = peak_corrected_contour_bp_sig_ls[idx]
            peak_corrected_contour_time = peak_corrected_contour_time_ls[idx] 

            
            # Plot all_relevant_peaks_loc
            ax3.plot(contour_time[all_relevant_peaks_loc], contour_bp_sig[all_relevant_peaks_loc], 'o', markerfacecolor="None", markeredgecolor='cyan', color="cyan")
            # Plot threshold for all_peaks
            ax3.plot(contour_time, np.zeros_like(contour_bp_sig)+contour_obj_avg_ampl, linestyle='--', color="cyan")

            # Plot prom_peaks_loc
            ax3.plot(contour_time[prom_peaks_loc], contour_bp_sig[prom_peaks_loc], "x", color="orange")
            # Plot threshold for peak heights
            ax3.plot(contour_time, np.zeros_like(contour_bp_sig)+prom_peaks_height_th, "--", color="orange")

            if wdw_contours_df.at[idx,'visual_valid']:
                # Create a Rectangle patch
                rect = patches.Rectangle((contour_time[0], np.min(an_raw_signal)), contour_time[-1]-contour_time[0], np.max(an_raw_signal)-np.min(an_raw_signal), linewidth=1, edgecolor='r', facecolor='none')
                ax1.add_patch(rect)

            # Choose legend and plot colors
            hfo_color = "lightcoral"
            hfo_legend_color= 'k'
            if wdw_contours_df.at[idx,'bp_ok'] :
                hfo_color = "dodgerblue"
                if wdw_contours_df.at[idx,'spect_ok']:
                    hfo_color = "lime"
                hfo_legend_color = hfo_color
            ax3.plot(peak_corrected_contour_time, peak_corrected_contour_bp_sig, '-', color=hfo_color, linewidth=sigs_lw)

            # String for legend
            tuple_a = ((wdw_contours_df.at[idx,'all_relevant_peaks_nr']), (wdw_contours_df.at[idx,'all_relevant_peaks_avg_freq']))
            tuple_b = ((wdw_contours_df.at[idx,'prom_peaks_nr']), (wdw_contours_df.at[idx,'prom_peaks_avg_freq']))
            tuple_c = ((wdw_contours_df.at[idx,'prom_peaks_freqs_stddev']), (wdw_contours_df.at[idx,'prom_peaks_avg_amplitude_stability']))
            tuple_d = ((wdw_contours_df.at[idx,'TF_Complexity']), (wdw_contours_df.at[idx,'NrSpectrumPeaks']), (wdw_contours_df.at[idx,'SumFreqPeakWidths']))
            tuple_e = ((wdw_contours_df.at[idx,'nr_overlapping_objs']), (wdw_contours_df.at[idx,'bp_sig_complexity']))
        
            obj_legend_str = f"{idx}. ({wdw_contours_df.at[idx,'max_hfo_sine_corr']:.2f})"
            obj_legend_str += f"({tuple_a[0]}, {tuple_a[1]:.2f})"
            obj_legend_str += f"({tuple_b[0]:.0f}, {tuple_b[1]:.2f})"
            obj_legend_str += f"({tuple_c[0]:.2f}, {tuple_c[1]:.2f})"
            obj_legend_str += f"({tuple_d[0]:.2f}, {tuple_d[1]:.0f}, {tuple_d[2]:.2f})"
            obj_legend_str += f"({tuple_e[0]:.0f}, {tuple_e[1]:.2f})"

            all_objs_legend_str.append(obj_legend_str)
            all_objs_legend_colors.append(hfo_legend_color)               
            pass

        
        # The legend with the feature values needs to be plotted in ax1 so that it doesn't cover the signals in ax1
        # So the following plotting acts more like a placeholder for the legends
        ax5.plot(contour_time_ls[0], contour_raw_sig_ls[0], '-', color='w', linewidth=sigs_lw)
        for idx in np.arange(nr_wdw_objects):
            ax5.plot(contour_time_ls[idx], contour_raw_sig_ls[idx], '-', color='w', linewidth=sigs_lw)
        ax5.legend(all_objs_legend_str, loc="upper left", frameon=True, labelcolor=all_objs_legend_colors) #, framealpha=0.1
        ax5.axis('off')

        plt.suptitle(pat_name +"\n"+ mtg)
        

        # tp_cnt, fp_cnt, tn_cnt, fn_cnt
        #Test_DLP
        temp_pat_name = pat_name[0:int(len(pat_name)/3)]
        #if tp_cnt>0 or fp_cnt>0 or fn_cnt > 0:
        if tp_cnt>0 or fn_cnt > 0 or bp_ok_present:
            new_out_path = out_path / 'TP_FP_FN'
            os.makedirs(new_out_path, exist_ok=True)
            plt.savefig(new_out_path / f"{temp_pat_name}--{fig_title}.png", bbox_inches='tight')
        else:
            pass
            # new_out_path = out_path / 'TN'
            # os.makedirs(new_out_path, exist_ok=True)
            # plt.savefig(new_out_path+fig_title+'.png', bbox_inches='tight')

        fig_filepath = out_path+fig_title+'.png'
        plt.savefig(fig_filepath, bbox_inches='tight')

        plt.close(2)

        print(f"HFO Objects Plot total_time={time.time()-start_time}")
        pass
    
    return wdw_contours_df

if __name__ == "__main__":
    pass