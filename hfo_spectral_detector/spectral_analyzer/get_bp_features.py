import numpy as np
from collections import defaultdict
from scipy.signal import find_peaks, peak_prominences, correlate
from sklearn.preprocessing import minmax_scale

def get_sinusoidal_correlation(fs, bp_signal, hfo_freqs):
    # find max correlation with sinusoidal signal
    max_hfo_sine_corr = 0

    #two_hfo_cycles_samp_len = np.ceil(fs*2/hfo_freqs[2])
    #if len(bp_signal) >= two_hfo_cycles_samp_len:
    ts = np.arange(0, 2, 1.0/fs)  # time vector
    sine_freqs = [hfo_freqs[1]]
    if (hfo_freqs[2]-hfo_freqs[0]) > 10:
        #sine_freqs = np.arange(hfo_freqs[0], hfo_freqs[2],1)
        sine_freqs = np.round(np.linspace(hfo_freqs[0],hfo_freqs[2],10))

    for sine_freq in sine_freqs:
        ys = np.sin(2*np.pi*sine_freq*ts)
        sine_T_samples = int(np.ceil(fs/sine_freq)+1)
        for lag in np.arange(0, sine_T_samples, 2):
            res = np.corrcoef(ys[lag:lag+len(bp_signal)], bp_signal)[0,1]
            if res > max_hfo_sine_corr:
                max_hfo_sine_corr = res

    return max_hfo_sine_corr

def get_sinusoidal_dotproduct(fs, bp_signal, hfo_freqs):
    # find max correlation with sinusoidal signal
    max_hfo_sine_corr = 0

    #two_hfo_cycles_samp_len = np.ceil(fs*2/hfo_freqs[2])
    #if len(bp_signal) >= two_hfo_cycles_samp_len:
    ts = np.arange(0, 2, 1.0/fs)  # time vector
    sine_freqs = [hfo_freqs[1]]
    if (hfo_freqs[2]-hfo_freqs[0]) > 10:
        #sine_freqs = np.arange(hfo_freqs[0], hfo_freqs[2],1)
        sine_freqs = np.round(np.linspace(hfo_freqs[0],hfo_freqs[2],10))

    bp_signal = minmax_scale(bp_signal)
    for sine_freq in sine_freqs:
        ys = np.sin(2*np.pi*sine_freq*ts)
        sine_T_samples = int(np.ceil(fs/sine_freq)+1)
        for lag in np.arange(0, sine_T_samples, 2):
            ys_lag = ys[lag:lag+len(bp_signal)]
            ys_lag = minmax_scale(ys_lag)
            max_dp = np.sum(ys_lag**2)
            res = np.dot(ys_lag, bp_signal)/max_dp
            if res > max_hfo_sine_corr:
                max_hfo_sine_corr = res

    return max_hfo_sine_corr

def get_bp_features(fs, bp_signal, hfo_freqs):
    
    # Initiaize variables wih feature values
    max_hfo_sine_corr = 0
    all_relevant_peaks_locs  =[]
    all_relevant_peaks_avg_freq = 0
    all_relevant_peaks_freq_stddev = 0
    all_relevant_peaks_amplitude_stability = 0
    all_relevant_peaks_prominence_stability = 0

    bp_sig_ampl = np.max(bp_signal)-np.min(bp_signal)
    bp_sig_avg_ampl = np.mean(bp_signal)
    bp_sig_std = np.std(bp_signal)
    bp_sig_pow = np.mean(np.power(bp_signal,2))

    prom_peaks_prom_th = 0
    prom_peaks_loc  =[]
    prom_peaks_avg_freq = 0
    prom_peaks_freq_stddev = 0
    prom_peaks_avg_amplitude_stability = 0
    prom_peaks_prominence_stability = 0

    prom_peaks_prom_th_pos_max = 0
    prom_peaks_prom_th_neg_max = 0

    start_sample_correction = 0
    end_sample_correction = 0

    eoi_signal = bp_signal

    # exhaustive peak search
    #eps_locs, _ = find_peaks(bp_signal, height=np.min(bp_signal))
    eps_locs, _ = find_peaks(bp_signal)

    # bp_time = np.arange(len(bp_signal))
    # plt.plot(bp_time, bp_signal, '-k')
    # plt.plot(bp_time[eps_locs], bp_signal[eps_locs], 'xr')
    # plt.show()
    # plt.pause(1)
    # plt.close()

    if len(eps_locs)>=2:
        eps_proms = peak_prominences(bp_signal, eps_locs)[0]
        prom_peaks_prom_th_pos_max = (np.max(bp_signal)-bp_sig_avg_ampl)
        prom_peaks_prom_th_neg_max = (bp_sig_avg_ampl-np.min(bp_signal))
        prom_peaks_prom_th = prom_peaks_prom_th_pos_max
        if prom_peaks_prom_th_neg_max > prom_peaks_prom_th:
            prom_peaks_prom_th = prom_peaks_prom_th_neg_max

        prom_peaks_sel = np.logical_and(eps_proms>=prom_peaks_prom_th, bp_signal[eps_locs]>bp_sig_avg_ampl)
        prom_peaks_loc = eps_locs[prom_peaks_sel]
        prom_peaks_proms = eps_proms[prom_peaks_sel]

        if len(prom_peaks_loc)>=2:

            # estimate frequency and std.dev. based on the prominent peaks
            prom_peaks_freqs = fs/np.diff(prom_peaks_loc) # peaks
            prom_peaks_avg_freq = np.mean(prom_peaks_freqs)
            prom_peaks_freq_stddev = np.std(prom_peaks_freqs)
    
            start_sample_correction = int(np.round(prom_peaks_loc[0] - (fs/prom_peaks_avg_freq)/2))
            end_sample_correction = int(np.round(len(bp_signal)-(prom_peaks_loc[-1] + (fs/prom_peaks_avg_freq)/2)))

            if start_sample_correction<0:
                start_sample_correction=0
            if end_sample_correction<0:
                end_sample_correction=0

            eoi_signal = bp_signal[0+start_sample_correction:len(bp_signal)-end_sample_correction]

            # calculate the stability of the prominent peaks amplitudes
            max_prom_peak_ampl = np.max(bp_signal[prom_peaks_loc])
            prom_peaks_avg_amplitude_stability = (np.sum(bp_signal[prom_peaks_loc]/max_prom_peak_ampl)-1)/(len(prom_peaks_loc)-1)
            prom_peaks_prominence_stability = (np.sum(prom_peaks_proms/np.max(prom_peaks_proms))-1)/(len(prom_peaks_proms)-1)

            #max_hfo_sine_dp = get_sinusoidal_dotproduct(fs, eoi_signal, hfo_freqs)
            max_hfo_sine_corr = get_sinusoidal_correlation(fs, eoi_signal, hfo_freqs)

            all_relevant_peaks_prom_th = prom_peaks_prom_th*0.1
            all_relevant_peaks_sel = np.logical_and.reduce((eps_locs>=prom_peaks_loc[0], eps_locs<=prom_peaks_loc[-1], eps_proms>=all_relevant_peaks_prom_th))
            all_relevant_peaks_locs = eps_locs[all_relevant_peaks_sel]
            all_relevant_peaks_proms = eps_proms[all_relevant_peaks_sel]
            if len(all_relevant_peaks_locs)>=2:
                # estimate frequency and std.dev. based on all peaks within prominent peaks
                all_relevant_peaks_freqs = fs/np.diff(all_relevant_peaks_locs) # peaks
                all_relevant_peaks_avg_freq = np.mean(all_relevant_peaks_freqs)
                all_relevant_peaks_freq_stddev = np.std(all_relevant_peaks_freqs)

                # calculae the stability of the relevant peaks amplitudes
                max_relevant_peak_ampl = np.max(bp_signal[all_relevant_peaks_locs])
                all_relevant_peaks_amplitude_stability = (np.sum(bp_signal[all_relevant_peaks_locs]/max_relevant_peak_ampl)-1)/(len(all_relevant_peaks_locs)-1)
                all_relevant_peaks_prominence_stability = (np.sum(all_relevant_peaks_proms/np.max(all_relevant_peaks_proms))-1)/(len(all_relevant_peaks_proms)-1)
                

    if eoi_signal.shape[0] != bp_signal.shape[0]:   
        bp_sig_ampl = np.max(eoi_signal)-np.min(eoi_signal)
        bp_sig_avg_ampl = np.mean(eoi_signal)
        bp_sig_std = np.std(eoi_signal)
        bp_sig_pow = np.mean(np.power(eoi_signal,2))

    bp_signal_diff = np.diff(eoi_signal)
    bp_sig_activity = np.var(eoi_signal)
    bp_sig_avg_mobility = np.sqrt(np.var(bp_signal_diff)/np.var(eoi_signal))
    bp_sig_complexity = np.sqrt(np.var(np.diff(bp_signal_diff))/np.var(bp_signal_diff))/bp_sig_avg_mobility


    bp_feats = defaultdict(list)
    bp_feats['bp_sig_ampl'] = bp_sig_ampl
    bp_feats['bp_sig_avg_ampl'] = bp_sig_avg_ampl
    bp_feats['bp_sig_std'] = bp_sig_std
    bp_feats['bp_sig_pow'] = bp_sig_pow

    bp_feats['bp_sig_activity'] = bp_sig_activity
    bp_feats['bp_sig_avg_mobility'] = bp_sig_avg_mobility
    bp_feats['bp_sig_complexity'] = bp_sig_complexity

    bp_feats['max_hfo_sine_corr'] = max_hfo_sine_corr
    bp_feats['all_relevant_peaks_nr'] = len(all_relevant_peaks_locs)
    bp_feats['all_relevant_peaks_avg_freq'] = all_relevant_peaks_avg_freq
    bp_feats['all_relevant_peaks_freq_stddev'] = all_relevant_peaks_freq_stddev
    bp_feats['all_relevant_peaks_amplitude_stability'] = all_relevant_peaks_amplitude_stability
    bp_feats['all_relevant_peaks_prominence_stability'] = all_relevant_peaks_prominence_stability
    bp_feats['prom_peaks_nr'] = len(prom_peaks_loc)
    bp_feats['prom_peaks_avg_freq'] = prom_peaks_avg_freq
    bp_feats['prom_peaks_freqs_stddev'] = prom_peaks_freq_stddev
    bp_feats['prom_peaks_avg_amplitude_stability'] = prom_peaks_avg_amplitude_stability
    bp_feats['prom_peaks_prominence_stability'] = prom_peaks_prominence_stability

    display_prom_peak_th = bp_sig_avg_ampl+prom_peaks_prom_th_pos_max
    if prom_peaks_prom_th_neg_max > prom_peaks_prom_th_pos_max:
        display_prom_peak_th = bp_sig_avg_ampl-prom_peaks_prom_th_pos_max


    return bp_feats, start_sample_correction, end_sample_correction, all_relevant_peaks_locs, prom_peaks_loc, bp_sig_avg_ampl, display_prom_peak_th