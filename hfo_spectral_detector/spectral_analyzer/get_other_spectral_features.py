import numpy as np
from collections import defaultdict
from scipy.signal import find_peaks, peak_widths
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def get_polynomial_regression(freqs, signal=None, degree=2):

    assert signal is not None

    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(freqs.reshape(-1, 1))

    model = LinearRegression()
    model.fit(x_poly, signal)

    score = model.score(x_poly, signal)
    # print('Polynomial R2 score is: ', score)

    y_pred = model.predict(x_poly)

    # plt.plot(freqs, signal, 'r', label='TF Plot')
    # plt.plot(freqs, y_pred, 'b', label=f'Polyfit ({degree}°)')
    # plt.legend()
    # plt.close()

    return y_pred, score

def get_other_spectral_features(fs, dcmwt_freqs, dcmwt_matrix):
    
    other_feats = defaultdict(list)

    max_freq_pow = np.max(dcmwt_matrix, axis=1)

    # Complexity of frequency domain
    _, score = get_polynomial_regression(np.array(dcmwt_freqs), max_freq_pow, degree=5)
    tf_complexity = 1 - score
    other_feats['TF_Complexity'] = tf_complexity

    # Nr. Peaks in Frequency Domain
    prominence_th = np.abs(np.max(max_freq_pow) - np.mean(max_freq_pow)) * 0.25
    peaks, _ = find_peaks(max_freq_pow, prominence=prominence_th)
    nr_peaks_freq_domain = len(peaks)
    if np.argmax(max_freq_pow) == 0:
        nr_peaks_freq_domain += 1

    other_feats['NrSpectrumPeaks'] = nr_peaks_freq_domain
		
	# Sum of frequency peak-widths
    sum_freq_peak_width = np.sum(np.round(peak_widths(max_freq_pow, peaks)[0]))
    sum_freq_peak_width /= (dcmwt_freqs[-1]-dcmwt_freqs[0])
    other_feats['SumFreqPeakWidths'] = sum_freq_peak_width

	
	# NI value
    ni_val = 0
    other_feats['NI'] = ni_val

    return other_feats