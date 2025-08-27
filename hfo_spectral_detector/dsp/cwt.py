from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt


def get_complex_morlet_wavelet(fs, inst_freq, nr_cycles):
    """
    This function generates a complex Morlet wavelet.

    Parameters
    ----------
    fs : float
        Sampling frequency.
    inst_freq : float
        Instantaneous frequency.
    nr_cycles : float, optional
        Number of cycles. The default is 7.

    Returns
    -------
    cmw : numpy.ndarray
        Complex Morlet wavelet.

    """
    kernel_len = round((nr_cycles/inst_freq)*fs)*2
    if np.mod(kernel_len, 2) == 0:
        kernel_len += 1

    t = np.arange(-1*(kernel_len-1)/2, (kernel_len-1)/2+1, 1)/fs

    s = nr_cycles/(2*np.pi*inst_freq)
    A = 1 / np.sqrt(s*np.sqrt(np.pi))
    theta = (2*np.pi*inst_freq*t)

    gauss_wave = np.exp((-1*np.power(t, 2)) / (2*np.power(s, 2)))
    complex_sinus = np.cos(theta) + 1j*np.sin(theta)

    cmw = np.multiply(gauss_wave, complex_sinus)
    cmw = np.multiply(A, cmw)

    # Force mean of zero
    cmw -= np.mean(np.real(cmw))

    # Normalize wavelet
    norm = 'l1_norm'  # 'l1_norm', 'l2_norm'
    if norm == 'l1_norm':
        l1_norm = np.sum(np.abs(cmw))
        cmw = np.divide(cmw, l1_norm)
    elif norm == 'l2_norm':
        l2_norm = np.sqrt(np.sum(np.abs(np.square(cmw))))
        cmw = np.divide(cmw, l2_norm)

    return cmw


def cmwt_serial(signal, fs, freqs, nr_cycles=7):
    """
    This function applies the complex Morlet wavelet transform (CMWT) to a
    time series signal in the frequency domain using a serial implementation.

    Parameters
    ----------
    signal : numpy.ndarray
        The time series signal, represented as a 1-D numpy array.
    fs : float
        The sampling frequency of the time series signal, in Hz.
    freqs : numpy.ndarray
        The frequencies at which to compute the CMWT, represented as a 1-D numpy
        array.
    nr_cycles : float or numpy.ndarray, optional
        The number of cycles to use in the Morlet wavelet, or a 1-D numpy array of
        numbers of cycles for each frequency, default is 7.

    Returns
    -------
    freqs : numpy.ndarray
        The frequencies at which the CMWT was computed, represented as a 1-D numpy
        array.
    cmwtm : numpy.ndarray
        The complex Morlet wavelet transform of the time series signal, represented
        as a 2-D numpy array, with dimensions (len(freqs), len(signal)).

    """

    if type(nr_cycles) == int:
        nr_cycles = np.full(len(freqs), nr_cycles)

    cmwtm = np.zeros((len(freqs), len(signal)), np.float64)
    for i in np.arange(len(freqs)):
        cmw_kernel = get_complex_morlet_wavelet(fs, freqs[i], nr_cycles[i])
        cmwt = np.convolve(signal, cmw_kernel, mode='same')
        cmwt = np.multiply(cmwt, np.conj(cmwt))
        cmwtm[i, :] = cmwt

    return freqs, cmwtm


def cmwt_loop(sampling_rate, sig, freq, cycles):
    """
    This function applies the complex Morlet wavelet transform (CMWT) to a
    time series signal in the frequency domain using a parallel implementation.

    Parameters
    ----------
    sampling_rate : float
        The sampling frequency of the time series signal, in Hz.
    sig : np.ndarray
        The time series signal, represented as a 1-D numpy array.
    freq : float
        The frequency at which to compute the CMWT.
    cycles : float
        The number of cycles to use in the Morlet wavelet.

    Returns
    -------
    np.ndarray
        The complex Morlet wavelet transform of the time series signal, represented
        as a 1-D numpy array.

    """

    cmw_kernel = get_complex_morlet_wavelet(sampling_rate, freq, cycles)

    # cmwt_real = np.convolve(sig, np.real(cmw_kernel), mode='same')
    # cmwt_imag = np.convolve(sig, np.imag(cmw_kernel), mode='same')
    # cmwt = cmwt_real + 1j * cmwt_imag

    cmwt = np.convolve(sig, cmw_kernel, mode='same')
    # cmwt = np.multiply(cmwt, np.conj(cmwt))

    return cmwt


def dcmwt(signal, fs, freqs, nr_cycles=7):
    """
    This function applies the complex Morlet wavelet transform (CMWT) to a
    time series signal in the frequency domain using a parallel implementation.

    Parameters
    ----------
    signal : np.ndarray
        The time series signal, represented as a 1-D numpy array.
    fs : float
        The sampling frequency of the time series signal, in Hz.
    freqs : np.ndarray
        The frequencies at which to compute the CMWT, represented as a 1-D numpy
        array.
    nr_cycles : float or np.ndarray, optional
        The number of cycles to use in the Morlet wavelet, or a 1-D numpy array of
        numbers of cycles for each frequency, default is 7.

    Returns
    -------
    freqs : np.ndarray
        The frequencies at which the CMWT was computed, represented as a 1-D numpy
        array.
    cmwtm : np.ndarray
        The complex Morlet wavelet transform of the time series signal, represented
        as a 2-D numpy array, with dimensions (len(freqs), len(signal)).

    """

    if type(nr_cycles) == int:
        nr_cycles = np.full(len(freqs), nr_cycles)

    cmwtm = np.zeros((len(freqs), len(signal)), np.float64)

    # threads, processes
    n_jobs = cpu_count()-1
    cmwtm_pl = Parallel(n_jobs=int(n_jobs), prefer='threads')(
        delayed(cmwt_loop)(sampling_rate=fs, sig=signal,
                           freq=freqs[i], cycles=nr_cycles[i])
        for i in range(len(freqs))
    )

    cmwtm_pl = np.power(cmwtm_pl, 2)
    cmwtm = np.abs(cmwtm_pl)

    return freqs, cmwtm


if __name__ == "__main__":

    fs = 2000
    freqs = np.arange(60, 520, 10)
    for i in range(len(freqs)):
        get_complex_morlet_wavelet(fs, freqs[i], 8)
        pass
