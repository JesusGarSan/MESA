"""
We want to parallelize each of the windwos
We want to segment the signs in windows.
We want to be able to apply overlap and windowing
"""
import numpy as np
import scipy
from scipy.signal import get_window, ShortTimeFFT


def fft_bin(signal:float, n_bins:int, sr:float):
    fft = np.fft.fft(signal, n=n_bins)
    freqs = np.fft.fftfreq(n_bins, d=1/sr)    
    return fft, freqs


def save(matrix, column_names = None, filepath=".data/matrix.mat"):
    dict = {'matrix': matrix}
    # Turn 1D arrays into row arrays
    if len(matrix.shape) == 1: matrix = matrix[np.newaxis, :]

    if column_names is not None:
        if len(column_names) != matrix.shape[1]:
            raise ValueError("The number of column names does not match the number of columns in the matrix.")
        dict['column_names'] = column_names

    try:
        scipy.io.savemat(filepath, dict)
    except Exception as e:
        print(e)
        return False
        
    return True

def stft(signal, sr, win_samples, window = "boxcar", padding="odd", t_phase = 0):
    win = get_window(window, win_samples)
    SFT = ShortTimeFFT(win,win_samples,sr)

    time = SFT.t(len(signal)) + t_phase
    freq = SFT.f
    Zxx = SFT.stft(signal, padding=padding)

    return time, freq, Zxx

def spectrogram(signal, sr, win_samples, window = "boxcar", padding="odd", t_phase = 0):
    win = get_window(window, win_samples)
    SFT = ShortTimeFFT(win,win_samples,sr)

    time = SFT.t(len(signal)) + t_phase
    freq = SFT.f
    Zxx = SFT.spectrogram(signal, padding=padding)

    return time, freq, Zxx

