"""
We want to parallelize each of the windwos
We want to segment the signs in windows.
We want to be able to apply overlap and windowing
"""
import numpy as np

def fft_bin(signal:float, n_bins:int, sr:float):
    fft = np.fft.fft(signal, n=n_bins)
    freqs = np.fft.fftfreq(n_bins, d=1/sr)
    
    return fft, freqs


def set_windows(window, shift, t, sr):
    """
    window: Window length in seconds
    shift:  Window shift in seconds
    t:      Signal duration in secods
    sr:     Sampling rate of the signal in Hz
    """
    N = sr*t # Total samples in the signal
    window_n = window*sr
    shift_n = shift*sr

    return