""""
Feature extraction module of the MESA package.
Contains functions to compute features from input signals
"""
import numpy as np
import librosa
from typing import Literal

def stft(signal:np.array, win_length:int, hop:int = None, n_fft:int = None,
         window:str|tuple|np.ndarray="blackman", center:bool = False,
         freq_decimation_method:Literal["drop_value","average"]="drop_value"):
    """
    Computes a Short-Time Fourier Transform (STFT) with custom frequency-axis decimation.

    This function wraps librosa.stft. When n_fft < win_length, it bypasses the 
    standard requirement by computing the STFT at full win_length resolution 
    and then decimating the frequency axis according to the chosen method.

    Parameters
    ----------
    signal : np.ndarray
        The input audio signal (time series).
    win_length : int
        Number of samples per window.
    hop : int, optional
        Number of samples between successive frames. Defaults to win_length (0% overlap).
    n_fft : int, optional
        Desired number of FFT bins. If n_fft < win_length, decimation is performed.
        Defaults to win_length.
    window : str, tuple, or np.ndarray, optional
        Window function to apply. Defaults to "blackman".
    center : bool, optional
        If True, pads the signal so frames are centered. If False, the first frame 
        begins at t=0. Defaults to False.
    freq_decimation_method : {"drop_value", "average"}, optional
        Method used to reduce frequency resolution:
        - "drop_value": Keeps 1 out of every k bins (stride-based slicing).
        - "average": Averages blocks of k bins (spectral binning).

    Returns
    -------
    D : np.ndarray
        Complex-valued matrix of STFT coefficients. Shape: (1 + n_fft//2, t).

    Notes
    -----
    The decimation factor k is determined by the ratio of bins produced by 
    win_length vs n_fft. If the axis is not perfectly divisible by k, 
    the trailing bins are cropped to maintain a consistent output shape.
    """
    if hop is None:
        hop = win_length # No overlap

    freq_decimation = False    
    requested_n_fft = n_fft

    if n_fft is None:
        n_fft = win_length
    elif n_fft < win_length:
        n_fft = win_length
        freq_decimation = True # If a lower n_fft is desired, decimation will be performed

    D = librosa.stft(signal,
                win_length=win_length, hop_length=hop,
                n_fft=n_fft,
                window=window, center=center)
    

    if freq_decimation: # Decimate the freq resolution as desired
        current_bins = D.shape[-2]
        desired_bins = 1 + requested_n_fft // 2
        k = current_bins // desired_bins

        if k > 1:
            if freq_decimation_method == "drop_value": # Keep one out of every k elements
                D = D[..., :desired_bins * k:k, :]
                pass
                
            elif freq_decimation_method == "average": # Average every set of k elements
                D = D[..., : (current_bins // k) * k, :]
                new_shape = list(D.shape)
                new_shape[-2] = new_shape[-2] // k
                new_shape.insert(-1, k) 
                D = D.reshape(new_shape).mean(axis=-2)
                pass

    return D