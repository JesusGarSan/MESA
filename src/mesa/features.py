""""
Feature extraction module of the MESA package.
Contains functions to compute features from input signals
"""
import numpy as np
import librosa
from typing import Literal

def stft(signal: np.ndarray, win_length: int, hop: int = None, n_fft: int = None,
         window: str | tuple | np.ndarray = "blackman", center: bool = False,
         freq_decimation_method: Literal["drop_value", "average"] = "drop_value",
         demean_slices: bool = True):
    
    if hop is None:
        hop = win_length  # No overlap

    freq_decimation = False    
    requested_n_fft = n_fft

    if n_fft is None:
        n_fft = win_length
    elif n_fft < win_length:
        n_fft = win_length
        freq_decimation = True 

    if center:
        padding = [(0, 0)] * (signal.ndim - 1) + [(int(n_fft // 2), int(n_fft // 2))]
        signal = np.pad(signal, padding, mode='reflect')
        
    frames = librosa.util.frame(signal, frame_length=n_fft, hop_length=hop, axis=-1)
    
    if demean_slices:
        frames = frames - np.mean(frames, axis=-2, keepdims=True)
        
    fft_window = librosa.filters.get_window(window, n_fft, fftbins=True)
    
    window_shape = [1] * frames.ndim
    window_shape[-2] = n_fft
    fft_window = fft_window.reshape(window_shape)
    
    windowed_frames = frames * fft_window
    
    D = np.fft.rfft(windowed_frames, axis=-2)

    if freq_decimation:  # Decimate the freq resolution as desired
        current_bins = D.shape[-2]
        desired_bins = 1 + requested_n_fft // 2

        if freq_decimation_method == "drop_value":
            # Calculate stride k to yield exactly the number of desired bins
            k = current_bins // desired_bins
            idx = [slice(None)] * D.ndim
            idx[-2] = slice(None, desired_bins * k, k)
            D = D[tuple(idx)]
            
        elif freq_decimation_method == "average":
            # Split the frequency axis into exactly 'desired_bins' groups
            chunks = np.array_split(D, desired_bins, axis=-2)
            # Average each group along the frequency axis and stack them back together
            D = np.stack([chunk.mean(axis=-2) for chunk in chunks], axis=-2)

    return D

def stft_labels(D:np.array, sr:int, hop:int, n_fft:int=None):

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_frames = D.shape[-1]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)

    return times, freqs