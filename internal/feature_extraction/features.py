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


def save(filepath, matrix, row_names = None, column_names = None):
    dict = {'matrix': matrix}
    # Turn 1D arrays into row arrays
    if len(matrix.shape) == 1: matrix = matrix[np.newaxis, :]

    if row_names is not None:
        if len(row_names) != matrix.shape[0]:
            raise ValueError("The number of row names does not match the number of rows in the matrix.")
        dict['row_names'] = row_names

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

def load(filepath):
    """
    Loads the data saved by the 'save' function from a .mat file.

    Args:
        filepath (str): The path to the .mat file.

    Returns:
        tuple: A tuple containing the loaded matrix and, if it exists,
               the list of column names. Returns (None, None) if any
               error occurs during loading.
    """
    try:
        loaded_data = scipy.io.loadmat(filepath)
        matrix = loaded_data.get('matrix')
        column_names = loaded_data.get('column_names')

        # If a 1D array was saved, loadmat loads it as a row matrix,
        # here we return it to its original 1D shape if necessary.
        if matrix is not None and matrix.shape[0] == 1 and 'column_names' not in loaded_data:
            matrix = matrix.flatten()

        return matrix, column_names if column_names is not None else None
    except FileNotFoundError:
        print(f"Error: File not found at path: {filepath}")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None, None


def stft       (signal, sr, win_samples, window = "boxcar", padding="odd", detrend=None, n_bins:int = None, t_phase = 0, **kwargs):
    win = get_window(window, win_samples)
    SFT = ShortTimeFFT(win,win_samples,sr, mfft=n_bins)

    time = SFT.t(len(signal)) + t_phase
    freq = SFT.f
    Zxx = SFT.stft_detrend(signal, padding=padding, detr=detrend, **kwargs)

    return time, freq, Zxx

def spectrogram(signal, sr, win_samples, window = "boxcar", padding="odd", detrend=None, n_bins:int = None, t_phase = 0, **kwargs):
    win = get_window(window, win_samples)
    SFT = ShortTimeFFT(win,win_samples,sr, mfft=n_bins)

    time = SFT.t(len(signal)) + t_phase
    freq = SFT.f
    Sxx = SFT.spectrogram(signal, padding=padding, detr=detrend, **kwargs)

    return time, freq, Sxx

