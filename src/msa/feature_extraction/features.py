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
            raise ValueError(f"The number of row names {len(row_names)}does not match the number of rows in the matrix {matrix.shape[0]}.")
        dict['row_names'] = row_names

    if column_names is not None:
        if len(column_names) != matrix.shape[1]:
            raise ValueError(f"The number of column names {len(column_names)} does not match the number of columns in the matrix {matrix.shape[1]}.")
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


def get_times(signal, sr, win_samples, hop=None, window = "blackman", padding="odd", detrend=None, n_bins:int = None, t_phase = None, p0=1, **kwargs):
    if hop is None:
        hop = win_samples
    if t_phase is None:
        t_phase = - win_samples//sr / 2
    win = get_window(window, win_samples)
    SFT = ShortTimeFFT(win,hop,sr, mfft=n_bins)

    return SFT.t(len(signal), p0=p0, **kwargs) + t_phase

def stft       (signal, sr, win_samples, hop=None, window = "blackman", padding="odd", detrend=None, n_bins:int = None, t_phase = None, p0=1, **kwargs):
    if hop is None:
        hop = win_samples
    if t_phase is None:
        t_phase = - win_samples//sr / 2
    win = get_window(window, win_samples)
    SFT = ShortTimeFFT(win,hop,sr, mfft=n_bins)

    # Add half a window at the beginning of the signal to avoid scipy's default
    # center indexing of windows.
    signal = np.hstack([np.zeros(win_samples//2), signal])

    time = SFT.t(len(signal), p0=p0, **kwargs) + t_phase
    freq = SFT.f
    Zxx = SFT.stft_detrend(signal, padding=padding, detr=detrend, p0=p0, **kwargs)

    return time, freq, Zxx

def spectrogram(signal, sr, win_samples, hop=None, window = "blackman", padding="odd", detrend=None, n_bins:int = None, t_phase = None, p0=1, **kwargs):
    if hop is None:
        hop = win_samples
    if t_phase is None:
        t_phase = - win_samples//sr / 2
    win = get_window(window, win_samples)
    SFT = ShortTimeFFT(win,hop,sr, mfft=n_bins)

    # Add half a window at the beginning of the signal to avoid scipy's default
    # center indexing of windows.
    signal = np.hstack([np.zeros(win_samples//2), signal])

    time = SFT.t(len(signal), p0=p0, **kwargs) + t_phase
    freq = SFT.f
    Sxx = SFT.spectrogram(signal, padding=padding, detr=detrend, p0=p0, **kwargs)

    return time, freq, Sxx

def STFT(sr, win_samples, hop=None, window="blackman", n_bins:int = None, **kwargs):
    """
    Creates a scipy.ShortTimeFFT object with the selected parameters.

    Args:
        sr (float): Sampling rate in Hz.
        win_samples (int): Number of samples in each window (window length).
        hop (int, optional): Number of samples to advance between windows. 
            If None, defaults to win_samples (no overlap).
        window (str, optional): Windowing function to use. Supports any window 
            string compatible with scipy.signal.get_window. Defaults to "blackman".
        n_bins (int, optional): Number of FFT bins (mfft). If None, defaults 
            to win_samples.
        **kwargs: Additional keyword arguments for scipy.ShortTimeFFT.

    Returns:
        scipy.signal.ShortTimeFFT: The configured ShortTimeFFT object.
    """
    if hop is None:
        hop = win_samples
        
    win = get_window(window, win_samples)

    # Passing **kwargs here allows users to customize the SFT object 
    # with parameters like 'scale_to' or 'phase_shift'.
    SFT = ShortTimeFFT(win, hop, fs=sr, mfft=n_bins, **kwargs)

    return SFT



import multiprocessing as mp
from functools import partial

def _stft_worker(signal, SFT, **kwargs):
    """
    Internal worker function to process a single 1D signal.
    
    Checks for detrending requirements in kwargs before choosing the 
    appropriate ShortTimeFFT method.

    Args:
        signal (np.ndarray): 1D array representing a single signal segment.
        SFT (scipy.signal.ShortTimeFFT): The STFT configuration object.
        **kwargs: Arbitrary keyword arguments passed to the STFT method. 
            If "detrend" or "detr" is present, uses SFT.stft_detrend.

    Returns:
        np.ndarray: Complex-valued STFT coefficients (Zxx).
    """
    if "detr" in kwargs:
        return SFT.stft_detrend(signal, **kwargs)
    else:
        return SFT.stft(signal, **kwargs)

def stft_parallel(signal_tensor, SFT, cores=None, **kwargs):
    """
    Parallelizes STFT across a multi-dimensional tensor.
    
    This function flattens the leading dimensions of the input tensor,
    distributes the 1D STFT calculations across multiple CPU cores, 
    and reconstructs the multi-dimensional shape in the output.

    

    Args:
        signal_tensor (np.ndarray): Input data of shape (..., N), where N 
            is the number of samples in the signal.
        SFT (scipy.signal.ShortTimeFFT): Pre-configured SciPy STFT object.
        cores (int, optional): Number of parallel workers. Defaults to 
            mp.cpu_count() // 2.
        **kwargs: Additional arguments passed to the STFT worker 
            (e.g., detrend='linear').

    Returns:
        tuple: (t, f, zxx_reshaped)
            - t (np.ndarray): 1D array of time points.
            - f (np.ndarray): 1D array of frequency bins.
            - zxx_reshaped (np.ndarray): Complex STFT tensor of shape 
              (..., n_freqs, n_times).
    """
    if cores is None:
        cores = mp.cpu_count() // 2

    original_shape = signal_tensor.shape
    # Flatten everything except the last dimension (the signal)
    flat_signals = signal_tensor.reshape(-1, original_shape[-1])
    
    worker_with_args = partial(_stft_worker, SFT=SFT, **kwargs)

    with mp.Pool(processes=cores) as pool:
        results = pool.map(worker_with_args, flat_signals)

    zxx_array = np.array(results) 
    
    f = SFT.f
    t = SFT.t(original_shape[-1])
    
    # Reconstruct shape: Leading dims + Frequency Dim + Time Dim
    final_shape = original_shape[:-1] + zxx_array.shape[1:]
    zxx_reshaped = zxx_array.reshape(final_shape)

    return t, f, zxx_reshaped