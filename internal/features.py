"""
We want to parallelize each of the windwos
We want to segment the signs in windows.
We want to be able to apply overlap and windowing
"""
import numpy as np
import scipy

def fft(signal:float, n_bins:int, sr:float):
    fft = np.fft.fft(signal, n=n_bins)
    freqs = np.fft.fftfreq(n_bins, d=1/sr)    
    return fft, freqs


def save(matrix, column_names = None, filepath="./output.mat"):
    dict = {'matrix': matrix}

    if column_names is not None:
        if len(column_names) != matrix.shape[1]:
            raise ValueError("The number of column names does not match the number of columns in the matrix.")
        dict['column_names'] = column_names

    scipy.io.savemat(filepath, dict)
    return True