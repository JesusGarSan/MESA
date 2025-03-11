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


def save(matrix, row_labels = None, column_labels = None, filepath="./output.mat", verbose = False):
    saved = False
    try:
        dict = {'matrix': matrix}

        if row_labels is not None:
            if len(row_labels) != matrix.shape[0]:
                raise ValueError("The number of row labels does not match the number of rows in the matrix.")
            dict['row_labels'] = row_labels

        if column_labels is not None:
            if len(column_labels) != matrix.shape[1]:
                raise ValueError("The number of column labels does not match the number of columns in the matrix.")
            dict['column_labels'] = column_labels

        scipy.io.savemat(filepath, dict)
        saved=True

    finally:
        if verbose:
            if saved: print(f"File saved successfully at {filepath}.")
            else: print(f"Failed to save file at {filepath}.")

    return saved