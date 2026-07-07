#!/usr/bin/env python3
import argparse
import os
from scipy.io import savemat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa.features import stft, stft_labels
from mesa.visualization import plot_signal, dual_plot, spectrogram
import librosa
from airPLS import airPLS
from mesa.fusion import unfold_2D


def main(window_length_fraction=None, hop_fraction=None, window_function="boxcar", verbose = False):
    # Determine window_length from fraction or fallback to CLI/default
    signal_length = 45001
    
    if window_length_fraction is not None:
        window_length = int(window_length_fraction * signal_length)
    else:
        # CLI Argument Parsing fallback
        parser = argparse.ArgumentParser(
            description="Process GC-FID data using STFT with configurable parameters."
        )
        parser.add_argument("--window_length", type=int, default=None)
        parser.add_argument("--n_fft", type=int, default=None)
        parser.add_argument("--hop", type=int, default=None)
        parser.add_argument("--window", type=str, default="boxcar")
        parser.add_argument("--filter", action="store_true", default=True)
        args = parser.parse_args()
        
        window_length = args.window_length if args.window_length is not None else signal_length // 100
        window_function = args.window

    # Derived relational parameters matching original logic
    n_fft = window_length
    
    if hop_fraction is not None:
        hop = int(hop_fraction * window_length)
    else:
        try:
            hop = args.hop if args.hop is not None else window_length // 1
        except NameError:
            hop = window_length // 1

    filter_freqs = True
    center = False
    sr = 200
    
    # %% Read data
    F_data = pd.read_csv("external/data/GC-FID/Y_data.csv", header=None)
    F_data_matrix = F_data.to_numpy()

    # Add null factor for overfitting testing
    null = np.floor(np.random.rand(96+8,1)*4)
    F_data_matrix = np.hstack((F_data_matrix, null))

    idx = np.where(np.sum(F_data_matrix[:,0:3], axis=1) == 0)
    idx_blank_24h = [0, 1, 2, 3]
    idx_blank_72h = [52, 53, 54, 55]

    df = pd.read_csv("external/data/GC-FID/X_data.csv", header=None)
    X = df.to_numpy().T

    # %% STFT calculation
    D = stft(X, window_length, hop, window=window_function, center=center, n_fft=n_fft)
    Sxx = D
    times, freqs = stft_labels(D, sr, hop, n_fft=n_fft)
    
    # %% Delete high frequencies
    if filter_freqs:
        mask = np.ones(len(freqs), dtype=bool)

        mask[freqs > 5.0] = False
        mask[freqs < 0.0] = False

        freqs = freqs[mask]
        Sxx = Sxx[:, mask]
        n_frequencies = len(freqs)

    # Calculate averages of groups
    idx_24h = np.where(F_data_matrix[:, 0] == 1)[0]
    idx_72h = np.where(F_data_matrix[:, 0] == 2)[0]

    # Averages across freqs before removing baseline
    avg_24h = np.mean(Sxx[idx_24h], axis=0)
    avg_72h = np.mean(Sxx[idx_72h], axis=0)  

    # Blanks
    avg_blanks_24h = np.mean(Sxx[idx_blank_24h], axis=0)
    avg_blanks_72h = np.mean(Sxx[idx_blank_72h], axis=0)

    Sxx[idx_24h] -= avg_blanks_24h
    Sxx[idx_72h] -= avg_blanks_72h

    # Averages across freqs after removing baseline
    avg_B_24h = np.mean(Sxx[idx_24h], axis=0)
    avg_B_72h = np.mean(Sxx[idx_72h], axis=0)

    # %% Delete blanks
    if Sxx.shape[0] > 96: 
        Sxx = np.delete(Sxx, idx, axis=0)
        F_data_matrix = np.delete(F_data_matrix, idx, axis=0)

    if verbose:
        print(f"Total samples: {F_data_matrix.shape[0]}")
        # Time, treatment, sex, order, null
        from collections import Counter
        print(f"Inoculation time levels: {dict(Counter(F_data_matrix[:,0].astype(int)))}") 
        print(f"Treatment levels:        {dict(Counter(F_data_matrix[:,1].astype(int)))}") 
        print(f"Sex levels:              {dict(Counter(F_data_matrix[:,2].astype(int)))}") 
        print(f"Order levels:            {dict(Counter(F_data_matrix[:,3].astype(int)))}") 
        print(f"Null levels:             {dict(Counter(F_data_matrix[:,4].astype(int)))}") 
    # %% Unfolding
    subjects = np.arange(96) + 1
    labels = [subjects, freqs, times]
    label_names = ["subjects", "freq", "time"]


    X_unfold, labels_unfold, label_names = unfold_2D(
        Sxx, rows=[0], cols=[1, 2], labels=labels, label_names=label_names
    )
    # print(Sxx.shape)
    # print(X_unfold.shape)
    
    # %% Save the data
    path = "external/examples/GC-FID/GC-FID.mat"
    label_names = ["subjects", "freq", "time"]
    subjects = np.arange(96) + 1
    savemat(
        path, {
            "data": X_unfold,
            "label_names": np.array(label_names, dtype=object),
            "obs_l": np.array(labels_unfold[0], dtype=object),
            "var_l": np.array(labels_unfold[1], dtype=object),
            "F_data": F_data_matrix
        }
    )



if __name__ == "__main__":
    main(0.1, 0.5, 'boxcar', verbose = True) # "test"
    # main(0.01, 0.50, 'boxcar') # "optimal"
    # main(0.25, 0.50   , 'boxcar') # sub-opimal
    # main(0.0025, 0.50, 'boxcar') # overdone
    # main(1.00, 1.00   , 'boxcar') # FFT limit case