#!/usr/bin/env python3
# %% Import libraries
import argparse
from scipy.io import savemat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa.features import stft, stft_labels
from mesa.visualization import plot_signal, dual_plot, spectrogram
import librosa
from airPLS import airPLS

def main():
    # %% CLI Argument Parsing
    parser = argparse.ArgumentParser(
        description="Process GC-FID data using STFT with configurable parameters."
    )
    parser.add_argument(
        "--window_length", 
        type=int, 
        default=None, 
        help="Window length for STFT. Defaults to 45001 // 100"
    )
    parser.add_argument(
        "--n_fft", 
        type=int, 
        default=None, 
        help="Number of FFT bins. Defaults to window_length // 1"
    )
    parser.add_argument(
        "--hop", 
        type=int, 
        default=None, 
        help="Hop length for STFT. Defaults to window_length // 1"
    )
    parser.add_argument(
        "--window", 
        type=str, 
        default="boxcar", 
        help="Window function type. Default: 'boxcar'"
    )
    parser.add_argument(
        "--filter", 
        action="store_true", 
        default=True, 
        help="Apply high frequency filtering (0.0 to 5.0 Hz). Default: True"
    )
    
    args = parser.parse_args()

    # Dynamic default assignments to match original relational logic
    if args.window_length is None:
        window_length = 45001 // 100  # 450
    else:
        window_length = args.window_length

    n_fft = args.n_fft if args.n_fft is not None else window_length // 1
    hop = args.hop if args.hop is not None else window_length // 1
    window = args.window
    filter_freqs = args.filter

    center = False
    sr = 200
    subtitle = "Corrected pipeline + STFT"
    
    print("--- Running STFT Pipeline ---")
    print(f"Parameters -> window_length: {window_length}, n_fft: {n_fft}, hop: {hop}, window: {window}")
    # print("------------------------------\n")

    # %% Read data
    F_data = pd.read_csv("external/data/GC-FID/Y_data.csv", header=None)
    F_data_matrix = F_data.to_numpy()

    idx = np.where(np.sum(F_data_matrix, axis=1) == 0)
    idx_blank_24h = [0, 1, 2, 3]
    idx_blank_72h = [52, 53, 54, 55]
    # print(f"Identified blanks: {idx[0]}")
    # print(f"24h blank ids: {idx_blank_24h}")
    # print(f"72h blank ids: {idx_blank_72h}\n")

    df = pd.read_csv("external/data/GC-FID/X_data.csv", header=None)
    X = df.to_numpy().T

    # print(f"Data   matrix X.shape: {X.shape}")
    # print(f"Design matrix F.shape: {F_data_matrix.shape}")

    # %% STFT calculation
    D = stft(X, window_length, hop, window=window, center=center, n_fft=n_fft)
    Sxx = D
    # print(f" Sxx.shape: {Sxx.shape} # samples, freqs, times")
    times, freqs = stft_labels(D, sr, hop, n_fft=n_fft)
    
    # %% Delete high frequencies
    if filter_freqs:
        print("Filtering frequencies...")
        mask = np.ones(len(freqs), dtype=bool)

        mask[freqs > 5.0] = False
        mask[freqs < 0.0] = False

        freqs = freqs[mask]
        Sxx = Sxx[:, mask]
        n_frequencies = len(freqs)

        # print(f"New Sxx.shape: {Sxx.shape} # samples, freqs, times")
        # print(f"New freqs.shape: {freqs.shape}\n")

    # Calculate averages of groups
    idx_24h = np.where(F_data_matrix[:, 0] == 1)[0]
    idx_72h = np.where(F_data_matrix[:, 0] == 2)[0]

    # print(f"ids of group 24h:\n{idx_24h}")
    # print(f"ids of group 72h:\n{idx_72h}\n")

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
    if Sxx.shape[0] > 96: # Checks if the blanks have been deleted
        print("Deleting blanks after blank average removal...")
        # print(f"Blank ids: {idx}")
        Sxx = np.delete(Sxx, idx, axis=0)
        F_data_matrix = np.delete(F_data_matrix, idx, axis=0)

    # %% Unfolding
    from mesa.fusion import unfold_2D
    subjects = np.arange(96) + 1
    labels = [subjects, freqs, times]
    label_names = ["subjects", "freq", "time"]
    X_unfold, labels_unfold, label_names = unfold_2D(
        Sxx, rows=[0], cols=[1, 2], labels=labels, label_names=label_names
    )
    # print(f"X_unfold.shape: {X_unfold.shape} # samples, freqs x times")
    
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

    print(f"Data saved at {path}.")

if __name__ == "__main__":
    main()