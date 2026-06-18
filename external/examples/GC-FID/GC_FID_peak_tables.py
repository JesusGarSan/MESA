#!/usr/bin/env python3
# %%
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



# %% Read data
X = pd.read_csv("external/data/GC-FID/X_peak.csv", header=None)
Y = pd.read_csv("external/data/GC-FID/Y_peak.csv", header=None)

print(X.shape)
print(Y.shape)
# %% Save the data
path = "external/examples/GC-FID/GC-FID_peaks.mat"
savemat(
    path, {
        "data": X,
        "label_names": np.array([], dtype=object),
        "obs_l": np.array([], dtype=object),
        "var_l": np.array([], dtype=object),
        "F_data": Y
    }
)