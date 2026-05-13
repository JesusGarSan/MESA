import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

subject_info = pd.read_csv("external/data/EEG/subject-info.csv")

subject_ids             = subject_info["Subject"].to_numpy(dtype=str)
subject_ages            = subject_info["Age"].to_numpy(dtype=int)
subject_genders         = subject_info["Gender"].to_numpy(dtype=str)
subject_rec_years       = subject_info["Recording year"].to_numpy(dtype=int)
subject_n_subtractions  = subject_info["Number of subtractions"].to_numpy(dtype=float)
subject_quality         = subject_info["Count quality"].to_numpy(dtype=int)
operations = [1, 2]
operation_labels = ["Baseline", "Arithmetic"]

subject_id = subject_ids[10]
operation = operations[0]
file = f"external/data/EEG/{subject_id}_{operation}.edf"
data = mne.io.read_raw_edf(file)
channels = data.ch_names
channels = [x[4:] for x in channels]


import argparse
import sys

parser = argparse.ArgumentParser(description="Procesamiento de EEG con STFT")

parser.add_argument(
    "--window", 
    type=float, 
    default=1.0, 
    help="Longitud de la ventana en segundos (default: 1.0)"
)
parser.add_argument(
    "--hop_fraction", 
    type=float, 
    default=0.25, 
    help="Fracción de desplazamiento (hop) respecto a la ventana, ej: 0.25 para 1/4 (default: 0.25)"
)

args = parser.parse_args()


print(f"""Runing Feature extraction for paramters:
      window length: {args.window}s
      hop fraction:  {args.hop_fraction}  
      """)

# Read data
Y = np.full((len(subject_ids), len(operations), len(channels), int(1e5)), np.nan)

for i, operation in enumerate(operations[:]):
    for j, subject_id in enumerate(subject_ids[:]):
            file = f"external/data/EEG/{subject_id}_{operation}.edf"
            data = mne.io.read_raw_edf(file, verbose=False)
            aux = data.get_data()[:, :]
            last_id = aux.shape[-1]-1000
            Y[j,i,:, 0:last_id] = aux[:, 0:last_id] # Quitamos los últimos segundos, que son artificios.


mask = ~np.isnan(Y).any(axis=(0, 1, 2))
Y = Y[:, :, :, mask]
print(f"Signal tensor dimension: {Y.shape}\n")

# STFT
from mesa.features import stft, stft_labels
import librosa
# A high-pass filter with 0.5 Hz cut-off frequency, low-pass filter with 45 Hz cut-off frequency and a power line notch filter (50 Hz) were used
# We are looking at frequencies in the range: [0.5, 45] Hz

# Downsample
original_sr = 500
target_sr = 100
Y = librosa.resample(Y, orig_sr = original_sr, target_sr = target_sr)

# STFT params
sr = target_sr
window_length = int(args.window*sr)
n_fft = window_length# // 10
hop = int(window_length * args.hop_fraction)
window = "boxcar"
n_bins = window_length
center = True

D = stft(Y,window_length,hop, center=center, n_fft = n_fft)
Sxx = np.abs(D)**2
Sxx = librosa.amplitude_to_db(np.abs(D))

times, freqs = stft_labels(D, sr, hop, n_fft = n_fft)

print(f"Sxx.shape: {Sxx.shape}")
print(f"dimensions: (subject, operation, channel, freqs, times)")

# Filter
### Eliminate filtered frequencies
# A high-pass filter with 0.5 Hz cut-off frequency, low-pass filter with 45 Hz cut-off frequency and a power line notch filter (50 Hz) were used
mask = np.ones(len(freqs), dtype=bool)

mask[freqs > 45.0] = False
mask[freqs <  0.5] = False
# id_50Hz = np.argmin(np.abs(freqs - 50.0))
# mask[id_50Hz] = False


freqs = freqs[mask]
Sxx = Sxx[:,:,:, mask, :]
n_frequencies = len(freqs)
print(f"Nº of frequencies after  masking the filter: {n_frequencies}")

# Normalization
if False:

    # Block scaling: Arithemtic
    # We remove the mean of every axis other than arithmetic
    # Specify the acis whose effects you would like to highlight
    mu    = np.nanmean(Sxx, axis=(0, 1, 2, 4), keepdims=True)
    sigma = np.nanstd (Sxx, axis=(0, 1, 2, 4), keepdims=True) + 1e-12
    print(f"mu.shape: {mu.shape}")

    # Mean-Centering
    Sxx = Sxx - mu
    # Autoscaling
    # Sxx = Sxx/sigma

# Labels
subjects = np.arange(36)+1
operations = np.arange(2)
# channels = np.arange(21)+1

# Get a subset of subjects
# Sxx = Sxx[0:20]
# subjects = subjects[0:20]
# Subset of channels
Sxx = Sxx[:,:, 0:20]
channels = channels[0:20]

labels = [subjects, operations, channels, freqs, times]
label_names = ["subjects", "operations", "channels", "freq", "time"]

# Collapse axis
axis = [2] # modes/axis to replace by their mean
Sxx = np.nanmean(Sxx, axis=tuple(axis), keepdims = False);

for ax in axis[::-1]:
    labels.pop(ax); 
    label_names.pop(ax);

# Unfold
from mesa.fusion import unfold_2D
X_unfold, labels_unfold, label_names = unfold_2D(Sxx,rows=[0,1], cols=[2,3], labels=labels, label_names=label_names)


# Save data
from scipy.io import savemat

print("Saving data...")
savemat(
    "external/examples/EEG.mat", {"data": X_unfold,
                    "label_names": np.array(label_names, dtype=object),
                    "obs_l": np.array(labels_unfold[0],  dtype=object),
                    "var_l": np.array(labels_unfold[1],  dtype=object)}
)


