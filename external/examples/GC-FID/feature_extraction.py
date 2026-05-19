# %% Import libraries
from scipy.io import savemat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa.features import stft, stft_labels
from mesa.visualization import plot_signal, dual_plot, spectrogram
import librosa
from airPLS import airPLS

# % Read data
F_data = pd.read_csv("external/data/GC-FID/Y_data.csv", header=None)
F_data_matrix = F_data.to_numpy()

idx = np.where(np.sum(F_data_matrix, axis=1)==0)

df = pd.read_csv("external/data/GC-FID/X_data.csv", header=None)
X = df.to_numpy()
X = np.delete(X, idx, axis=1).T
N_signals, signal_samples = X.shape

# %% Initial visualization
plt.plot(X[:,:].T, alpha=0.25);
# plt.ylim(-200, 2000)

# %% Baseline reduction using airPLS
print("Removing baseline...")
for sample_id in range(X.shape[0]):
    correction = airPLS(X[sample_id])
    X[sample_id] -= correction

# %% Visualization after baseline reduction
plt.plot(X[:,:].T, alpha=0.25);
# plt.ylim(-200, 2000)

# %% SNV - Autoscale row-wise
if False: # Just testing (24/04/2026)
    for i in range(X.shape[0]):
        aux = X[i, :]
        mu = np.nanmean(aux)
        sigma = np.nanstd(aux)
        X[i,:] = (aux - mu) /sigma
# %% Visualization after SNV reduction
plt.plot(X[:,:].T, alpha=0.25);
plt.ylim(-0.5, 5)

# %% STFT parameters and calculation
sr = 200
win_length = 500
hop_length = 250
n_fft = win_length

D = stft(X, win_length, hop_length, n_fft)
Sxx = np.abs(D)**2

times, freqs = stft_labels(D, sr, hop_length, n_fft)

# %% Visualization
signal_id = 10
fig = dual_plot(X[signal_id], Sxx[signal_id], sr, win_length, hop_length, n_fft);
axes = fig.get_axes()
axes[1].set_ylim(0,5)

# %% Filter frequencies
freqs_id = np.where(freqs<=5)
freqs = freqs[freqs_id]
Sxx = Sxx[:,freqs_id, :]


# %% Preprocessing - Sample scaling
if True:
    # Specify the axis whose effects you would like to highlight
    mu    = np.nanmean(Sxx, axis=(0), keepdims=True)
    sigma = np.nanstd (Sxx, axis=(0), keepdims=True) + 1e-12
    print(f"mu.shape: {mu.shape}")

    # Mean-Centering
    Sxx = Sxx - mu
    # Autoscaling
    # Sxx = Sxx/sigma


# %% Unfolding
from mesa.fusion import unfold_2D
subjects = np.arange(96)+1
labels = [subjects, freqs, times]
label_names = ["subjects", "freq", "time"]
X_unfold, labels_unfold, label_names = unfold_2D(Sxx,rows=[0,1], cols=[2], labels=labels, label_names=label_names)

# %% Read complementary data
X_peak = pd.read_csv("data/X_peak.csv", header=None)
F_peak = pd.read_csv("data/Y_peak.csv", header=None)
F_data = pd.read_csv("data/Y_data.csv", header=None)
F_data = np.delete(F_data.to_numpy(), idx, axis = 0)

# %% Save data
data_dict = {
    "X_raw": X_raw,
    "X_sample_scaled": X_sample_scaled,
    "X_block_scaled": X_block_scaled,
    "X_norm": X_norm,
    "times": times,
    "freqs": freqs,
    "X_fft": X_FFT,

    "X_peak": X_peak,
    "F_peak": F_peak,
    "F_data": F_data,
}
savemat("data.mat", data_dict)

