# %% Import libraries
import os, sys
sys.path.insert(1, os.getcwd())

from internal.feature_extraction.energy_check import bins_check
from internal.feature_extraction import features
from internal.simulation.generator import *
from internal.visualization import plot
import matplotlib.pyplot as plt


import numpy as np


# %% Set the parameters
show_plots = True
save_data = True

sr = 100;       # Hz. Sampling rate
t = 60;         # s. Duration of the signal
N = 100;      # Number of different frequencies composing the signal
n_samples = int(t*sr)

window_length = 1.0     # s. Length of the windows in seconds
window_samples = int(window_length*sr)
shift  = window_length  # s. Length of the window shift in seconds
n_bins = int(window_samples);  # Number of bins to use for the STFT calculation
# n_bins = n_bins*10
n_windows = int(t/window_length)
n_freqs = int((n_bins//2))

f0 = [12.0, 13.5, 22.0, 22.15, 32.0, 32.075 ] # Hz. Main frequencies of the simulated signals
sigma_f = 7.5 # standard deviation around the frequencies chosen for the signal generation
sigma_A = 0.5 # standard deviation around the amplitudes chosen for the signal generation
m = len(f0)

n_sensors = 5; # Number of sensors in the simulation
sensors = []

print(f"Forced FFT resolution: {sr/n_bins}Hz")
print(f"True FFT resolution: {sr/(window_length*sr)}Hz")
bins_check(sr, window_length, n_bins)

# %% Generate the signals
signals = np.zeros((n_sensors, n_samples))
times = np.zeros((n_sensors, n_samples))

for i in range(n_sensors):
    F = generate_frequencies(N, f0, sigma_f,sr)
    center = [10 + i**2]*m
    A = generate_amplitudes(N, center,sigma_A)
    phi = generate_phase(N*m)

    times[i,:], signals[i, :] = generate_signal(F, A, sr, t, phi)

    # Convolution
    x_aux = times[i,:] - t/(5/3) # Peak on the middle of the signal
    convolution = 1* np.exp(-(x_aux/10)**2)
    signals[i, :] *= convolution

# %% Plot the signals
if show_plots:
    fig, axes = plt.subplots(n_sensors, sharex=True)
    min_sig = np.min(signals)
    max_sig = np.max(signals)
    for i in range(n_sensors):
        _, axes[i] = plot.signal(times[i,:], signals[i,:], axes[i])
        axes[i].set_ylim(min_sig, max_sig)

    fig.supxlabel("Time")
    fig.supylabel("Amplitude")
    fig.suptitle("Raw signals")
    fig.show()
    # input("Continue?")

# %% Calculate the STFT (Spectrogram)
times_stft = np.zeros((n_sensors, n_windows))
freqs_stft = np.zeros((n_sensors,n_freqs))
Sxx = np.zeros((n_sensors,n_freqs, n_windows))

for i in range(n_sensors):
    time, freq, Sx = features.spectrogram(signals[i,:], sr, window_samples, t_phase=window_length/2)
    times_stft[i,:] = time[:-1]
    freqs_stft[i,:] = freq[:-1]
    Sxx[i,:,:] = Sx[:-1, :-1]

# %% Plot the Spectrograms
if show_plots:
    fig, axes = plt.subplots(n_sensors, sharex=True)
    for i in range(n_sensors):
        _, axes[i], mesh = plot.spectrogram(times_stft[i], freqs_stft[i], Sxx[i], ax=axes[i], vmin=0, vmax=np.max(Sxx))


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(mesh, cax=cbar_ax)

    fig.supxlabel("Time")
    fig.supylabel("Frequency")
    fig.suptitle("Spectrograms")
    fig.show()

# %% Save Spectrogram data
if save_data:
    for i in range(n_sensors):
        features.save(f"data/spectrogram_sensor_{i}.mat",np.squeeze(Sxx[i]),freqs_stft[i])
    pass

# %% Finish
input("End?")