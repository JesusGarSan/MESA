# %% Import libraries
import os, sys
sys.path.insert(1, os.getcwd())

from internal.feature_extraction import features
from internal.visualization import plot

import obspy
from obspy import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt

# %% Execution parameters

show_plots = True
save_data = True

# %% Read local data
path = "./data/9F.NUPH..*.D.2021.143"
st = obspy.read(path)

# %% Pre-process the signal
st.trim(starttime = UTCDateTime("2021-05-23T13:30:00"), endtime = UTCDateTime("2021-05-23T14:00:00"))
# st.detrend("demean")
detrend = "constant" # Apply demean on each window
# detrend = None # Don't apply demean on each window

# %% Plot the signals
# st.plot(block=False)

# %% STFT parameters
n_channels = len(st)
sr = st[0].stats.sampling_rate # We are assuming that all traces have the same sr
window_length = 60.0     # s. Length of the windows in seconds
window_samples = int(window_length*sr)
shift  = window_length  # s. Length of the window shift in seconds
n_bins = int(window_samples);  # Number of bins to use for the STFT calculation

n_windows = int(st[0].stats.npts/window_samples) # We are assuming that all traces have the same number of points
n_freqs = int((n_bins//2))

# %% Calculate the STFT (Spectrogram)
times_stft = np.zeros((n_channels, n_windows))
freqs_stft = np.zeros((n_channels,n_freqs))
Zre = np.zeros((n_channels,n_freqs, n_windows))
Zim = np.zeros((n_channels,n_freqs, n_windows))
Sxx = np.zeros((n_channels,n_freqs, n_windows))

for i in range(n_channels):
    time, freq, Sx = features.stft(st[i].data, sr, window_samples, t_phase=window_length/2, n_bins=n_bins, detrend=detrend)
    times_stft[i,:] = time[:-1]
    freqs_stft[i,:] = freq[:-1]
    Zre[i,:,:] = Sx[:-1, :-1].real
    Zim[i,:,:] = Sx[:-1, :-1].imag
Sxx = Zre**2 + Zim**2

# %% Plot the Spectrograms
fig, axes = plt.subplots(n_channels, sharex=True)
for i in range(n_channels):
    _, axes[i], mesh = plot.spectrogram(times_stft[i], freqs_stft[i], Sxx[i], ax=axes[i],
                                        # vmin=0, vmax=np.max(Sxx),
                                        logscale=True)
    axes[i].set_ylabel(f"{st[i].stats.channel}")
    # axes[i].set_ylim(0,3)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(mesh, cax=cbar_ax)

fig.supxlabel("Time")
fig.supylabel("Frequency")
fig.suptitle("Spectrograms")
fig.show()

# %% Save Spectrogram data
if save_data:
    for i in range(n_channels):
        features.save(f"data/spectrogram_channel_{st[i].stats.channel}.mat",np.squeeze(Sxx[i]).T,row_names=times_stft[i],column_names=freqs_stft[i])
        print(f"Data saved at: data/spectrogram_channel_{st[i].stats.channel}.mat")

# %% Unfold the channels along the columns:
print(Sxx.shape)
n_sensors, n_rows, n_columns = Sxx.shape
data = np.zeros((n_rows, n_columns*n_sensors))
for i in range(n_sensors):
    data[:, i*n_columns:(i+1)*n_columns] = Sxx[i]

# %% Call Matlab to use the Meda Toolbox
import subprocess

matlab_script = 'seismic_signals_meda'
subprocess.run(["matlab", "-nodesktop", "-nosplash", "-r", f"cd('external/tutorials'); {matlab_script};"])
# %% Finish
input("End?")
