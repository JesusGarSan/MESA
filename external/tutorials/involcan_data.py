# %% Import libraries
import os, sys
sys.path.insert(1, os.getcwd())

from src.msa.feature_extraction import features
from src.msa.visualization import plot
# from internal.meda import meda

import obspy
from obspy import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker 

# %% Execution parameters
show_plots = True
save_data = True


# %% Read files
path = "/media/gsus/76C8A92EC8A8EE15/data/mseed/"#C7.PLPI..*.D.*.*"

files = ['*.PLPI..*.D.2021.213', '*.PLPI..*.D.2021.214', '*.PLPI..*.D.2021.215', '*.PLPI..*.D.2021.216', '*.PLPI..*.D.2021.217', '*.PLPI..*.D.2021.218', '*.PLPI..*.D.2021.219', '*.PLPI..*.D.2021.220', '*.PLPI..*.D.2021.221', '*.PLPI..*.D.2021.222', '*.PLPI..*.D.2021.223', '*.PLPI..*.D.2021.224', '*.PLPI..*.D.2021.225', '*.PLPI..*.D.2021.226', '*.PLPI..*.D.2021.227', '*.PLPI..*.D.2021.228', '*.PLPI..*.D.2021.229', '*.PLPI..*.D.2021.230', '*.PLPI..*.D.2021.231', '*.PLPI..*.D.2021.232', '*.PLPI..*.D.2021.233', '*.PLPI..*.D.2021.234', '*.PLPI..*.D.2021.235', '*.PLPI..*.D.2021.236', '*.PLPI..*.D.2021.237', '*.PLPI..*.D.2021.238', '*.PLPI..*.D.2021.239', '*.PLPI..*.D.2021.240', '*.PLPI..*.D.2021.241', '*.PLPI..*.D.2021.242', '*.PLPI..*.D.2021.243', '*.PLPI..*.D.2021.244', '*.PLPI..*.D.2021.245', '*.PLPI..*.D.2021.246', '*.PLPI..*.D.2021.247', '*.PLPI..*.D.2021.248', '*.PLPI..*.D.2021.249', '*.PLPI..*.D.2021.250', '*.PLPI..*.D.2021.251', '*.PLPI..*.D.2021.252', '*.PLPI..*.D.2021.253', '*.PLPI..*.D.2021.254', '*.PLPI..*.D.2021.255', '*.PLPI..*.D.2021.256', '*.PLPI..*.D.2021.257', '*.PLPI..*.D.2021.258', '*.PLPI..*.D.2021.259', '*.PLPI..*.D.2021.260', '*.PLPI..*.D.2021.261', '*.PLPI..*.D.2021.262', '*.PLPI..*.D.2021.263', '*.PLPI..*.D.2021.264', '*.PLPI..*.D.2021.265', '*.PLPI..*.D.2021.266', '*.PLPI..*.D.2021.267', '*.PLPI..*.D.2021.268', '*.PLPI..*.D.2021.269', '*.PLPI..*.D.2021.270', '*.PLPI..*.D.2021.271', '*.PLPI..*.D.2021.272', '*.PLPI..*.D.2021.273', '*.PLPI..*.D.2021.274', '*.PLPI..*.D.2021.275', '*.PLPI..*.D.2021.276', '*.PLPI..*.D.2021.277', '*.PLPI..*.D.2021.278', '*.PLPI..*.D.2021.279', '*.PLPI..*.D.2021.280', '*.PLPI..*.D.2021.281', '*.PLPI..*.D.2021.282', '*.PLPI..*.D.2021.283', '*.PLPI..*.D.2021.284', '*.PLPI..*.D.2021.285', '*.PLPI..*.D.2021.286', '*.PLPI..*.D.2021.287', '*.PLPI..*.D.2021.288', '*.PLPI..*.D.2021.289', '*.PLPI..*.D.2021.290', '*.PLPI..*.D.2021.291', '*.PLPI..*.D.2021.292', '*.PLPI..*.D.2021.293', '*.PLPI..*.D.2021.294', '*.PLPI..*.D.2021.295', '*.PLPI..*.D.2021.296', '*.PLPI..*.D.2021.297', '*.PLPI..*.D.2021.298', '*.PLPI..*.D.2021.299', '*.PLPI..*.D.2021.300', '*.PLPI..*.D.2021.301', '*.PLPI..*.D.2021.302', '*.PLPI..*.D.2021.303', '*.PLPI..*.D.2021.304', '*.PLPI..*.D.2021.305', '*.PLPI..*.D.2021.306', '*.PLPI..*.D.2021.307', '*.PLPI..*.D.2021.308', '*.PLPI..*.D.2021.309', '*.PLPI..*.D.2021.310', '*.PLPI..*.D.2021.311', '*.PLPI..*.D.2021.312', '*.PLPI..*.D.2021.313', '*.PLPI..*.D.2021.314', '*.PLPI..*.D.2021.315', '*.PLPI..*.D.2021.316', '*.PLPI..*.D.2021.317', '*.PLPI..*.D.2021.318', '*.PLPI..*.D.2021.319', '*.PLPI..*.D.2021.320', '*.PLPI..*.D.2021.321', '*.PLPI..*.D.2021.322', '*.PLPI..*.D.2021.323', '*.PLPI..*.D.2021.324', '*.PLPI..*.D.2021.325', '*.PLPI..*.D.2021.326', '*.PLPI..*.D.2021.327', '*.PLPI..*.D.2021.328', '*.PLPI..*.D.2021.329', '*.PLPI..*.D.2021.330', '*.PLPI..*.D.2021.331', '*.PLPI..*.D.2021.332', '*.PLPI..*.D.2021.333', '*.PLPI..*.D.2021.334', '*.PLPI..*.D.2021.335', '*.PLPI..*.D.2021.336', '*.PLPI..*.D.2021.337', '*.PLPI..*.D.2021.338', '*.PLPI..*.D.2021.339', '*.PLPI..*.D.2021.340', '*.PLPI..*.D.2021.341', '*.PLPI..*.D.2021.342', '*.PLPI..*.D.2021.343', '*.PLPI..*.D.2021.344', '*.PLPI..*.D.2021.345', '*.PLPI..*.D.2021.346', '*.PLPI..*.D.2021.347', '*.PLPI..*.D.2021.348', '*.PLPI..*.D.2021.349', '*.PLPI..*.D.2021.350', '*.PLPI..*.D.2021.351', '*.PLPI..*.D.2021.352', '*.PLPI..*.D.2021.353', '*.PLPI..*.D.2021.354', '*.PLPI..*.D.2021.355', '*.PLPI..*.D.2021.356', '*.PLPI..*.D.2021.357', '*.PLPI..*.D.2021.358', '*.PLPI..*.D.2021.359', '*.PLPI..*.D.2021.360', '*.PLPI..*.D.2021.361', '*.PLPI..*.D.2021.362', '*.PLPI..*.D.2021.363', '*.PLPI..*.D.2021.364', '*.PLPI..*.D.2021.365', '*.PLPI..*.D.2022.001', '*.PLPI..*.D.2022.002', '*.PLPI..*.D.2022.003', '*.PLPI..*.D.2022.004', '*.PLPI..*.D.2022.005', '*.PLPI..*.D.2022.006', '*.PLPI..*.D.2022.007', '*.PLPI..*.D.2022.008', '*.PLPI..*.D.2022.009', '*.PLPI..*.D.2022.010', '*.PLPI..*.D.2022.011', '*.PLPI..*.D.2022.012', '*.PLPI..*.D.2022.013', '*.PLPI..*.D.2022.014', '*.PLPI..*.D.2022.015', '*.PLPI..*.D.2022.016', '*.PLPI..*.D.2022.017', '*.PLPI..*.D.2022.018', '*.PLPI..*.D.2022.019', '*.PLPI..*.D.2022.020', '*.PLPI..*.D.2022.021', '*.PLPI..*.D.2022.022', '*.PLPI..*.D.2022.023', '*.PLPI..*.D.2022.024', '*.PLPI..*.D.2022.025', '*.PLPI..*.D.2022.026', '*.PLPI..*.D.2022.027', '*.PLPI..*.D.2022.028', '*.PLPI..*.D.2022.029', '*.PLPI..*.D.2022.030', '*.PLPI..*.D.2022.031', '*.PLPI..*.D.2022.032']
files = ['C7.PPMA..HHE.D.2021.262'] # PLPI, PPMA, PGAR
title = ""
filepath = []

for file in files:
    filepath.append(path+file)
    
print(f"Reading files ...")

if type(filepath)==str:
    st = obspy.read(filepath)
else: 
    st = obspy.read(filepath[0])
    for i in range(1,len(filepath)):
        try:
            st+= obspy.read(filepath[i])
            print(f"Reading {filepath[i]} data...")
        except:print(f"{filepath[i]} not read.")

# st.trim( UTCDateTime("2021-09-19T00:00:00"),
        #  UTCDateTime("2021-09-19T00:10:00"))

print(st.__str__(extended=True))

print(f"Merging {len(st)} traces ...")
st.merge()

detrend = "constant" # Apply demean on each window
if show_plots:
    fig = st.plot(show=False, type="normal", size =(1200,200))
    fig.axes[0].legend().remove()
    fig.texts[0].set_text(title)
    fig.texts[0].set_fontsize(30)
    for ax in fig.axes:
        if ax.legend_ is not None:
            ax.legend().remove()

    # 2. Aumentar el tamaño de los xticks y yticks a 20
    # Itera sobre todos los subplots/ejes en la figura
    for ax in fig.axes:
        ax.locator_params(axis='y', nbins=3)
        date_formatter = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_formatter)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=5))

        ax.set_ylabel("Amplitude", fontsize =20)

        # Aumentar tamaño de las etiquetas de los xticks
        for label in ax.get_xticklabels():
            label.set_fontsize(20)

        # Aumentar tamaño de las etiquetas de los yticks
        for label in ax.get_yticklabels():
            label.set_fontsize(24)




    # fig.show()
    fig.savefig("external/tutorials/involcan_signal.png", bbox_inches='tight')



# %% STFT parameters
n_channels = len(st)
sr = st[0].stats.sampling_rate # We are assuming that all traces have the same sr
window_length = 3600.0     # s. Length of the windows in seconds
window_samples = int(window_length*sr)
shift  = window_length/2  # s. Length of the window shift in seconds
n_bins = int(window_samples);  # Number of bins to use for the STFT calculation

n_windows = int(st[0].stats.npts/window_samples) # We are assuming that all traces have the same number of points
n_freqs = int((n_bins//2))

# %% Calculate the STFT (Spectrogram)
print(f"Calculating STFT ...")
times_stft = np.zeros((n_channels, n_windows))
freqs_stft = np.zeros((n_channels,n_freqs))
Zre = np.zeros((n_channels,n_freqs, n_windows))
Zim = np.zeros((n_channels,n_freqs, n_windows))
Sxx = np.zeros((n_channels,n_freqs, n_windows))

signals = []
for i in range(len(st)):
    signals.append(st[i].data)

startime = st[-1].stats.starttime.datetime
endtime = st[-1].stats.endtime.datetime
# del st

for i in range(n_channels):
    print(f"Computing for trace {i}...")
    time, freq, Sx = features.stft(st[i].data, sr, window_samples, "hann", "odd",
                                   detrend,n_bins, t_phase=window_length/2)
    times_stft[i,:] = time[:-1]
    freqs_stft[i,:] = freq[:-1]
    Zre[i,:,:] = Sx[:-1, :-1].real
    Zim[i,:,:] = Sx[:-1, :-1].imag
Sxx = Zre**2 + Zim**2


# %% Plot the Spectrograms
if show_plots:
    print("Plotting spectrogram")

    fig, axes = plt.subplots(n_channels, sharex=True, figsize=(13,2))
    if n_channels==1: axes = [axes]
    for i in range(n_channels):
        _, axes[i], mesh = plot.spectrogram(times_stft[i], freqs_stft[i], Sxx[i], ax=axes[i],
                                            # vmin=0, vmax=np.max(Sxx),
                                            logscale=True)
        # axes[i].set_ylabel(f"{st[i].stats.channel}")
        axes[i].tick_params(axis='y', labelsize=18)



    # timeUTC = np.linspace(startime, endtime, num= len(time)-1, dtype='datetime64[h]').astype("datetime64[m]")
    timeUTC = np.linspace(startime, endtime, num= len(times_stft[0]), dtype='datetime64[h]').astype("datetime64[m]")

    aux = mdates.date2num(timeUTC)

    skip:int = 3
    locs = times_stft[0, 0::skip]
    timeUTC = timeUTC[0::skip]

    date_formatter = mdates.DateFormatter('%H:%M')
    formatted_labels = [date_formatter(mdates.date2num(t)) for t in timeUTC]

    axes[-1].set_xticks(locs[1:], labels=formatted_labels[1:],fontsize=18)
    # fig.autofmt_xdate()

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # cbar = fig.colorbar(mesh, cax=cbar_ax)
    # cbar.ax.tick_params(labelsize=18) 

    # fig.supxlabel("Time")
    fig.supylabel("Frequency (Hz)", fontsize=18, x =.07)
    fig.suptitle("Spectrograms")
    fig.suptitle(title, fontsize=24, y =.95)
    # fig.show()
    fig.savefig("external/tutorials/involcan_spectrogram.png", bbox_inches='tight')

# %% Save Spectrogram data
if save_data:
    for i in range(n_channels):
        features.save(f"data/spectrogram_channel_{st[i].stats.channel}.mat",np.squeeze(Sxx[i]).T,row_names=times_stft[i],column_names=freqs_stft[i])
        print(f"Data saved at: data/spectrogram_channel_{st[i].stats.channel}.mat")




# %%

fig, axes = plt.subplots(2,1, figsize = (14,3), sharex=True)
_,_,mesh = plot.spectrogram(times_stft[i], freqs_stft[i], Sxx[i], ax=axes[1],
                                            # vmin=0, vmax=np.max(Sxx),
                                            logscale=True)
timeUTC = np.linspace(startime, endtime, num= len(times_stft[0]), dtype='datetime64[h]').astype("datetime64[m]")
aux = mdates.date2num(timeUTC)
skip:int = 3
locs = times_stft[0, 0::skip]
timeUTC = timeUTC[0::skip]
date_formatter = mdates.DateFormatter('%H:%M')
formatted_labels = [date_formatter(mdates.date2num(t)) for t in timeUTC]
axes[1].set_xticks(locs[1:], labels=formatted_labels[1:],fontsize=18)

signal = st[0].data

axes[0].plot(np.linspace(0, np.max(times_stft), len(signal)),signal, c='black')
axes[0].set_xlim(0, np.max(times_stft))

labelsize = 14
axes[0].set_ylabel("Amplitude", fontsize = labelsize)
axes[1].set_ylabel("Frequency \n(Hz)", fontsize = labelsize)

ticksize = 12
axes[0].tick_params(axis='y', labelsize=ticksize)
axes[1].tick_params(axis='y', labelsize=ticksize)


tbox = axes[0].get_position()
bbox = axes[1].get_position()

fig.subplots_adjust(right=0.86)
cbar_ax = fig.add_axes([0.87, bbox.y0, 0.01, bbox.height])
cbar = fig.colorbar(mesh, cax=cbar_ax)
cbar.ax.tick_params(labelsize=ticksize) 
cbar.ax.yaxis.get_offset_text().set_fontsize(ticksize)
offset_text = cbar.ax.yaxis.get_offset_text()
offset_text.set_position((2.5, offset_text.get_position()[1]))


fig.savefig("external/tutorials/double_plot.png", bbox_inches='tight')

input("End?")