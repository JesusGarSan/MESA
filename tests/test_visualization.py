from mesa.visualization import spectrogram, plot_signal, dual_plot, multi_spectrogram
from mesa.features import stft
import numpy as np

sr = 200
t0 = 0
t1 = 3
x = np.linspace(t0, t1, sr*(t1-t0))
y = np.cos(x)
win_length = int(sr*0.25)
D = stft(y, win_length)
Sxx = np.abs(D)**2

import matplotlib.pyplot as plt
show_figure = False


def test_plot_signal_minimal_call():
    fig  = plot_signal(y, sr)
    if show_figure: plt.show()
    return

def test_spectrogram_minimal_call():
    fig, _  = spectrogram(Sxx,sr,win_length)
    if show_figure: plt.show()
    return

def test_spectrogram_centered():
    fig, _  = spectrogram(Sxx,sr,win_length, center=True)
    if show_figure: plt.show()
    return

def test_spectrogram_hop():
    hop = int(sr*.15)
    D = stft(y, win_length, hop)
    Sxx = np.abs(D)**2

    fig, _  = spectrogram(Sxx,sr,win_length, hop)
    if show_figure: plt.show()
    return

def test_spectrogram_hop_centered():
    hop = int(sr*.15)
    D = stft(y, win_length, hop, center=True)
    Sxx = np.abs(D)**2

    fig, _  = spectrogram(Sxx,sr,win_length, hop, center=True)
    if show_figure: plt.show()
    return

def test_dual_plot_minimal_call():
    fig = dual_plot(y, Sxx,sr,win_length)
    ax = fig.get_axes()
    # ax[1].set_xlim([-.25,3.25])
    if show_figure: plt.show()
    return

def test_dual_plot_hop():
    hop = int(sr*.15)
    D = stft(y, win_length, hop)
    Sxx = np.abs(D)**2

    fig = dual_plot(y, Sxx,sr,win_length,hop)
    ax = fig.get_axes()
    # ax[1].set_xlim([-.25,3.25])
    if show_figure: plt.show()
    return

def test_dual_plot_hop_centered():
    hop = int(sr*.15)
    D = stft(y, win_length, hop, center=True)
    Sxx = np.abs(D)**2

    fig = dual_plot(y, Sxx,sr,win_length,hop, center=True)
    ax = fig.get_axes()
    # ax[1].set_xlim([-.25,3.25])
    if show_figure: plt.show()
    return

def test_multi_spectrogram_minimal_call():
    Sxx_multi = Sxx[np.newaxis, np.newaxis, :,:]
    Sxx_multi = np.tile(Sxx_multi, (3,2,1,1,))
    Sxx_multi[1,1,0:2,0:3] = 300
    Sxx_multi[0,0] = 0*Sxx_multi[0,0]
    fig, axes = multi_spectrogram(Sxx_multi, rows=[1], cols=[0],
                                  sr = sr, win_length=win_length)
    for row in axes:
        row[0].set_ylim(0,20)
    if show_figure: plt.show()

    return

def test_spectrogram_trimmed():
    Sxx_trimmed = Sxx[0:11,:]
    fig, _  = spectrogram(Sxx_trimmed,sr,win_length, y_coords = np.arange(11))
    if show_figure: plt.show()
    return
