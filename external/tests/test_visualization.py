from mesa.visualization import spectrogram, plot_signal, dual_plot
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

def test_plot_signal_minimal_call():
    fig  = plot_signal(y, sr)
    plt.show()
    return

def test_spectrogram_minimal_call():
    fig, _  = spectrogram(Sxx,sr,win_length)
    plt.show()
    return

def test_dual_plot_minimal_call():
    fig = dual_plot(y, Sxx,sr,win_length)
    ax = fig.get_axes()
    # ax[0].set_xlim([0,2])
    plt.show()
    return
