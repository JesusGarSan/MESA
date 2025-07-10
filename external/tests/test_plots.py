from msa.simulation import generate
from src.msa.feature_extraction import features 
from src.msa.feature_extraction.energy_check import get_bins
from src.msa.visualization import plot

import numpy as np
import matplotlib.pyplot as plt


import os
path = "external/tests/plots/"
if not os.path.exists(path): os.makedirs(path)

from pathlib import Path
Path(path).mkdir(parents=True, exist_ok=True)

N, sr, t = 10, 100, 2
f0 = [5., 15., 25.]
A0 = [10, 20, 15]

F = generate.frequencies(N, sigma=0.0, f0=f0)
A = generate.amplitudes(N, A0,sigma=0.)
phi = generate.phase(N*len(f0))

x, y = generate.signal(F, A, sr, t, phi)

def test_plot_signal():
    fig, ax = plot.signal(x,y)
    fig.savefig(path+"signal.png")
    assert os.path.exists(path+"signal.png")

def test_plot_fft():
    fft, freqs = features.fft_bin(signal=y, n_bins=get_bins(sr, t), sr=sr)

    fig = plot.fft(freqs, fft,"module")
    fig.savefig(path+"fft_module.png")
    assert os.path.exists(path+"fft_module.png")

    fig = plot.fft(freqs, fft,"unfold")
    fig.savefig(path+"fft_unfolded.png")
    assert os.path.exists(path+"fft_unfolded.png")

def test_plot_spectrogram():
    path = "external/tests/plots/"
    if not os.path.exists(path): os.makedirs(path)

    N, sr, t = 10, 100, 3
    f0 = [5., 15., 25.]
    A0 = [10, 20, 15]
    F = generate.frequencies(N, sigma=0.0, f0=f0)
    A = generate.amplitudes(N, A0,sigma=0.0)
    phi=0
    x, y = generate.signal(F, A, sr, t, phi)

    x_aux = x - t/2 # Peak on the middle of the signal
    convolution = 1* np.exp(-(x_aux/10)**2)
    y*=convolution

    win_length = 1 #s
    win_samples = int(win_length*sr)

    time, freq, Sxx = features.spectrogram(y, sr, win_samples,"boxcar", "odd", t_phase =win_length/2)
    fig, ax, _ =plot.spectrogram(time, freq, Sxx)
    fig.savefig(path+"/spectrogram.png")
    assert os.path.exists(path+"spectrogram.png")


def test_pulse():

    sr = 200
    T = 4

    t0=1
    t = np.linspace(0, T, int(T*sr))
    A = 11.2
    b = 1.7
    w_k = 2*np.pi * 6.1
    phi = .0

    fig, ax = plt.subplots(2,1, figsize=(10,4))

    signal = generate.pulse(A,b,t0,t,w_k,phi)
    _,ax[0] = plot.signal(t, signal, ax=ax[0])


    win_length = .1 #s 
    win_samples = int(win_length*sr)

    time, freq, Sxx = features.spectrogram(signal, sr, win_samples,"boxcar", "odd", t_phase =win_length/2)
    _, ax[1], mesh =plot.spectrogram(time, freq, Sxx,logscale=False, ax=ax[1])
    
    
    
    fig.suptitle("Pulse with exponential decay\n" + fr"$\Delta f = {1/win_length}$, $\Delta T = {win_length}$")
    ax[0].set_ylabel("Amplitude", )
    ax[1].set_ylabel("Frequency \n(Hz)", )
    ax[1].set_xlabel("Time (s)", )
    fig.subplots_adjust(right=0.86)
    tbox = ax[0].get_position()
    bbox = ax[1].get_position()
    cbar_ax = fig.add_axes([0.87, bbox.y0, 0.01, bbox.height])
    cbar = fig.colorbar(mesh, cax=cbar_ax)

    fig.savefig(path+"/pulse.png")


    return

def test_chirp():

    A = 11.2
    f0, f_max = 2, 15
    sr = 1000
    T = 10
    t = np.linspace(0, T, int(sr*T))
    phi = 0.0

    fig, ax = plt.subplots(2,1, figsize=(10,4))

    signal = generate.chirp(t, 1, 1, 5, 2, 8, 'linear', 0, 8)
    _,ax[0] = plot.signal(t, signal, ax = ax[0])



    win_length = .5 #s
    win_samples = int(win_length*sr)

    time, freq, Sxx = features.spectrogram(signal, sr, win_samples, win_samples,"boxcar", "odd", t_phase =win_length/2)
    _, ax[1], mesh =plot.spectrogram(time, freq, Sxx,logscale=False, ax = ax[1],ylim=(0,20))
    
    fig.suptitle("Chirp\n" + fr"$\Delta f = {1/win_length}$, $\Delta T = {win_length}$")
    ax[0].set_ylabel("Amplitude", )
    ax[1].set_ylabel("Frequency \n(Hz)", )
    ax[1].set_xlabel("Time (s)", )
    fig.subplots_adjust(right=0.86)
    tbox = ax[0].get_position()
    bbox = ax[1].get_position()
    cbar_ax = fig.add_axes([0.87, bbox.y0, 0.01, bbox.height])
    cbar = fig.colorbar(mesh, cax=cbar_ax)

    fig.savefig(path+"/chirp.png")
    plt.show()

    return


def test_undefined_sin():
    sr = 1000
    T = 0.2
    t0 = 0.01
    t = np.linspace(t0,T, sr)

    fig, ax = plt.subplots(2,1, figsize=(10,4))

    signal = generate.chirp_sin(t, b = 0)

    _,ax[0] = plot.signal(t, signal, ax = ax[0])

    win_length = 0.02 #s
    win_samples = int(win_length*sr)

    time, freq, Sxx = features.spectrogram(signal, sr, win_samples,"boxcar", "odd", t_phase =win_length/2)
    _, ax[1], mesh =plot.spectrogram(time, freq, Sxx,logscale=False, ax = ax[1], ylim=(0,200))

    fig.suptitle("sin(1/x)\n" + fr"$\Delta f = {1/win_length}$, $\Delta T = {win_length}$")
    ax[0].set_ylabel("Amplitude", )
    ax[1].set_ylabel("Frequency \n(Hz)", )
    ax[1].set_xlabel("Time (s)", )
    fig.subplots_adjust(right=0.86)
    tbox = ax[0].get_position()
    bbox = ax[1].get_position()
    cbar_ax = fig.add_axes([0.87, bbox.y0, 0.01, bbox.height])
    cbar = fig.colorbar(mesh, cax=cbar_ax)

    fig.savefig(path+"/undefined_sin.png")



    return

def test_spectrogram_grid():

    signal = np.random.rand(600)
    sr=100
    T = 60
    t = np.linspace(0, T, int(sr*T))

    N, M = 2,3
    
    Sxxs=[]
    times = []
    freqs=[]

    windows = [1,3,6,10]
    shift_fractions = [1/1, 1/3, 1/5, 1/10]

    for window in windows:
        for shift_fraction in shift_fractions:
            win_samples = int(sr*window)
            hop = int(win_samples*shift_fraction)
            time, freq, Sxx = features.spectrogram(signal, sr, win_samples, hop, "boxcar","odd", "linear")
            times.append(time)
            freqs.append(freq)
            Sxxs.append(Sxx)

    fig, ax = plot.grid(N, M, text0="Horizontal", arrow0=(0.05, .90))
    for n in range(N):
        for m in range(M):
            plot.spectrogram(times[n+m], freqs[n+m], Sxxs[n+m], ax=ax[n,m])

    plt.show()
    return