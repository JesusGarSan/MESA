from src.msa.simulation import generator
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

F = generator.generate_frequencies(N, sigma=0.0, f0=f0)
A = generator.generate_amplitudes(N, A0,sigma=0.)
phi = generator.generate_phase(N*len(f0))

x, y = generator.generate_signal(F, A, sr, t, phi)

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
    F = generator.generate_frequencies(N, sigma=0.0, f0=f0)
    A = generator.generate_amplitudes(N, A0,sigma=0.0)
    phi=0
    x, y = generator.generate_signal(F, A, sr, t, phi)

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

    signal = generator.generate_pulse(A,b,t0,t,w_k,phi)
    _,ax[0] = plot.signal(t, signal, ax=ax[0])


    win_length = .1 #s 1/T
    win_samples = int(win_length*sr)

    time, freq, Sxx = features.spectrogram(signal, sr, win_samples,"hann", "odd", t_phase =win_length/2)
    _, ax[1], mesh =plot.spectrogram(time, freq, Sxx,logscale=False, ax=ax[1])
    
    
    
    fig.suptitle("Pulse with exponential decay\n" + f"$\Delta f = {1/win_length}$, $\Delta T = {win_length}$")
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

def test_non_stationary():

    A = 11.2
    f0, f_max = 2, 15
    sr = 1000
    T = 10
    t = np.linspace(0, T, int(sr*T))
    phi = 0.0

    fig, ax = plt.subplots(2,1, figsize=(10,4))

    signal = generator.generate_non_stationary(A, f0, f_max, sr, T, phi)
    _,ax[0] = plot.signal(t, signal, ax = ax[0])



    win_length = .1 #s
    win_samples = int(win_length*sr)

    time, freq, Sxx = features.spectrogram(signal, sr, win_samples,"hann", "odd", t_phase =win_length/2)
    _, ax[1], mesh =plot.spectrogram(time, freq, Sxx,logscale=False, ax = ax[1],ylim=(0,20))
    
    fig.suptitle("Non stationary signal\n" + f"$\Delta f = {1/win_length}$, $\Delta T = {win_length}$")
    ax[0].set_ylabel("Amplitude", )
    ax[1].set_ylabel("Frequency \n(Hz)", )
    ax[1].set_xlabel("Time (s)", )
    fig.subplots_adjust(right=0.86)
    tbox = ax[0].get_position()
    bbox = ax[1].get_position()
    cbar_ax = fig.add_axes([0.87, bbox.y0, 0.01, bbox.height])
    cbar = fig.colorbar(mesh, cax=cbar_ax)

    fig.savefig(path+"/non_stationary.png")

    return


def test_undefined_sin():
    sr = 5000
    T = 0.5
    t0 = 0.01
    t = np.linspace(t0,T, sr)

    fig, ax = plt.subplots(2,1, figsize=(10,4))

    signal = generator.generate_undefined_sin(t, b = 0)

    _,ax[0] = plot.signal(t, signal, ax = ax[0])

    win_length = 0.01 #s
    win_samples = int(win_length*sr)

    time, freq, Sxx = features.spectrogram(signal, sr, win_samples,"boxcar", "odd", t_phase =win_length/2)
    _, ax[1], mesh =plot.spectrogram(time, freq, Sxx,logscale=False, ax = ax[1])

    fig.suptitle("sin(1/x)\n" + f"$\Delta f = {1/win_length}$, $\Delta T = {win_length}$")
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