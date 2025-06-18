from src.feature_extraction import energy_check
from src.simulation import generator
from src.feature_extraction import features 
from src.feature_extraction.energy_check import get_bins
from src.visualization import plot

import numpy as np
import os

path = "external/tests/data/"

N, sr, t = 10, 100, 2
f0 = [5., 15., 25.]
A0 = [10, 20, 15]

F = generator.generate_frequencies(N, sigma=0.0, f0=f0)
A = generator.generate_amplitudes(N, A0,sigma=0.)
phi=0.0

x, y = generator.generate_signal(F, A, sr, t, phi)

def test_fft_bin_energy():
    """ Check energy conservation with correct binning. """
    fft, freqs = features.fft_bin(signal=y, n_bins=get_bins(sr, t), sr=sr)

    E_t = energy_check.energy_t(y)
    E_f = energy_check.energy_f(fft)

    print(E_t, E_f)
    assert np.isclose(E_t, E_f)

def test_fft_bin_under_energy():
    """ Check energy conservation with correct under-binning. """
    fft, freqs = features.fft_bin(signal=y, n_bins=get_bins(sr, t)//10, sr=sr)

    E_t = energy_check.energy_t(y)
    E_f = energy_check.energy_f(fft)

    print(E_t, E_f)
    assert (E_t > E_f)


def test_save():
    if not os.path.exists(path): os.makedirs(path)
    fft, freqs = features.fft_bin(signal=y, n_bins=get_bins(sr, t), sr=sr)

    features.save(path+"matrix.mat",fft, column_names=freqs)
    assert os.path.exists(path+"matrix.mat")



def test_stft_spectrogram_equivalence():
    N, sr, t = 10, 100, 10
    F = generator.generate_frequencies(N, sigma=10.0, f0=f0)
    A = generator.generate_amplitudes(N, A0,sigma=10.)
    x, y = generator.generate_signal(F, A, sr, t, phi)

    x_aux = x - t/2 # Peak on the middle of the signal
    convolution = 1* np.exp(-(x_aux/10)**2)
    y*=convolution

    win_length = 1 #s
    win_samples = int(win_length*sr)

    time, freq, Sxx = features.spectrogram(y, sr, win_samples,"boxcar", "odd", t_phase =win_length/2, detrend="constant")
    time, freq, Zxx = features.stft(y, sr, win_samples,"boxcar", "odd", t_phase =win_length/2, detrend="constant")
    assert np.isclose(np.mean(Sxx), np.mean(Zxx.imag**2+Zxx.real**2))
