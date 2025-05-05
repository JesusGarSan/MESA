import feature_extraction.energy_check
import feature_extraction.features
import simulation.generator as generator
import feature_extraction
from feature_extraction.energy_check import get_bins

import numpy as np

N, sr, t = 10, 100, 2
f0 = [5., 15., 25.]
A0 = [10, 20, 15]

F = generator.generate_frequencies(N, sigma=0.0, f0=f0)
A = generator.generate_amplitudes(N, A0,sigma=0.)
phi=0.0

x, y = generator.generate_signal(F, A, sr, t, phi)

def test_fft_bin_energy():
    """ Check energy conservation with correct binning. """
    fft, freqs = feature_extraction.features.fft_bin(signal=y, n_bins=get_bins(sr, t), sr=sr)

    E_t = feature_extraction.energy_check.energy_t(y)
    E_f = feature_extraction.energy_check.energy_f(fft)

    print(E_t, E_f)
    assert np.isclose(E_t, E_f)

def test_fft_bin_under_energy():
    """ Check energy conservation with correct under-binning. """
    fft, freqs = feature_extraction.features.fft_bin(signal=y, n_bins=get_bins(sr, t)//10, sr=sr)

    E_t = feature_extraction.energy_check.energy_t(y)
    E_f = feature_extraction.energy_check.energy_f(fft)

    print(E_t, E_f)
    assert (E_t > E_f)


def test_save():
    import os
    path = "tests/data/"
    if not os.path.exists(path): os.makedirs(path)
    fft, freqs = feature_extraction.features.fft_bin(signal=y, n_bins=get_bins(sr, t), sr=sr)

    print(freqs.shape, fft.shape)
    feature_extraction.features.save(fft, freqs, path+"matrix.mat")
    assert os.path.exists(path+"matrix.mat")

