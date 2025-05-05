import feature_extraction.energy_check
import feature_extraction.features
import simulation.generator as generator
import feature_extraction
from feature_extraction.energy_check import get_bins
import visualization.plot as plot

import numpy as np

N, sr, t = 10, 100, 2
f0 = [5., 15., 25.]
A0 = [10, 20, 15]

F = generator.generate_frequencies(N, sigma=0.0, f0=f0)
A = generator.generate_amplitudes(N, A0,sigma=0.)
phi = generator.generate_phase(N*len(f0))

x, y = generator.generate_signal(F, A, sr, t, phi)

def test_plot_signal():
    fig = plot.signal(x,y)
    fig.savefig("tests/plots/signal.png")

    fft, freqs = feature_extraction.features.fft_bin(signal=y, n_bins=get_bins(sr, t), sr=sr)
    fig = plot.fft(freqs, fft,"module")
    fig.savefig("tests/plots/fft_module.png")
    fig = plot.fft(freqs, fft,"unfold")
    fig.savefig("tests/plots/fft_unfolded.png")

