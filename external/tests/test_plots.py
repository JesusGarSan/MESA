from internal.simulation import generator
from internal.feature_extraction import features 
from internal.feature_extraction.energy_check import get_bins
from internal.visualization import plot

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

