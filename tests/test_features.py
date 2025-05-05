import feature_extraction.energy_check
import feature_extraction.features
import simulation.generator as generator
import feature_extraction
import visualization.plot as plot

import numpy as np
# np.random.seed(123)

N, sr, t = 10, 100, 2
f0 = [5., 15.]
A0 = [10, 20]

F = generator.generate_frequencies(N, sigma=0.0, f0=f0)
A = generator.generate_amplitudes(N, A0,sigma=0.)
phi = generator.generate_phase(N*len(f0),)
phi=0.0

x, y = generator.generate_signal(F, A, sr, t, phi)

def test_fft_bin():
    fft, freqs = feature_extraction.features.fft_bin(signal=y, n_bins=sr*t, sr=sr)

    fig = plot.fft(freqs, fft)
    fig.savefig("tests/plots/test_fft_bin.png")

    E_t = feature_extraction.energy_check.energy_t(y)
    E_f = feature_extraction.energy_check.energy_f(fft)

    print(E_t, E_f)
    assert np.isclose(E_t, E_f)


    return


