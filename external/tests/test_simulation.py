import msa.simulation.generate as generate

def test_generator(N=10, sr = 10, t = 1.5, **kwargs):
    freq = generate.generate_frequencies(N, **kwargs)
    A = generate.generate_amplitudes(N, **kwargs)
    phi = generate.generate_phase(N, **kwargs)
    x, y = generate.generate_signal(freq, A, sr, t, phi)

    assert x.shape == y.shape
    assert x.shape[0] == int(sr*t)


