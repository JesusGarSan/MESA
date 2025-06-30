import src.msa.simulation.generator as generator

def test_generator(N=10, sr = 10, t = 1.5, **kwargs):
    freq = generator.generate_frequencies(N, **kwargs)
    A = generator.generate_amplitudes(N, **kwargs)
    phi = generator.generate_phase(N, **kwargs)
    x, y = generator.generate_signal(freq, A, sr, t, phi)

    assert x.shape == y.shape
    assert x.shape[0] == int(sr*t)


