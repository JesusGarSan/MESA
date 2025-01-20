import numpy as np

def energy_t(signal):
    E_t = np.sum(np.abs(signal)**2)
    return E_t


def energy_f(fft):
    N = len(fft)
    E_f = (np.sum(np.abs(fft)**2))/N
    return E_f

def Parseval(signal, fft, verbose = True):
    E_t = energy_t(signal)
    E_f = energy_f(fft)

    if verbose:
        print(f"""
        Energy time domain:      {E_t}
        Energy frequency domain: {E_f}
        Ratio:                   {E_t/E_f}
        """)   

    assert np.isclose(E_t, E_f), f"The energy is not conserved. Ratio: {E_t/E_f}"

    return