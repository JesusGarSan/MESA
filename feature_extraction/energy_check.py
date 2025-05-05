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

def bins_check(sr, t, n_bins):
    # sr:       (Hz) Sampling rate
    # t:        (s) Duration of the signal
    # n_bins:   Number of different frequencies composing the signal
    if n_bins < int(sr*t):
        print(f"Warning!\n  The number of FFT bins ({n_bins}) is smaller than the number of time samples ({int(sr*t)}). The energy might not be conserved")
    if n_bins > int(sr*t):
        print(f"Warning!\n  The number of FFT bins ({n_bins}) is greater than the number of time samples ({int(sr*t)}). The resulting frequency resolution will not be accurate")
    return

def get_bins(sr, t):
    return int(sr*t)