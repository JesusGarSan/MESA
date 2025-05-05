"""
We want to parallelize each of the windwos
We want to segment the signs in windows.
We want to be able to apply overlap and windowing
"""
import numpy as np
import scipy
import scipy.signal

def fft_bin(signal:float, n_bins:int, sr:float):
    fft = np.fft.fft(signal, n=n_bins)
    freqs = np.fft.fftfreq(n_bins, d=1/sr)    
    return fft, freqs


def save(matrix, column_names = None, filepath=".data/matrix.mat"):
    dict = {'matrix': matrix}
    # Turn 1D arrays into row arrays
    if len(matrix.shape) == 1: matrix = matrix[np.newaxis, :]

    if column_names is not None:
        if len(column_names) != matrix.shape[1]:
            raise ValueError("The number of column names does not match the number of columns in the matrix.")
        dict['column_names'] = column_names

    try:
        scipy.io.savemat(filepath, dict)
    except Exception as e:
        print(e)
        return False
        
    return True





if __name__ == '__main__':
    import os, sys
    sys.path.insert(1, os.getcwd())
    import simulation.generator as generator
    import feature_extraction.features
    from feature_extraction.energy_check import get_bins
    from visualization import plot

    N, sr, t = 10, 100, 10
    f0 = [5., 15., 25.]
    A0 = [10, 20, 15]

    F = generator.generate_frequencies(N, sigma=0.0, f0=f0)
    A = generator.generate_amplitudes(N, A0,sigma=0.)
    phi=0.0

    x, y = generator.generate_signal(F, A, sr, t, phi)

    x_aux = x - t/2 # Peak on the middle of the signal
    convolution = 1* np.exp(-(x_aux/10)**2)
    y *= convolution

    fig = plot.signal(x,y)
    fig.show()
    
    from scipy.signal import ShortTimeFFT

    win_samples = int(sr*1)

    win = scipy.signal.get_window("boxcar", win_samples)
    SFT = ShortTimeFFT(win,win_samples,sr)

    # Compute the STFT
    Zxx = SFT.spectrogram(y,padding="odd")


    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.pcolormesh(range(len(Zxx[0])), SFT.f, Zxx)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()
    plt.show()
