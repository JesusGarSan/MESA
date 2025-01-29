# %%
import numpy as np
import matplotlib.pyplot as plt

from functions.simule_data import *
from functions.feature_extraction import *
import functions.plot
from functions.tests import Parseval
from Processor import Processor


if __name__ == "__main__":
    print("Running main ...")

    """ Set the parameters """
    sr = 100;       # Hz. Sampling rate
    t = 10;         # s. Duration of the signal
    n_bins = 1000;  # Number of bins to use for the DFT calculation
    N = 10000;      # Number of different frequencies composing the signal

    
    """ Generate the signal """
    freq = generate_frequencies(N, sr=sr);
    A = generate_amplitudes(N, sigma= 10);
    x, y = generate_signal(freq, A, t=t, sr=sr);


    """ Add high frequency noise """
    AN = generate_amplitudes(10);
    f0 = np.random.normal(1000, 0.1, 10)
    _, noise = generate_signal(f0, AN, t=t, sr=sr)
    y += noise;

    """ Create signal object """
    signal = Processor(y, sr)
    signal.check()

    """ Plot the signal and its FFT """
    # Plot the signal
    signal.plot_signal()
    plt.show(block=False)

    # Calculate the FFTs
    signal.fft_bin(n_bins)
    # Plot the FFTs
    signal.plot_fft()
    plt.show(block=False)

    """ Run checks """
    signal.Parseval()
    
    """ Apply windowing and calculate FFT"""
    signal.set_windows(1, 0.5);
    signal.fft_bin_window(n_bins)

    input()
    # %%
    """ Create data matrix """
    ffts = signal.fft_windows[:, 0:n_bins//2]
    fft_freq = signal.freqs_windows[0:n_bins//2]
    fft_freq_str = [f"{round(freq, 2)} Hz" for freq in fft_freq]
    # save(np.abs(ffts), column_names=fft_freq, filepath="./data/")


