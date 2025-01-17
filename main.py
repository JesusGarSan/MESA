import numpy as np
import scipy
import matplotlib.pyplot as plt

from simule_data import *
from feature_extraction import *
import plot
from tests import Parseval


if __name__ == "__main__":
    print("Running main ...")

    """ Set the parameters """
    sr = 100;       # Hz. Sampling rate
    t = 1;          # s. Duration of the signal
    n_bins = 200;   # Number of bins to use for the DFT calculation
    N = 10000;      # Number of different frequencies composing the signal

    
    """ Generate the signal """
    freq = generate_frequencies(N, sr=sr);
    A = generate_amplitudes(N);
    x, y = generate_signal(freq, A, t=t, sr=sr);


    # Calculate the FFTs
    fft, fft_freq = fft_bin(y, n_bins=n_bins, sr=sr)

    # Plot the signal
    plot.signal(x,y)
    plt.show(block=False)

    # Plot the FFTs
    plot.fft(fft_freq, fft)
    plt.show(block=False)

    # Run checks
    Parseval(y, fft)
    



    input()


