import numpy as np
import scipy
import matplotlib.pyplot as plt

from simule_data import *
from feature_extraction import *



if __name__ == "__main__":
    print("Running main ...")

    N = 1000;
    freq = generate_frequencies(N);
    A = generate_amplitudes(N);
    x, y = generate_signal(freq, A, t=100)

    fig, ax = plt.subplots()
    plt.title("Raw signal")
    plt.plot(x,y)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    fft, fft_freq = fft_bin(y, 100, 100)
    # Keep only the positive values
    fft_freq = fft_freq[0:len(fft_freq)//2]
    fft = fft[0:len(fft)//2]
    
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Fast Fourier Transform")
    ax[0].set_title("Real part")
    ax[0].bar(fft_freq, np.abs(np.real(fft)))
    ax[0].grid()
    ax[1].set_title("Imaginary part")
    ax[1].bar(fft_freq, np.abs(np.imag(fft)))
    ax[1].grid()
    ax[1].set_xlabel("Frequencies (Hz)")
    plt.show()



