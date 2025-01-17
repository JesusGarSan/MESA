import matplotlib.pyplot as plt
import numpy as np

def signal(x, y):
    fig, ax = plt.subplots()
    plt.title("Raw signal")
    plt.plot(x,y)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid()
    return fig

def fft(fft_freq, fft):
    # Only positive values for plotting
    fft_freq_positive = fft_freq[0:len(fft_freq)//2]
    fft_positive = fft[0:len(fft)//2]

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Fast Fourier Transform")
    ax[0].set_title("Real part")
    ax[0].bar(fft_freq_positive, np.abs(np.real(fft_positive)))
    ax[0].grid()
    ax[1].set_title("Imaginary part")
    ax[1].bar(fft_freq_positive, np.abs(np.imag(fft_positive)))
    ax[1].grid()
    ax[1].set_xlabel("Frequencies (Hz)")
    return fig