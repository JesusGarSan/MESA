import matplotlib.pyplot as plt
import numpy as np

def signal(x, y):
    fig, ax = plt.subplots()
    plt.title("Raw signal")
    plt.plot(x,y)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.xlim(x[0], x[-1])
    return fig

def fft(fft_freq, fft):
    # Only positive values for plotting
    fft_freq_positive = fft_freq[0:len(fft_freq)//2]
    fft_positive = fft[0:len(fft)//2]
    bar_width = (fft_freq_positive[-1] - fft_freq_positive[0])/len(fft_freq_positive)

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Fast Fourier Transform")

    ax[0].set_title("Real part")
    ax[0].bar(fft_freq_positive, np.abs(np.real(fft_positive)), bar_width)
    ax[0].grid()
    ax[0].set_xlim(fft_freq_positive[0]-1, fft_freq_positive[-1]+1)

    ax[1].set_title("Imaginary part")
    ax[1].bar(fft_freq_positive, np.abs(np.imag(fft_positive)), bar_width)
    ax[1].set_xlim(fft_freq_positive[0]-1, fft_freq_positive[-1]+1)
    ax[1].grid()
    ax[1].set_xlabel("Frequencies (Hz)")
    return fig