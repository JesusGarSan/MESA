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

def fft(fft_freq, fft, mode = 'module'):
    # Only positive values for plotting
    fft_freq_positive = fft_freq[0:len(fft_freq)//2]
    fft_positive = fft[0:len(fft)//2]
    bar_width = (fft_freq_positive[-1] - fft_freq_positive[0])/len(fft_freq_positive)

    if mode == 'module':
        fig, ax = plt.subplots()
        fig.suptitle("Fast Fourier Transform")
        ax.bar(fft_freq_positive, np.abs(fft_positive), bar_width)
        ax.set_xlim(fft_freq_positive[0]-1, fft_freq_positive[-1]+1)
        ax.grid()
        ax.set_xlabel("Frequencies (Hz)")

    if mode == 'unfold':
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


def spectrogram(t, f, Sxx, xlim=None, show = False):
    fig, ax = plt.subplots()
    plt.pcolormesh(t, f, Sxx)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()
    if show: plt.show()
    return fig, ax