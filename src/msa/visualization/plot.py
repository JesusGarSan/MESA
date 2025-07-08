import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def signal(x, y, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle("Raw signal")
        fig.supxlabel("Time")
        fig.supylabel("Amplitude")
    else: fig = None
    ax.plot(x,y)
    ax.grid()
    ax.set_xlim(x[0], x[-1])
    return fig, ax

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


def spectrogram(t:np.ndarray, f:np.ndarray, Sxx:np.ndarray, xlim:tuple=None, ylim:tuple=None, show:bool = False, vmin:float=None, vmax:float=None, logscale=False, ax = None):
    """
    Plots a spectrogram

    Args:
        t (array_like): The times of the measurement.
        f (array_like): The frequencies.
        Sxx (array_like): The power spectral density.
        xlim (tuple, optional): Limits for the x-axis (time). Defaults to None.
        show (bool, optional): If True, displays the plot. Defaults to False.
        vmin (float, optional): Minimum value for the color scale. Defaults to None.
        vmax (float, optional): Maximum value for the color scale. Defaults to None.
        logscale (bool, optional): If True, applies a logarithmic color scale. Defaults to False.

    Returns:
        tuple: (fig, ax, mesh) - The figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots()
        # fig.suptitle("Raw signal")
        fig.supxlabel("Time (s)")
        fig.supylabel("Frequency (Hz)")
    else: fig = None

    if logscale:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)  # Use LogNorm for logarithmic scaling
        mesh = ax.pcolormesh(t, f, Sxx, norm=norm)
    else:
        mesh = ax.pcolormesh(t, f, Sxx, vmin=vmin, vmax=vmax) # default linear scale

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    mesh.set_clim(vmin,vmax)

    if fig is not None:
        fig.colorbar(mesh, ax = ax)
    
    if show: plt.show()
    
    return fig, ax, mesh