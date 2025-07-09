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


def spectrogram(t:np.ndarray, f:np.ndarray, Sxx:np.ndarray, xlim:tuple=None, ylim:tuple=None, cmap = 'viridis', show:bool = False, vmin:float=None, vmax:float=None, logscale=False, ax = None):
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
        mesh = ax.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap)
    else:
        mesh = ax.pcolormesh(t, f, Sxx, vmin=vmin, vmax=vmax, cmap=cmap) # default linear scale

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    mesh.set_clim(vmin,vmax)

    if fig is not None:
        fig.colorbar(mesh, ax = ax)
    
    if show: plt.show()
    
    return fig, ax, mesh


def grid(n_rows:int, n_columns:int, figsize:tuple=(10,10), axis:bool = True, x0:float=0.05, x1:float=0.95, text0:str = '', text1:str=''):

    fig, ax = plt.subplots(n_rows,n_columns, figsize=figsize, sharex=True, sharey=True)

    ax_aux = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ax_aux.axis("off")

    ax_aux.annotate('',xytext=(x0,x0), xy=(x0,x1), # start, end (arrow)
                    arrowprops=dict(arrowstyle='->', linewidth=3, mutation_scale=30, color='darkgreen'))
    ax_aux.annotate('',xytext=(x0,x0), xy=(x1,x0), # start, end (arrow)
                    arrowprops=dict(arrowstyle='->', linewidth=3, mutation_scale=30, color='darkgreen'))
    
    fig.text((x1+x0)/2, x0/2, text0, ha='center', va='bottom', fontsize=12)    
    fig.text(x0/2, (x1+x0)/2, text1, ha='center', va='bottom', rotation=90, fontsize=12)  

    # plt.tight_layout()  
    
    return fig, ax


def spectrogram_grid(ts, fs, Sxxs, nrows, ncols, xlim=None, ylim=None, vmin=None, vmax=None, logscale=False, figsize=(12, 8), show=True):
    """
    Plots a grid of spectrograms.

    Args:
        ts (list): List of time arrays.
        fs (list): List of frequency arrays.
        Sxxs (list): List of power spectral densities (2D arrays).
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.
        xlim, ylim, vmin, vmax, logscale: Passed to individual spectrograms.
        figsize (tuple): Size of the full figure.
        show (bool): Whether to show the figure.
    """
    total_plots = nrows * ncols
    num_spectrograms = len(Sxxs)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.supxlabel("Time (s)")
    fig.supylabel("Frequency (Hz)")

    for i in range(total_plots):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        if i < num_spectrograms:
            t = ts[i]
            f = fs[i]
            Sxx = Sxxs[i]
            spectrogram(t, f, Sxx, xlim=xlim, ylim=ylim, vmin=vmin, vmax=vmax, logscale=logscale, ax=ax)
        else:
            ax.axis('off')  # Hide unused subplots

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes