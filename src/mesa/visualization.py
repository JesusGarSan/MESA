""""
Visualization module of the MESA package.
Contains functions to visualize signals, spectrograms and more
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import librosa
from typing import Literal, List

def plot_signal(signal: np.ndarray, sr: int, 
                x_axis: Literal["time", "s", "ms", "samples"] = "time",
                label: str = None, 
                color: str = None, 
                alpha: float = 1.0,
                offset: float = 0.0,
                ax: Axes = None):
    """
    Plots the waveform of a raw input signal.

    This function wraps librosa.display.waveshow to provide a consistent 
    interface with the stft and spectrogram functions. It supports plotting 
    into existing axes and handles temporal scaling.

    Parameters
    ----------
    signal : np.ndarray
        The input audio signal (time series).
    sr : int
        The sampling rate of the signal.
    x_axis : {"time", "s", "ms", "samples"}, optional
        The type of x-axis scale. Defaults to "time" (seconds).
    label : str, optional
        Label for the plot legend.
    color : str, optional
        Color of the waveform.
    alpha : float, optional
        Opacity of the waveform. Defaults to 1.0.
    ax : matplotlib.axes.Axes, optional
        An existing axes object to plot into. If None, a new figure and axes 
        are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """
    # Type verification and axis handling
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        if not isinstance(ax, Axes):
            raise TypeError("ax must be an instance of matplotlib.axes.Axes")
        fig = ax.get_figure()

    librosa.display.waveshow(signal, sr=sr, axis=x_axis, ax=ax, 
                             label=label, color=color, alpha=alpha, offset=offset)

    # Standardize labels and limits
    if x_axis in ["time", "s"]:
        ax.set_xlabel("Time (s)")
    elif x_axis == "ms":
        ax.set_xlabel("Time (ms)")
    elif x_axis == "samples":
        ax.set_xlabel("Samples")

    ax.set_ylabel("Amplitude")
    
    ax.set_xlim(left=0)

    if label:
        ax.legend()

    return fig

def spectrogram(Sxx: np.ndarray, sr:int, win_length:int, hop:int = None, n_fft:int = None,
                center:bool = False,
                x_axis:Literal["time", 'h','m','s','ms','lag','lag_h','lag_m','lag_s','lag_ms']="time",
                y_axis:Literal[None,"none","off", 'linear', 'fft','hz','log','fft_note','fft_svara',
                               'mel','cqt_hz','cqt_note','cqt_svara','vqt_fjs']="hz",
                colorbar:bool = True,
                cb_kwargs: dict = None,
                ax = None, **kwargs):
    """
    Visualizes a spectrogram with support for non-centered window alignment.

    This function wraps librosa.display.specshow. If center=False, it manually 
    re-aligns the x-axis ticks so that labels correspond to the start of the 
    STFT windows rather than the center, providing a more intuitive timeline 
    for causal windowing.

    Parameters
    ----------
    Sxx : np.ndarray
        The spectrogram magnitude (or power) matrix.
    sr : int
        The sampling rate of the input signal.
    win_length : int
        The window length used in the STFT calculation.
    hop : int, optional
        The hop length used in the STFT. Defaults to win_length (no overlap).
    n_fft : int, optional
        The FFT size used in the STFT. Defaults to win_length.
    center : bool, optional
        Whether the STFT was centered. If False, ticks are shifted left by 
        half a window length to align with window starts. Defaults to False.
    x_axis : str, optional
        The type of x-axis scale (e.g., "time", "s", "ms"). Defaults to "time".
    y_axis : str, optional
        The type of y-axis scale (e.g., "hz", "log", "mel"). Defaults to "hz".
    colorbar : bool, optional
        If True, adds a colorbar to the figure. Defaults to True.
    ax : matplotlib.axes.Axes, optional
        An existing axes object to plot into. If None, a new figure and axes 
        are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    mesh : matplotlib.collections.QuadMesh
        The mesh object returned by specshow.

    Notes
    -----
    When center=False, librosa still places ticks at the center of the bins. 
    This function calculates a temporal offset (win_length / 2 / sr) and 
    subtracts it from the current tick locations to ensure the labels 
    align with the left edge of each temporal bin.
    """
    if hop is None:
        hop = win_length # No overlap
    if n_fft is None:
        n_fft = win_length

    if ax is None: 
        fig, ax = plt.subplots(1,1)
    else:
        fig = ax.get_figure()

    # Make the xticks not centered
    x_coords = None
    if not center and x_axis == "time":
        frames = np.arange(Sxx.shape[1])
        x_coords = librosa.frames_to_time(frames, sr=sr, hop_length=hop, n_fft=n_fft)
        shift = librosa.samples_to_time(hop, sr=sr)
        window_length_seconds = librosa.samples_to_time(win_length, sr = sr)
        x_coords = x_coords - window_length_seconds/2 + shift/2

    mesh = librosa.display.specshow(Sxx, sr=sr, hop_length=hop, n_fft=n_fft, win_length = win_length,
                             x_axis=x_axis, y_axis=y_axis, ax = ax,
                             x_coords=x_coords, **kwargs)
    if colorbar:
            # Default to empty dict if None
            cb_kwargs = cb_kwargs or {}
            fig.colorbar(mesh, ax=ax, **cb_kwargs)
    return fig, mesh


def dual_plot(signal: np.ndarray, Sxx: np.ndarray, sr: int, 
              win_length: int, hop: int = None, n_fft: int = None,
              center: bool = False,
              x_axis:Literal["time", 'h','m','s','ms','lag','lag_h','lag_m','lag_s','lag_ms']="time",
              y_axis:Literal[None,"none","off", 'linear', 'fft','hz','log','fft_note','fft_svara',
                               'mel','cqt_hz','cqt_note','cqt_svara','vqt_fjs']="hz",
              figsize: tuple = (10, 8),
              cb_kwargs={'orientation': 'horizontal', 'pad': 0.15, 'label': 'Magnitude','aspect':80}):
    """
    Creates a stacked subplot showing the raw signal waveform above its spectrogram.
    
    The two plots share the same x-axis (time), allowing for synchronized 
    visual analysis of temporal and spectral features.

    Parameters
    ----------
    signal : np.ndarray
        The raw time-series audio signal.
    Sxx : np.ndarray
        The spectrogram magnitude/power matrix.
    sr : int
        The sampling rate of the signal.
    win_length : int
        The window length used for the STFT.
    hop : int, optional
        The hop length used for the STFT.
    n_fft : int, optional
        The FFT size used for the STFT.
    center : bool, optional
        Whether the STFT was centered. Affects tick alignment in the spectrogram.
    x_axis : str, optional
        The time unit for the x-axis. Defaults to "time".
    y_axis : str, optional
        The scale for the spectrogram y-axis. Defaults to "hz".
    figsize : tuple, optional
        The dimensions of the figure (width, height).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    if hop is None:
        hop = win_length

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=figsize, 
                                            gridspec_kw={'height_ratios': [1, 2]})

    plot_signal(signal, sr=sr, x_axis=x_axis, ax=ax_top)
    ax_top.set_title("Waveform")
    ax_top.set_xlabel("") 

    _, mesh = spectrogram(Sxx, sr=sr, win_length=win_length, hop=hop, 
                             n_fft=n_fft, center=center, x_axis=x_axis, 
                             y_axis=y_axis, ax=ax_bottom,
                             cb_kwargs=cb_kwargs)
    ax_bottom.set_title("Spectrogram")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    return fig


def multi_spectrogram(Sxx: np.ndarray, 
                      rows: List[int], 
                      cols: List[int], 
                      sr: int, 
                      win_length: int, 
                      hop:int = None,
                      n_fft:int = None,
                      **kwargs):

    row_set = set(rows)
    col_set = set(cols)
    
    if not row_set.isdisjoint(col_set):
        overlapping = row_set.intersection(col_set)
        raise ValueError(f"Indices {overlapping} cannot be in both 'rows' and 'cols'.")
    
    total_dims = set(range(Sxx.ndim))
    provided_dims = row_set.union(col_set)
    if not provided_dims.issubset(total_dims):
        invalid = provided_dims - total_dims
        raise IndexError(f"Indices {invalid} are out of bounds for array with {Sxx.ndim} dimensions.")

    row_sizes = [Sxx.shape[i] for i in rows]
    col_sizes = [Sxx.shape[i] for i in cols]
    
    num_rows = np.prod(row_sizes, dtype=int)
    num_cols = np.prod(col_sizes, dtype=int)
    
    # Pre-calculate global min/max for a shared color scale
    vmin = kwargs.pop('vmin', np.min(Sxx))
    vmax = kwargs.pop('vmax', np.max(Sxx))
    
    fig, axes = plt.subplots(num_rows, num_cols, 
                             figsize=(num_cols * 4, num_rows * 3), 
                             squeeze=False,
                             constrained_layout=True,
                             sharex=True,
                             sharey=True)
    

    meshes = [] # Store all meshes
    for r_idx, r_coords in enumerate(np.ndindex(*row_sizes)):
        for c_idx, c_coords in enumerate(np.ndindex(*col_sizes)):
            
            slicer = [slice(None)] * Sxx.ndim
            for i, dim in enumerate(rows): slicer[dim] = r_coords[i]
            for i, dim in enumerate(cols): slicer[dim] = c_coords[i]
                
            Sxx_slice = Sxx[tuple(slicer)]
            
            while Sxx_slice.ndim > 2:
                Sxx_slice = Sxx_slice[0]
            
            ax = axes[r_idx, c_idx]
            
            # Call spectrogram with forced vmin/vmax and no individual colorbars
            _, mesh = spectrogram(Sxx_slice, sr=sr, win_length=win_length, hop=hop, n_fft=n_fft,
                                    ax=ax, colorbar=False, 
                                    cb_kwargs={"vmin": vmin, "vmax":vmax},
                                    vmin = vmin, vmax = vmax, **kwargs)
            meshes.append(mesh)

            # Clean up redundant labels
            if c_idx > 0: ax.set_ylabel("")
            if r_idx < num_rows - 1: ax.set_xlabel("")
            
    # Add one global colorbar using the last mesh created
    # We pass the list of axes to ensure the colorbar doesn't only shrink the last subplot
    if meshes:
        cb = fig.colorbar(meshes[0], ax=axes.ravel().tolist(), location='right', aspect=40)
        cb.set_label('Magnitude (dB)' if 'y_axis' in kwargs else 'Power')

    return fig, axes