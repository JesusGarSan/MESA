from src.msa import generate
from src.msa.visualization import plot
from src.msa.feature_extraction import features

from mspc_pca.pca import *
from mspc_pca.omeda import *
from mspc_pca.mspc import *
from mspc_pca.ckf import *
import mspc_pca.plot as pca_plot

import numpy as np

T = 60 #s. Duration in secods
sr = 100 # Sampling rate
n_samples = int(T*sr)

time = np.linspace(0,T, n_samples)

# Generate Chirps
chirp   = generate.chirp(time, A=1, f0=0, f1=50, t0=1,t1=60,method='linear',phi=0)*generate.decay(time, 0.0, 1)
chirp  += generate.chirp(time, A=1, f0=50, f1=0, t0=1,t1=60,method='linear',phi=0)*generate.decay(time, 0.0, 1)
chirp  += generate.chirp(time, A=1.5, f0=10, f1=15, t0=25,t1=30,method='linear',phi=0)*generate.decay(time, 0.1, 25)
chirp  += generate.chirp(time, A=2, f0=5, f1=5.5, t0=40,t1=41,method='linear',phi=0)*generate.decay(time, 0.2, 40)
chirp  += generate.chirp(time, A=3, f0=40, f1=10, t0=15,t1=30,method='linear',phi=0)*generate.decay(time, 0.5, 15)

# Generate white noise
A = 2
white = np.random.rand(int(T*sr))*A*2 - A

# Synthetic signal
signal = chirp + white 

# Parameters
sr = sr #Hz. Sampling rate is often fixed by our sensors. Via Nyquist, it determines the maximum frequency we can observe.
window_sizes = [1, 3, 6] #s. Longer windows provide better frequency resolution and worse temporal resolution.
shift_fractions = [1, 0.5, 0.1]#s. Shorter shifts provide better temporal resolution.
n_bins = None # Number of different frequencies to consider for the FFT. Its value should be equal to the number of samples per window. If lower, the frequency resolution will deminish.

times, freqs, Sxxs = [], [], []
for window in window_sizes:
    for shift_fraction in shift_fractions:
            win_samples = int(sr*window)
            hop = int(win_samples*shift_fraction)
            time, freq, Sxx = features.spectrogram(signal, sr, win_samples, hop, "boxcar","odd", "linear")
            times.append(time)
            freqs.append(freq)
            Sxxs.append(Sxx)

plot.grid(len(window_sizes), len(shift_fractions), text0=fr"Window size", text1=rf"Window shift")
plt.show()