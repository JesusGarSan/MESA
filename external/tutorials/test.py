import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window, ShortTimeFFT
T = 4.0#s
sr = 100

n_samples = int(T*sr)
t = np.linspace(0, T, n_samples)

signal = np.sin(t)

window_size = 1
win_samples = int(window_size*sr)

print(n_samples, win_samples)

win = get_window("boxcar", win_samples)
SFT = ShortTimeFFT(win, win_samples,sr, mfft=win_samples, phase_shift=win_samples//2)
k, p = SFT.upper_border_begin(n=n_samples)
print(k,p)

