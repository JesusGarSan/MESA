from functions.simule_data import *
from functions.feature_extraction import *
from Processor import Processor

""" Set the parameters """
sr = 100;       # Hz. Sampling rate
t = 15;         # s. Duration of the signal
N = 10000;      # Number of different frequencies composing the signal

window = 1   # s.
shift  = 1   # s.  
n_bins = 200;  # Number of bins to use for the DFT calculation

n_sensors = 5; # Number of sensors in the simulation
sensors = []

plot = True
if plot: 
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(n_sensors, 1, sharex=True)
    ymax = 0; ymin = 0;

for i in range(n_sensors):
    """ Generate the signal """
    y = np.zeros(sr*t)
    # Main frequencies of the signal
    f = [2.0, 5.5] #Hz
    for f0 in f:
        freq = generate_frequencies(N, sr=sr, f0=f0);
        center = 10 + i**2 # Cada sensor es más sensible que el anterior
        A = generate_amplitudes(N, center=center, sigma= 1);
        x, y_aux = generate_signal(freq, A, t=t, sr=sr);
        y+= y_aux

    x_aux = x - 5
    # y *= (x_aux) *1*np.exp(-x_aux) # envolvente de la señal (evento detectado)
    # y *= np.sin(x/t*4)
    # y *= np.exp(x_aux)
    y *= 1* np.exp(-(x_aux/3)**2)

    if plot:
        ymin = min([ymin, min(y)])
        ymax = max([ymax, max(y)])
        ax[i].plot(x, y)


    """ Create signal object """
    sensors.append(Processor(y, sr))
    """ Apply windowing and calculate FFT"""
    sensors[i].set_windows(window, shift);
    sensors[i].fft_bin_window(n_bins)
    """ Create data matrix """
    ffts = sensors[i].fft_windows[:, 0:n_bins//2]
    fft_freq = sensors[i].freqs_windows[0:n_bins//2]
    fft_freq_str = [f"{round(freq, 2)} Hz" for freq in fft_freq]
    filepath = f"./data/sensor_{i+1}.mat"
    save(np.abs(ffts), column_names=fft_freq_str, filepath=filepath)
    print(f"Sensor {i+1} data saved at {filepath}")


if plot:
    for i in range(n_sensors):
        ax[i].set_ylim([ymin, ymax])
    plt.savefig("./data/raw_signals.png")
    plt.show(block=False)
