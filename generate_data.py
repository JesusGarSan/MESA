from functions.simule_data import *
from functions.feature_extraction import *
from functions.tests import *
from Processor import Processor

""" Set the parameters """
sr = 100;       # Hz. Sampling rate
t = 300;         # s. Duration of the signal
N = 1000;      # Number of different frequencies composing the signal

window = 1.0     # s. Length of the windows in seconds
shift  = window  # s. Length of the window shift in seconds
n_bins = int(window*sr/2);  # Number of bins to use for the DFT calculation

f = [1.70, 11.111] # Hz. Main frequencies of the simulated signals
sigma_f = .5 # standard deviation around the frequencies chosen for the signal generation
sigma_A = 1. # standard deviation around the amplitudes chosen for the signal generation

n_sensors = 5; # Number of sensors in the simulation
sensors = []

print(f"FFT resolution: {sr/n_bins}Hz")
print(f"True FFT resolution: {sr/(window*sr)}Hz")
bins_check(sr, window, n_bins)

plot = True
if plot: 
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(n_sensors, 1, sharex=True)
    ymax = 0; ymin = 0;

for i in range(n_sensors):

    """ Generate the signal """
    y = np.zeros(sr*t)
    for f0 in f:
        freq = generate_frequencies(N, sr=sr, f0=f0,sigma=sigma_f);
        center = 10 + i**2 # Cada sensor es más sensible que el anterior
        A = generate_amplitudes(N, center=center, sigma= sigma_A);
        x, y_aux = generate_signal(freq, A, t=t, sr=sr);
        y+= y_aux

    """ Signal convolution """
    x_aux = x - t/2 # Peak on the middle of the signal
    y *= 1* np.exp(-(x_aux/3)**2)
    
    # Plot the signal of each sensor
    if plot:
        ymin = min([ymin, min(y)])
        ymax = max([ymax, max(y)])
        ax[i].plot(x, y)


    """ Create signal object """
    sensors.append(Processor(y, sr))

    """ Apply windowing and calculate FFT"""
    sensors[i].set_windows(window, shift);
    sensors[i].fft_bin_window(n_bins)
    sensors[i].check_fft()

    """ Create data matrix """
    ffts = sensors[i].fft_windows[:, 0:n_bins//2]
    fft_freq = sensors[i].freqs_windows[0:n_bins//2]
    fft_freq_str = [f"{round(freq, 2)} Hz" for freq in fft_freq]

    """ Save the data matrix """
    filepath = f"./data/sensor_{i+1}.mat"
    save(np.abs(ffts), column_names=fft_freq_str, filepath=filepath)
    print(f"Sensor {i+1} data saved at {filepath}")




if plot:
    for i in range(n_sensors):
        # Set the same Y-axis for all sensors
        ax[i].set_ylim([ymin, ymax])
    # Save the plots
    plt.savefig("./data/raw_signals.png")
    plt.show(block=False)
