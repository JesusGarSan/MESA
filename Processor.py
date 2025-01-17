import numpy as np
from feature_extraction import *
import plot

class Processor:
    def __init__(self, signal, sample_rate, x=None):
        self.signal = signal
        self.sr = sample_rate

        self.duration = len(signal)/sample_rate
        self.n_samples = len(signal)
        self.n_bins = None

        if x == None:
            self.x = np.linspace(0, self.duration, self.n_samples)
        else: self.x = x

    def check(self):
        print(f"""
        Nº samples:     {self.n_samples}
        sample rate:    {self.sr} Hz
        signal duration:{self.duration} s
        FFT bins:       {self.n_bins} 
        """)

    """ Feature extraction functions """
    def fft(self, n_bins):
        self.n_bins = n_bins;
        self.fft, self.freqs = fft_bin(self.signal, n_bins, self.sr)


    """ Plotting functions """
    def plot_signal(self):
        return plot.signal(self.x, self.signal)
    def plot_fft(self, mode = 'module'):
        return plot.fft(self.freqs, self.fft, mode)