import numpy as np
from src import *

class Processor:
    def __init__(self, signal, sample_rate, x=None):
        self.signal = signal
        self.sr = sample_rate

        self.duration = len(signal)/sample_rate
        self.n_samples = len(signal)
        self.n_bins = None
        self.fft_resolution = None

        self.window = None
        self.shift = None
        self.overlap = None
        self.N_windows = None

        if x == None:
            self.x = np.linspace(0, self.duration, self.n_samples)
        else: self.x = x

    def check(self):
        print(f"""
        signal duration: {self.duration} s
        sample rate:     {self.sr} Hz
        Nº samples:      {self.n_samples}
        """)
        
    def check_fft(self):
        print(f"""
        signal duration:{self.duration} s
        sample rate:    {self.sr} Hz
        Nº samples:     {self.n_samples}
        FFT bins:       {self.n_bins} 
        FFT resolution: {self.fft_resolution} Hz
        window length:  {self.window} s
        window overlap: {self.overlap} s 
        window shift:   {self.shift} s
        Nº of windows:  {self.N_windows}
        """)

    def Parseval(self, verbose=True):
        test.Parseval(self.signal, self.fft, verbose)

    def bins_check(self):
        test.bins_check(self.sr, self.duration, self.n_bins)

    def bins_check_window(self):
        test.bins_check(self.sr, self.window, self.n_bins)

    """ Feature extraction functions """
    def fft_bin(self, n_bins):
        self.n_bins = n_bins;
        self.fft_resolution = self.sr/n_bins;
        self.window = self.duration;
        self.shift = None
        self.overlap = None
        self.fft, self.freqs = fft_bin(self.signal, n_bins, self.sr)

    def set_windows(self, window, shift = None):
        """
        window: Window length in seconds
        shift:  Window shift in seconds
        """
        if shift == None: shift = window
        overlap = window - shift

        window_samples = int(window * self.sr)
        shift_samples = shift*self.sr
        overlap_samples = overlap * self.sr

        N_windows = (self.n_samples - overlap_samples) // (window_samples - overlap_samples)

        self.window_start_ids = np.arange(0, self.n_samples - window_samples + 1, shift_samples,
                                          dtype=int);

        self.window = window;
        self.shift = shift;
        self.overlap = overlap;
        self.N_windows = int(N_windows);
        self.window_samples = window_samples;
        self.overlap_samples = overlap_samples;
        self.shift_samples = shift_samples;
    
    def fft_bin_window(self, n_bins):
        self.n_bins = n_bins;
        self.fft_resolution = self.sr/n_bins;
        fft_windows = np.zeros((self.N_windows, n_bins), dtype=np.complex_)
        
        for i in range(self.N_windows):
            start_id = self.window_start_ids[i]
            end_id = start_id + self.window_samples
            signal = self.signal[start_id:end_id]
            fft, freqs =  fft_bin(signal, n_bins, self.sr)
            fft_windows[i] = fft

        self.fft_windows = fft_windows;
        self.freqs_windows = freqs;
        
    """ Plotting functions """
    def plot_signal(self):
        return plot.signal(self.x, self.signal)
    def plot_fft(self, mode = 'module'):
        return plot.fft(self.freqs, self.fft, mode)