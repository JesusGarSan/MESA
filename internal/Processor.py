import numpy as np
from scipy.signal import ShortTimeFFT, windows

import tests as test
from features import *
import plot
from generator import Generator

class Processor(ShortTimeFFT):
    def __init__(self, signal, sample_rate, time=None):
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

        if time is None:
            self.time = np.linspace(0, self.duration, self.n_samples)
        else: self.time = time

    def check(self):
        print(f"""
        signal duration: {self.duration} s
        sample rate:     {self.sr} Hz
        Nº samples:      {self.n_samples}
        """)
        
    def check_fft(self):
        print(f"""
        signal duration: {self.duration} s
        sample rate:     {self.sr} Hz
        Nº samples:      {self.n_samples}
        FFT bins:        {self.n_bins} 
        FFT resolution:  {self.fft_resolution} Hz
        window length:   {self.window} s
        window overlap:  {self.overlap} s 
        window shift:    {self.shift} s
        Nº of windows:   {self.N_windows}
        """)
        
    def check_stft(self):
        print(f"""
        signal duration: {self.duration} s
        sample rate:     {self.sr} Hz
        Nº samples:      {self.n_samples}
        FFT bins:        {self.f_pts} 
        FFT resolution:  {self.delta_f} Hz
        window length:   {self.window} s
        window overlap:  {self.overlap} s 
        window shift:    {self.shift} s
        Nº of windows:   {self.N_windows}
        """)

    def Parseval(self, verbose=True):
        test.Parseval(self.signal, self.fft, verbose)

    def bins_check(self):
        test.bins_check(self.sr, self.duration, self.n_bins)

    def bins_check_window(self):
        test.bins_check(self.sr, self.window, self.n_bins)

    """ Feature extraction functions """
    def fft(self, n_bins):
        self.n_bins = n_bins;
        self.fft_resolution = self.sr/n_bins;
        self.window = self.duration;
        self.shift = None
        self.overlap = None
        self.fft, self.freqs = fft(self.signal, n_bins, self.sr)


    def set_stft(self, window, shift=None, windowing = "boxcar", **kwargs):
        """
        window: Window length in seconds
        windowing: window type ("boxcar", "hann", etc.)
        shift:  Window shift in seconds
        """
        if shift == None: shift = window
        overlap = window - shift

        window_samples = int(window * self.sr);
        shift_samples = shift*self.sr;

        win = windows.get_window(windowing, window_samples)

        ShortTimeFFT.__init__(self, win, shift_samples, self.sr, **kwargs)

        overlap_samples = overlap*self.sr
        N_windows = (self.n_samples - overlap_samples) // (window_samples - overlap_samples)
        self.window = window;
        self.window_samples = window_samples;
        self.windowing = windowing;
        self.shift = shift;
        self.overlap = overlap;
        self.shift_samples = shift_samples;
        self.N_windows = N_windows
        self.time_stft = self.t(self.n_samples)
        self.scale = None;
        if 'scale_to' in kwargs:
            self.scale = kwargs['scale_to'];
    
    def stft(self, **kwargs):
        self.Zxx = super().stft(self.signal, **kwargs)
        return self.Zxx

    
    def spectrogram(self, **kwargs):
        self.Sxx = super().spectrogram(self.signal, **kwargs)
        return self.Sxx
    
    def save_stft(self, verbose=True, **kwargs):
        return save(self.Sxx.T, self.time_stft, self.f, verbose=verbose, **kwargs)

        
    """ Plotting functions """
    def plot_signal(self):
        return plot.signal(self.time, self.signal)
    def plot_fft(self, mode = 'module'):
        return plot.fft(self.freqs, self.fft, mode)
    def plot_spectrogram(self, show = False):
        return plot.spectrogram(self.time_stft, self.f,self.Sxx, show)

        


if __name__ == "__main__":
   
    """ Raw signal parameters """
    sr = 100;     # Hz. Sampling rate
    duration = 5;       # s. Duration of the signal

    """ Signal generation"""
    N = 100;      # Number of different frequencies to generate per frequency given
    f = [12.0, 13.5, 22.0, 22.15, 32.0, 32.075 ] # Hz. Main frequencies of the simulated signals
    sigma_f = 0.10 # standard deviation around the frequencies chosen for the signal generation
    sigma_A = 0.50 # standard deviation around the amplitudes chosen for the signal generation
    generator = Generator(sr,duration,N,f)
    time, signal = generator.generate_signal() 

    """ Initialize object """
    processor = Processor(signal, sr, time)
    processor.check()

    """ Get STFT parameters """
    window = 1 #s
    shift = window #s
    windowing = "boxcar"
    processor.set_stft(window, shift, windowing, scale_to='psd')
    processor.check_stft()

    """ STFT calculations """
    Zxx = processor.stft()
    Sxx = processor.spectrogram()
   
    """ Save the calculations"""
    processor.save_stft(filepath="./data/simulated/sensor_0.mat")

    """ Plots """
    fig, ax = processor.plot_signal();
    fig.show()


    fig, ax = processor.plot_spectrogram()
    ax.set_title(processor.scale)
    fig.show()
    input()

