import numpy as np
from scipy.signal import ShortTimeFFT, windows

import tests as test
import features
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

    """ Overwritten methods """
    from scipy.signal._short_time_fft import PAD_TYPE
    from collections.abc import Generator, Callable
    def _x_slices(self, x: np.ndarray, k_off: int, p0: int, p1: int,
                padding: PAD_TYPE) -> Generator[np.ndarray, None, None]:
        """Generate signal slices along last axis of `x` (start-based)."""
        from scipy.signal._short_time_fft import PAD_TYPE, get_args
        if padding not in (padding_types := get_args(PAD_TYPE)):
            raise ValueError(f"Parameter {padding=} not in {padding_types}!")
        pad_kws: dict[str, dict] = {  # possible keywords to pass to np.pad:
            'zeros': dict(mode='constant', constant_values=(0, 0)),
            'edge': dict(mode='edge'),
            'even': dict(mode='reflect', reflect_type='even'),
            'odd': dict(mode='reflect', reflect_type='odd'),
        }

        n, n1 = x.shape[-1], (p1 - p0) * self.hop
        k0 = p0 * self.hop + k_off  # start sample
        k1 = k0 + self.m_num  # end sample

        i0, i1 = max(k0, 0), min(k1, n)  # indexes to shorten x
        pad_width = [(0, 0)] * (x.ndim - 1) + [(-min(k0, 0), max(k1 - n, 0))]

        x1 = np.pad(x[..., i0:i1], pad_width, **pad_kws[padding])
        for k_ in range(0, n1, self.hop):
            yield x1[..., k_:k_ + self.m_num]

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
        FFT bins:        {self.mfft} 
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
        self.fft, self.freqs = features.fft(self.signal, n_bins, self.sr)


    def set_stft(self, window, shift=None, windowing = "boxcar", **kwargs):
        """
        window: Window length in seconds
        windowing: window type ("boxcar", "hann", etc.)
        shift:  Window shift in seconds
        """
        if shift == None: shift = window
        overlap = window - shift

        window_samples = int(window * self.sr);
        shift_samples = int(shift*self.sr);

        win = windows.get_window(windowing, window_samples)

        ShortTimeFFT.__init__(self, win, shift_samples, self.sr, **kwargs)

        overlap_samples = overlap*self.sr
        N_windows = int((self.n_samples - overlap_samples) // (window_samples - overlap_samples))
        self.window = window;
        self.window_samples = window_samples;
        self.windowing = windowing;
        self.shift = shift;
        self.overlap = overlap;
        self.shift_samples = shift_samples;
        self.N_windows = N_windows
        self.time_stft = self.t(self.n_samples)+window/2#+shift
        self.scale = None;
        if 'scale_to' in kwargs:
            self.scale = kwargs['scale_to'];
    
    def stft(self, **kwargs):
        self.Zxx = super().stft(self.signal, **kwargs)
        return self.Zxx

    
    def spectrogram(self, **kwargs):
        self.Sxx = super().spectrogram(self.signal, **kwargs)
        return self.Sxx
    

    """ Save data """
    def save_stft(self, verbose=True, **kwargs):
        return features.save(self.Sxx.T, self.time_stft, self.f, verbose=verbose, **kwargs)

        
    """ Plotting functions """
    def plot_signal(self):
        return plot.signal(self.time, self.signal)
    def plot_fft(self, mode = 'module'):
        return plot.fft(self.freqs, self.fft, mode)
    def plot_spectrogram(self, p0=0, p1=-1, show = False):
        t_ini = self.time[0]
        t_end = self.time[-1]
        t_ini = None
        t_end = None
        # return plot.spectrogram(self.time_stft, self.f,self.Sxx, [t_ini, t_end], show)
        return plot.spectrogram(self.time_stft[p0:p1], self.f,self.Sxx, [t_ini, t_end], show)

        

# ----- Testing suite ------
if __name__ == "__main__":
   
    """ Raw signal parameters """
    sr = 100;     # Hz. Sampling rate
    duration = 2.;       # s. Duration of the signal

    """ Signal generation"""
    N = 1;      # Number of different frequencies to generate per frequency given
    f = [12.0, 13.5, 22.0, 22.15, 32.0, 32.075 ] # Hz. Main frequencies of the simulated signals
    f = [10,20,30]
    A0 = [0.16, 0.18, 0.17]
    sigma_f = 0.00 # standard deviation around the frequencies chosen for the signal generation
    sigma_A = 0.00 # standard deviation around the amplitudes chosen for the signal generation
    convolution = np.hstack((np.linspace(0.5,1, int(duration*sr//2)), np.linspace(1,.5, int(duration*sr//2))))
    generator = Generator(sr,duration,N,f,sigma_f,A0,sigma_A,convolution)
    time, signal = generator.generate_signal() 

    """ Initialize object """
    processor = Processor(signal, sr, time)
    # processor.check()

    """ Get STFT parameters """
    window = 1.0 #s
    shift = window /4#s
    windowing = "boxcar"
    processor.set_stft(window, shift, windowing, scale_to=None)
    processor.check_stft()

    """ payground """
    p0, p1 = processor.p_range(processor.signal.shape[-1], None, None)
    p0 = 0
    # p1 = 5
    # p1 = 6
    # print(p0, p1)
    # print(processor.m_num_mid)
    # slices = processor._x_slices(processor.signal, 0, p0, p1,'edge')
    # for p_, x_ in enumerate(slices):
        # print(p_, x_.shape)
        # print(x_)
    # processor.time = processor.time[p0*processor.window_samples:p1*processor.window_samples]
    # processor.signal = processor.signal[p0*processor.window_samples:p1*processor.window_samples]



    # quit()
    """ STFT calculations """
    print(processor.time_stft)
    print(processor.time_stft[p0:p1])
    print(processor.m_num)
    # Sxx = processor.spectrogram(p0=p0, p1=p1, k_offset = +processor.m_num_mid)    
    Sxx = processor.spectrogram(p0=p0, p1=p1)    

    # print(processor.time_stft[p0:p1])
    # print(Sxx.shape)


    """ Save the calculations"""
    # processor.save_stft(filepath="./data/simulated/sensor_0.mat")

    """ Plots """
    fig, ax = processor.plot_signal();
    fig.show()

    fig, ax = processor.plot_spectrogram(p0, p1)
    ax.set_title(processor.scale)
    fig.show()
    input()

