import numpy as np


class Signal:
    """
    Represents a 1-dimensional signal with a sampling rate, waveform, and time axis.
    """
    def __init__(self, sr: float, y: np.ndarray, x: np.ndarray = None):
        """
        Initializes the Signal object.

        Args:
            sr (float): The sampling rate of the signal in Hz.
            y (np.ndarray): A 1-dimensional NumPy array representing the waveform (signal values).
            x (np.ndarray, optional): An optional 1-dimensional NumPy array representing the time axis.
                                               If None, a time axis based on the number of samples is created.
                                               Defaults to None.

        Raises:
            TypeError: If 'y' is not a NumPy array or if 'x' is provided and is not a NumPy array.
            ValueError: If 'y' is not a 1-dimensional array, or if 'x' is provided and is not a 1-dimensional array,
                        or if the shapes of 'x' and 'y' do not match.
        """
        if not isinstance(y, np.ndarray):
            raise TypeError("Argument 'y' must be a NumPy array.")
        if y.ndim != 1:
            raise ValueError("Argument 'y' must be a 1-dimensional array.")

        if x is not None:
            if not isinstance(x, np.ndarray):
                raise TypeError("Argument 'x' must be a NumPy array.")
            if x.ndim != 1:
                raise ValueError("Argument 'x' must be a 1-dimensional array.")
            if x.shape != y.shape:
                raise ValueError("The shapes of 'x' and 'y' must be the same.")
            self.x = x
        else:
            self.x = np.arange(stop=len(y))

        self.sr = sr  # Sample rate
        self.y = y    # Wave form
        self.n_samples = len(y) # Number of samples of the signal
        self.T = self.n_samples / self.sr # Duration of the signal

    def check(self):
        """
        Prints basic information about the signal.
        """
        print(f"""
        Signal data:
        sample rate:       {self.sr} Hz
        signal duration: {self.T} s
        Nº samples:        {self.n_samples}
        """)

if __name__ == '__main__':
    from simulation import generator

    sr = 100.0
    N = 10
    t = 10

    freq = generator.generate_frequencies(N,)
    A = generator.generate_amplitudes(N,)
    phi = generator.generate_phase(N, )
    x, y = generator.generate_signal(freq, A, sr, t, phi)

    # import obspy
    # st = obspy.read()
    # sr = st[0].stats.sampling_rate
    # y = st[0].data
    # x = None

    signal = Signal(sr, y, x)
    signal.check()

