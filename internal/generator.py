import numpy as np

def generate_signal(freq:float, A:float, sr = 100.0, t = 1.0, phi:float=0, verbose = False):
    """
        freq = [2.] #Hz. frequencies of the signal
        A    = [1.] # Amplitudes of the signal's frequencies
        sr   = 100. #Hz. sampling rate
        t    = 1.0  #s. Signal duration 
        phi  = 0 # radians. Phase given to each frequency
    """
    freq = freq if hasattr(freq, '__iter__') else [freq]
    A = A if hasattr(A, '__iter__') else [A]
    phi = phi if hasattr(phi, '__iter__') else [phi]

    if verbose:
        print(f"""Maximum observable frequency (Nyquist): {sr/2}Hz""")

    time = np.linspace(0, t, int(t*sr))
    y = A*np.sin(np.outer(time, 2*np.pi*freq)+phi)
    y = np.sum(y, axis = 1)

    return time, y 

def generate_frequencies(N, sigma = 0.1, mode = 'random', f0 = None, sr:float = 100.0):
    f_Ny = sr/2; # Won't generate frequencies higher than what's observable via Nyquist
    if mode == 'random':
        if f0 is None:
            f0 = np.random.rand(N) * f_Ny # Central frequency
        freq = np.random.normal(loc = f0, scale = sigma, size = N) # loc: Mean, scale = standard deviation
        freq = np.abs(freq) # Discard negative frequencies as they don't make physical sense
    return freq

    
def generate_amplitudes(N, center=100, sigma = 1, mode = 'random'):
    # N: Number of amplitudes to generate
    # center: Value aorund which amplitudes are centered
    if mode == 'random':
        A = np.random.normal(loc = center, scale = sigma, size = N) # loc: Mean, scale = standard deviation
    return A

def generate_phase(N):
    phi = np.random.rand(N)*np.pi*2
    return phi

def convolute(signal:float, convolution:float):
    assert signal.ndim == 1, f"signal should be 1-dimensional. signal.shape={signal.shape}"
    assert convolution.ndim == 1, f"convolution should be 1-dimensional. convolution.shape={convolution.shape}"
    N = len(signal)
    assert len(convolution) == N, f"convolution must have the same number of elements as signal ({len(signal)}.)"

    return signal*convolution

def save(f, A, phi, convolution, filepath='generator_data.npz', verbose = True):
    """
    Saves the arrays f, A, phi, and convolution to a .npz file.

    Args:
      f: 1D array of frequencies.
      A: 1D array of amplitudes.
      phi: 1D array of phases.
      convolution: 1D array of convolution.
      filepath: Path to the file where data will be saved.
    """
    saved = False
    try:
        # Verify if the lengths of f, A, and phi are equal
        if not (len(f) == len(A) == len(phi)):
            raise ValueError("The arrays f, A, and phi must have the same length.")

        # Save the arrays to a .npz file
        np.savez(filepath, f=f, A=A, phi=phi, convolution=convolution)
        saved = True
    finally:
        if verbose:
            if saved: print(f"File saved successfully at {filepath}.")
            else: print(f"Failed to save file at {filepath}.")

def load(filepath='generator_data.npz', verbose=True):
    """
    Loads the arrays f, A, phi, and convolution from a .npz file.

    Args:
      filepath: Path to the .npz file.
      verbose: If True, prints messages about the loading process.

    Returns:
      Tuple containing the arrays (f, A, phi, convolution), or None if an error occurs.
    """
    loaded = False
    import os
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at {filepath}.")

        data = np.load(filepath)
        f = data['f']
        A = data['A']
        phi = data['phi']
        convolution = data['convolution']
        loaded = True
        return f, A, phi, convolution

    except FileNotFoundError as e:
        if verbose:
            print(f"Error: {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"Error loading arrays: {e}")
        return None
    finally:
        if verbose:
            if loaded:
                print(f"File loaded successfully from {filepath}.")
            elif os.path.exists(filepath):
                print(f"File exists at {filepath}, but arrays were not loaded successfully.")
            else:
                pass # file doesnt exists, error message printed in the except block.

def ground_truth(f:float, A:float, phi:float, convolution:float, sr:float, t:float,
                 n_bins:int, fft_resolution:float, n_windows:int, window_samples:int,
                 **kwargs):
    assert (len(f) == len(A) == len(phi))

    # data = np.zeros((,len(f)))

    return
    _, data = generate_signal(f,A,sr,t,phi)
    print(data.shape)
    print(convolution.shape)
    if convolution is not None:
        data = np.abs(data * convolution[:, np.newaxis])
    # data *= A

    """ Average for the windows """
    data = data.reshape(n_windows, window_samples, len(f))
    data = np.mean(data, axis = 1)

    """ Average for the frequency resolution"""
    fft_res = fft_resolution
    n_ffts = n_bins//2
    data_avg = np.zeros((n_windows,n_ffts))
    
    for j in range(n_ffts):
        f_ids = np.where((f>fft_res*j) & (f<=fft_res*(j+1)))[0]
        if f_ids.size > 0:
            aux = data[:, f_ids]
            data_avg[:,j] = aux.mean(axis=1)

    data = np.abs(data_avg)

    if 'filepath' in kwargs:
        """ Save the ground truth data matrix """
        fft_freq += fft_res # Correction for the true data labels
        fft_freq_str = [f"{round(freq, 2)} Hz" for freq in fft_freq]
        save(data, column_names=fft_freq_str, **kwargs)

    return data




class Generator():

    def __init__(self, sr:float=100.0, duration:float=10, N:int=100,
                 f0:float=np.array([10.0]), sigma_f:float=np.array([0.1]),
                 A0:float = np.array([1.0]), sigma_A:float=np.array([0.1]),
                 convolution:float=None):
        """
        sr: Sampling rate of the signal to generate (Hz)
        N: Number of frequencies, amplitudes and phases to generate
        f0: Central frequencies to generate
        sigma_f: Standard deviations of the frequencies to generate
        A0: Central amplitudes to generate
        sigma_A: Standard deviations of the amplitudes to generate
        convolution: Convolution to apply to the generated signal
        """

        # ---- Assertions and checks -----
        f0 = np.array(f0, dtype=float, ndmin=1)
        A0 = np.array(A0, dtype=float, ndmin=1)
        sigma_f = np.array(sigma_f, dtype=float, ndmin=1)
        sigma_A = np.array(sigma_A, dtype=float, ndmin=1)

        if f0.ndim > 1: f0=np.squeeze(f0)
        assert f0.ndim == 1, f"f0 should be 1-dimensional. f0.shape={f0.shape}"

        if sigma_f.ndim > 1: sigma_f=np.squeeze(sigma_f)
        assert sigma_f.ndim == 1, f"sigma_f should be 1-dimensional. sigma_f.shape={sigma_f.shape}"

        if A0.ndim > 1: A0=np.squeeze(A0)
        assert A0.ndim == 1, f"A0 should be 1-dimensional. A0.shape={A0.shape}"
        
        if sigma_A.ndim > 1: sigma_A=np.squeeze(sigma_A)
        assert sigma_A.ndim == 1, f"sigma_A should be 1-dimensional. sigma_A.shape={sigma_A.shape}"


        M = len(f0)
        if len(A0)==1: A0 = np.array([A0[0]]*M)
        assert len(A0) == M, f"A0 should have the same number of elements than f0 ({M})"

        if len(sigma_f)==1: sigma_f = np.array([sigma_f[0]]*M)
        assert len(sigma_f) == M, f"sigma_f should have the same number of elements than f0 ({M})"

        if len(sigma_A)==1: sigma_A = np.array([sigma_A[0]]*M)
        assert len(sigma_A) == M, f"sigma_A should have the same number of elements than f0 ({M})"

        if convolution is not None:
            assert convolution.ndim == 1, f"convolution should be 1-dimensional. convolution.shape={convolution.shape}"

        # ----- Main code -----
        self.N = N
        self.M = M
        self.sr = sr
        self.duration = duration
        self.f0 = f0
        self.sigma_f = sigma_f
        self.A0 = A0
        self.sigma_A = sigma_A
        self.convolution = convolution


    def generate_signal(self):
        N = self.N
        M = self.M

        self.frequencies = np.zeros(N*M)
        self.amplitudes = np.zeros(N*M)
        self.phases = np.zeros(N*M)


        for id in range(M):
            self.frequencies[id*N:(id+1)*N] = generate_frequencies(N,self.sigma_f[id],f0=self.f0[id])
            self.amplitudes[id*N:(id+1)*N] = generate_amplitudes(N,self.A0[id],self.sigma_A[id])
            self.phases[id*N:(id+1)*N] = generate_phase(N)

        sort_ids = self.frequencies.argsort()
        self.frequencies = self.frequencies[sort_ids]
        self.amplitudes = self.amplitudes[sort_ids]
        self.phases = self.phases[sort_ids]

        self.time, self.signal = generate_signal(self.frequencies,self.amplitudes,self.sr,
                                      self.duration,self.phases)
        
        if self.convolution is not None:
            self.signal = convolute(self.signal, self.convolution)
        
        return self.time, self.signal


    def check(self):
        print(f"""
        sampling rate:      {self.sr} Hz
        duration:           {self.duration} s
        Main frequencies:   {self.f0} Hz
        Main amplitudes:    {self.A0}
              """)
        

    def save_data(self, **kwargs):
        return save(self.frequencies, self.amplitudes, self.phases, self.convolution, **kwargs)


    def save_object(self, filepath = "generator.pkl", verbose = True):
        """
        Saves the instance of the class using pickle.

        Args:
            filepath: Path to the file where the object will be saved.
        """
        import pickle
        saved = False
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            saved = True
        finally:
            if verbose:
                if saved: print(f"Object saved successfully at {filepath}.")
                else: print(f"Failed to save object at {filepath}.")

        return saved
    

    def ground_truth(self, n_bins:int, fft_resolution:float, n_windows:int, window_samples:int, **kwargs):
        return ground_truth(self.frequencies, self.amplitudes, self.phases, self.convolution, self.sr, self.duration,
                            n_bins,fft_resolution,n_windows,window_samples, **kwargs)

    def plot(self):
        import matplotlib.pyplot as plt
        width = np.min(np.diff(self.frequencies))
        width = width*self.N*self.M
        plt.bar(self.frequencies, self.amplitudes, width)
        plt.xlabel("Frequencies (Hz)")
        plt.ylabel("Amplitudes")
        plt.ylim([0, plt.ylim()[1]])
        plt.show()
        
        
        

if __name__ == '__main__':

    generator = Generator(sr=100.0,duration=10,N=100,
                          f0=[10,20],sigma_f=[.5,.5],
                          A0=[1,2],sigma_A=[1.,1.],
                          convolution=np.ones(10*100))
    generator.generate_signal()
    generator.check()
    generator.save_data(filepath="./data/generator/generator_data.npz")
    generator.save_object(filepath="./data/generator/generator.pkl")
    # generator.plot()

    # Ground truth
    generator.ground_truth(100, 1.0,5,100*1)