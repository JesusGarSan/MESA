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

    x = np.linspace(0, t, int(t*sr))
    y = A*np.sin(np.outer(x, 2*np.pi*freq)+phi)
    y = np.sum(y, axis = 1)

    return x, y 

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


""" Generator class """
class Generator():

    def __init__(self, sr:float=100.0, duration:float=10, N:int=100,
                 f0:float=np.array([10.0]), sigma_f:float=np.array([0.1]),
                 A0:float = np.array([1.0]), sigma_A:float=np.array([0.1])):
        """
        sr: Sampling rate of the signal to generate (Hz)
        N: Number of frequencies, amplitudes and phases to generate
        f0: Central frequencies to generate
        sigma_f: Standard deviations of the frequencies to generate
        A0: Central amplitudes to generate
        sigma_A: Standard deviations of the amplitudes to generate
        """

        # ---- Assertions and checks -----
        f0 = np.array(f0, dtype=float)
        A0 = np.array(A0, dtype=float)
        sigma_f = np.array(sigma_f, dtype=float)
        sigma_A = np.array(sigma_A, dtype=float)

        if f0.ndim > 1: f0=np.squeeze(f0)
        assert f0.ndim == 1, f"f0 should be 1-dimensional. f0.shape={f0.shape}"

        if sigma_f.ndim > 1: sigma_f=np.squeeze(sigma_f)
        assert sigma_f.ndim == 1, f"sigma_f should be 1-dimensional. sigma_f.shape={sigma_f.shape}"

        if A0.ndim > 1: A0=np.squeeze(A0)
        assert A0.ndim == 1, f"A0 should be 1-dimensional. A0.shape={A0.shape}"
        
        if sigma_A.ndim > 1: sigma_A=np.squeeze(sigma_A)
        assert sigma_A.ndim == 1, f"sigma_A should be 1-dimensional. sigma_A.shape={sigma_A.shape}"


        M = len(f0)
        assert len(A0) == M, f"A0 should have the same number of elements than f0 ({M})"
        assert len(sigma_f) == M, f"sigma_f should have the same number of elements than f0 ({M})"
        assert len(sigma_A) == M, f"sigma_A should have the same number of elements than f0 ({M})"


        # ----- Main code -----
        self.N = N
        self.M = M
        self.sr = sr
        self.duration = duration
        self.f0 = f0
        self.sigma_f = sigma_f
        self.A0 = A0
        self.sigma_A = sigma_A


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

        self.signal = generate_signal(self.frequencies,self.amplitudes,self.sr,
                                      self.duration,self.phases)
        
        return self.signal


    def check(self):
        print(f""""
        sampling rate:      {self.sr} Hz
        duration:           {self.duration} s
        Main frequencies:   {self.f0} Hz
        Main amplitudes:    {self.A0}
              """)
        
    def plot(self):
        import matplotlib.pyplot as plt
        width = np.min(np.diff(self.frequencies))
        width = width*self.N
        plt.bar(self.frequencies, self.amplitudes, width)
        plt.xlabel("Frequencies (Hz)")
        plt.ylabel("Amplitudes")
        plt.ylim([0, plt.ylim()[1]])
        plt.show()
        
        
        




if __name__ == '__main__':

    generator = Generator(sr=100.0,duration=10,N=100,
                          f0=[10,20],sigma_f=[.5,.5],
                          A0=[1,2],sigma_A=[1.,1.])
    generator.generate_signal()
    generator.check()
    generator.plot()