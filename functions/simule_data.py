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
    y = 0

    for f, a, p in zip(freq, A, phi):
        y += (a*np.sin(x * 2*np.pi*f 
            + p) # Apply phase
            # + ((np.random.rand()-0.5)*a)) # Apply random mean
            )   

    return x, y 

def generate_frequencies(N, sigma = 0.1, mode = 'random', f0 = None, sr:float = 100.0):
    f_Ny = sr/2; # Won't generate frequencies higher than what's observable via Nyquist
    if mode == 'random':
        if f0 is None:
            f0 = np.random.rand(N) * f_Ny # Central frequency
        freq = np.random.normal(loc = f0, scale = sigma, size = N) # loc: Mean, scale = standard deviation
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