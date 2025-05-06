import numpy as np

def generate_signal(freq:float, A:float, sr = 100.0, t = 1.0, phi:float=0, verbose = False):
    """
        freq = [2.] #Hz. frequencies of the signal
        A    = [1.] # Amplitudes of the signal's frequencies
        sr   = 100. #Hz. sampling rate
        t    = 1.0  #s. Signal duration 
        phi  = 0 # radians. Phase given to each frequency
    """

    m = 1
    try: m = max(m, len(freq))
    except: pass
    try: m = max(m, len(A))
    except: pass
    try: m = max(m, len(phi))
    except: pass


    freq = freq if hasattr(freq, '__iter__') else [freq]*m
    A = A if hasattr(A, '__iter__') else [A]*m
    phi = phi if hasattr(phi, '__iter__') else [phi]*m

    assert len(A) == len(freq) == len(phi)

    if verbose:
        print(f"""Maximum observable frequency (Nyquist): {sr/2}Hz""")

    x = np.linspace(0, t, int(t*sr))
    y = A*np.sin(np.outer(x, 2*np.pi*freq)+phi)
    y = np.sum(y, axis = 1)

    return x, y 

def generate_frequencies(N, sigma = 0.1, f0 = None, mode = 'random', sr:float = 100.0):
    if f0 is None:
        f_Ny = sr/2; # Won't generate frequencies higher than what's observable via Nyquist
        f0 = [np.random.rand(N) * f_Ny] # Central frequency
        
    f0 = f0 if hasattr(f0, '__iter__') else [f0]

    if mode == 'random':    

        frequencies = np.zeros(N*len(f0))
        for i, f  in enumerate(f0):
            freq = np.random.normal(loc = f, scale = sigma, size = N) # loc: Mean, scale = standard deviation
            frequencies[N*i:N*(i+1)] = np.abs(freq) # Discard negative frequencies as they don't make physical sense

    return frequencies

    
def generate_amplitudes(N, center=100, sigma = 1, mode = 'random'):
    # N: Number of amplitudes to generate
    # center: Value aorund which amplitudes are centered
    center = center if hasattr(center, '__iter__') else [center]

    if mode == 'random':
        A = np.zeros(N*len(center))
        for i, A0  in enumerate(center):
            A[N*i:N*(i+1)] = np.random.normal(loc = A0, scale = sigma, size = N) # loc: Mean, scale = standard deviation
    return A

def generate_phase(N):
    phi = np.random.rand(N)*np.pi*2
    return phi


def ground_truth(F, A, phi, sr, convolution = None):
    """ WIP """
    x, y = generate_signal(F, A, t=t, sr=sr, phi=phi);

    wave = np.sin(np.outer(x, np.pi*2*F)+phi)
    if convolution is None: convolution = np.ones(len(wave))
    data = np.abs(wave * convolution[:, np.newaxis]) # We take the absolute value of the final signal
    data *= A
    return data