import numpy as np
import scipy
import matplotlib.pyplot as plt


def simule_data(f0 = 2, n_f = 10, df = 0.1, sr = 100, t = 1, verbose = False):
    """
        f0  = 1.   #Hz. frequency of the signal
        n_f = 10   # Number of Complementary frequencies
        df  = 0.1  #Hz. Complementary frequencies step
        sr  = 100. #Hz. sampling rate
        t   = 15.  #s. Signal duration 
    """

    if verbose:
        print(f"""Maximum observable frequency (Nyquist): {sr/2}Hz""")

    x = np.linspace(0, t, int(t*sr))
    y = 0
    for f in np.arange(1, n_f)*df:
        y += np.sin(x * 2*np.pi*(f0 + f))

    return x, y 


if __name__ == "__main__":
    print("Running main ...")

    x, y = simule_data(f0 = 1,
        verbose = True)

    fig, ax = plt.subplots()
    plt.plot(x,y)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()


