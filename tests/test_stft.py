from mesa.features import stft, stft_labels
import numpy as np



def test_stft_minimal_call():
    y = np.ones(100)
    D = stft(y, 10)
    return

def test_stft_dimension():
    y = np.ones(100)
    D = stft(y, 10)
    assert D.shape == (6, 10) # n_freqs=10/2+1, n_times = 100/10

def test_stft_multi_signal_dimensions():
    y = np.ones((10, 20, 100)) # Last axis is interpreted as time
    D = stft(y, 10)
    assert D.shape == (10, 20, 6, 10) 

def test_stft_freq_decimation_dimensions():
    y = np.ones(500)
    D = stft(y, 50, n_fft=6)
    assert D.shape == (4, 10) # n_freqs=n_fft/2+1, n_times = 100/10
    D = stft(y, 50, n_fft=5, freq_decimation_method="average")
    assert D.shape == (3, 10) # n_freqs=n_fft/2+1, n_times = 100/10

def test_stft_labels():
    y = np.ones(10)
    D = stft(y,2,2,)
    times, freqs = stft_labels(D, 10, 2,2)
    assert (times == np.array([0,0.2,0.4,0.6,0.8])).all()
    assert (freqs == np.array([0.0,5.0])).all()

def test_stft_slice_demeaning():
    sr = 100
    y = np.ones(sr * 2) * 10.0 
    win_length = 10
    hop = 10

    D_demeaned = stft(y, win_length=win_length, hop=hop, demean_slices=True)
    D_normal   = stft(y, win_length=win_length, hop=hop, demean_slices=False)

    zero_hz_demeaned = np.abs(D_demeaned[0, :])
    zero_hz_normal = np.abs(D_normal[0, :])

    assert np.all(zero_hz_normal > 1.0), "Sanity check failed: Normal STFT should capture the DC offset."

    np.testing.assert_allclose(
        zero_hz_demeaned, 
        0.0, 
        atol=1e-7, 
        err_msg="Slice demeaning failed! The 0 Hz (DC) bin is not zero."
    )

