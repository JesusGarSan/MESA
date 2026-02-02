import src.msa.feature_extraction.features as features
import numpy as np

def test_build_STFT():
    SFT = features.STFT(100.0, 100, 33, "hann", None)
    return

def test_parallelization():
    SFT = features.STFT(100.0, 100, 33, "hann", None)
    signal_tensor = np.ones([3, 5, 4, 100])
    _, _, Zxx = features.stft_parallel(signal_tensor, SFT, None, detr="linear")

    assert signal_tensor.shape[:-1] == Zxx.shape[:-2]



