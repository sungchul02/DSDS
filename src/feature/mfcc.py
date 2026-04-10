import numpy as np
import librosa


def extract_mfcc(
    y,
    sr,
    n_mfcc=40,
    n_fft=2048,
    hop_length=512
):
    """
    MFCC feature 추출

    Returns:
        np.ndarray shape: (1, H, W)
    """

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # (H, W) → (1, H, W)
    mfcc = np.expand_dims(mfcc, axis=0)

    return mfcc