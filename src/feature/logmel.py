import numpy as np
import librosa


def extract_logmel(
    y,
    sr,
    n_mels=128,
    n_fft=2048,
    hop_length=512
):
    """
    Log-Mel Spectrogram 추출
    Returns: (1, H, W)
    """

    # 🔥 핵심: 짧은 오디오 대응
    if len(y) < n_fft:
        n_fft = len(y)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    # 표준화
    mean = np.mean(logmel)
    std = np.std(logmel) + 1e-6
    logmel = (logmel - mean) / std

    logmel = np.expand_dims(logmel, axis=0)

    return logmel