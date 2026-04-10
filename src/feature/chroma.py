import librosa
import numpy as np


def extract_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # 정규화
    chroma = (chroma - np.mean(chroma)) / (np.std(chroma) + 1e-6)

    # (1, H, W) 형태로 변환
    chroma = np.expand_dims(chroma, axis=0)

    return chroma