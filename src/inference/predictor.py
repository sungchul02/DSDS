import torch
import librosa
import numpy as np
from src.feature.feature_factory import get_feature


def predict(model, wav_path, feature_name):
    y, sr = librosa.load(wav_path, sr=22050)

    target_len = 22050 * 2
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    feature_fn = get_feature(feature_name)
    feat = feature_fn(y, sr)

    feat = feat[:, :128, :128]

    x = torch.tensor(feat).unsqueeze(0).float()

    model.eval()
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()

    return "drone" if pred == 1 else "no drone"