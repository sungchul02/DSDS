from src.feature.logmel import extract_logmel
from src.feature.mfcc import extract_mfcc
from src.feature.chroma import extract_chroma   # 이거 있어야 함


def get_feature(name):
    name = name.lower()

    if name == "logmel":
        return extract_logmel
    elif name == "mfcc":
        return extract_mfcc
    elif name == "chroma":
        return extract_chroma
    else:
        raise ValueError(f"Unknown feature: {name}")