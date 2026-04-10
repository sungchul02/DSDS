from src.model.cnn import CNNModel
from src.model.lstm import LSTMModel


def get_model(name):
    name = name.lower()

    if name == "cnn":
        return CNNModel()

    elif name == "lstm":
        return LSTMModel()

    else:
        raise ValueError(f"Unknown model: {name}")