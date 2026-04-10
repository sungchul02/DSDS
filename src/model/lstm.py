import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, num_layers=2, num_classes=2):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        # 👉 (B, H, W) 형태로 변환
        x = x.squeeze(1)

        # 👉 (B, W, H)로 변경 (sequence 형태)
        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)

        # 마지막 시점 사용
        out = out[:, -1, :]

        out = self.fc(out)

        return out