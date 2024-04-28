import torch.nn as nn
from config import NUM_CLASSES


class LSTMModel(nn.Module):
    def __init__(self, input_size=34, hidden_size=80, num_layers=10, output_size=NUM_CLASSES, dropout_prob=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
