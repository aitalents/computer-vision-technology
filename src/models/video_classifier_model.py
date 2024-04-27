import torch.nn as nn


class VideoClassifier(nn.Module):
    def __init__(self, num_classes, in_channels: int = 3):
        super(VideoClassifier, self).__init__()

        self.convolutions = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            # ... #
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            # ... #
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(-1, 256 * 2 * 2 * 2)
        x = self.fc(x)
        return x
