import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branch = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                padding=(0, kernel_size // 2, kernel_size // 2),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size, 1, 1),
                stride=(stride, 1, 1),
                padding=(kernel_size // 2, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, stride, stride),
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        out = self.branch(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class VideoClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoClassifier, self).__init__()
        self.convolutions = nn.Sequential(
            ResidualBlock(3, 64, stride=2),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.fc = nn.Sequential(nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
