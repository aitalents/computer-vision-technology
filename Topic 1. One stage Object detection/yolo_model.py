import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, num_classes=80):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.conv_layers = self._create_conv_layers()
        self.fc_layers = self._create_fc_layers()

    def _create_conv_layers(self):
        return nn.Sequential(
            self._create_conv_block(3, 192, kernel_size=7, stride=2, padding=1),
            nn.MaxPool2d(2, stride=2),
            self._create_conv_block(192, 256),
            nn.MaxPool2d(2, stride=2),
            self._create_conv_block(256, 512),
            nn.MaxPool2d(2, stride=2),
            self._create_conv_block(512, 1024),
            nn.MaxPool2d(2, stride=2),
            self._create_conv_block(1024, 1024, stride=2, padding=1),
            self._create_conv_block(1024, 1024),
        )

    def _create_conv_block(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def _create_fc_layers(self):
        S, B, C = self.S, self.B, self.num_classes
        return nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size()[0], -1)
        x = self.fc_layers(x)
        return x.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)
