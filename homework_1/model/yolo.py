import torch
import torch.nn as nn
from backbone import YoloBackbone


class Yolo(nn.Module):
	def __init__(self, backbone: YoloBackbone = YoloBackbone(), backbone_out_channels=1024, n_classes=80):
		self.n_classes = n_classes
		super(Yolo, self).__init__()
		self.backbone = backbone
		self.head = nn.Sequential(
			nn.Conv2d(backbone_out_channels, 1024, kernel_size=3, padding=1),
			# nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2),
			# nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
			# nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
			# nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Flatten(),
			nn.Linear(7*7*1024, 4096),
			nn.Dropout(0.5),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Linear(4096, 7 * 7 *(2 * 5 + self.n_classes)),
			nn.Sigmoid(),
			nn.Unflatten(1, (7, 7, (2 * 5 + self.n_classes)))
		)
		self.net = nn.Sequential(self.backbone, self.head)

	def forward(self, X):
		return self.net(X)
