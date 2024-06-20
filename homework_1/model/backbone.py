import torch
import torch.nn as nn


class YoloBackbone(nn.Module):
	def __init__(self):
		super(YoloBackbone, self).__init__()
		conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
			# nn.BatchNorm2d(64),
			nn.LeakyReLU(0.1, inplace=True)
		)
		pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		conv2 = nn.Sequential(
			nn.Conv2d(64, 192, kernel_size=3, padding=1),
			# nn.BatchNorm2d(192),
			nn.LeakyReLU(0.1, inplace=True)
		)
		pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		conv3 = nn.Sequential(
			nn.Conv2d(192, 128, kernel_size=1),
			# nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			# nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(256, 256, kernel_size=1),
			# nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			# nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1, inplace=True)
		)
		pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		conv4_part = nn.Sequential(
			nn.Conv2d(512, 256, kernel_size=1),
			# nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			# nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1, inplace=True)
		)
		conv4_modules = []
		for _ in range(4):
			conv4_modules.append(conv4_part)
		conv4 = nn.Sequential(
			*conv4_modules,
			nn.Conv2d(512, 512, kernel_size=1),
			# nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			# nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True)
		)
		pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
		conv5 = nn.Sequential(
			nn.Conv2d(1024, 512, kernel_size=1),
			# nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			# nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(1024, 512, kernel_size=1),
			# nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			# nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True)
		)
		self.net = nn.Sequential(conv1, pool1, conv2, pool2, conv3, pool3, conv4, pool4, conv5)

	def forward(self, X):
		return self.net(X)
