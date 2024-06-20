import sys
import torch
sys.path.append("E:\\cv_hw_1")
from dataloader.get_coco import load_coco, get_torch
from dataloader.transform import FiftyOneTorchDataset, Transformtsz, collate
from utils.loss import yolo_loss
from utils.weight_init import weight_init
import os
import time
from tqdm import tqdm
import numpy as np
from yolo import Yolo
import torchvision
import torch.nn as nn

class Accumulator(object):
	"""
	Sum a list of numbers over time
	from: https://github.com/dsgiitr/d2l-pytorch/blob/master/d2l/base.py
	"""
	def __init__(self, n):
		self.data = [0.0] * n
	def add(self, *args):
		self.data = [a + b for a, b in zip(self.data, args)]
	def reset(self):
		self.data = [0] * len(self.data)
	def __getitem__(self, i):
		return self.data[i]
	
class Timer(object):
	"""Record multiple running times."""
	def __init__(self):
		self.times = []
		self.start()

	def start(self):
		"""Start the timer"""
		self.start_time = time.time()

	def stop(self):
		"""Stop the timer and record the time in a list"""
		self.times.append(time.time() - self.start_time)
		return self.times[-1]

	def avg(self):
		"""Return the average time"""
		return sum(self.times)/len(self.times)

	def sum(self):
		"""Return the sum of time"""
		return sum(self.times)

	def cumsum(self):
		"""Return the accumuated times"""
		return np.array(self.times).cumsum().tolist()

def train(net, train_iter, test_iter, num_epochs, lr, momentum=0.9, weight_decay=5e-4, accum_batch_num=1, save_path='./chkpt', load=None, load_epoch=-1, pretrained=False):
	'''
	Train net work. Some notes for load & load_epoch:
	:param load: the file of model weights to load
	:param load_epoch: num of epoch already completed (minus 1). should be the same with the number in auto-saved file name.
	'''

	def print_and_log(msg, log_file):
		print(msg)
		with open(log_file, 'a', encoding='utf8') as f:
			f.write(msg + '\n')

	def update_lr(opt, lr):
		for param_group in opt.param_groups:
			param_group['lr'] = lr

	os.makedirs(save_path, exist_ok=True)
	log_file = os.path.join(save_path, f'log-{time.time_ns()}.txt')

	if load:
		net.load_state_dict(torch.load(load))
	elif pretrained:
		net.head.apply(weight_init)
	else:
		# init params
		net.apply(weight_init)

	if not torch.cuda.is_available():
		net = net.to(torch.device('cpu'))
		devices = [torch.device('cpu')]
	else:
		net = net.to(torch.device('cuda'))
		devices = [torch.device('cuda')]

	# define optimizer
	if isinstance(lr, float):
		tlr = lr
	else: tlr = 0.001

	optimizer = torch.optim.SGD(net.parameters(), lr=tlr, momentum=momentum, weight_decay=weight_decay)

	# visualization

	num_batches = len(train_iter)
	# train
	for epoch in range(num_epochs - load_epoch - 1):
		# adjust true epoch number according to pre_load
		epoch = epoch + load_epoch + 1

		# define metrics: train loss, sample count
		metrics = Accumulator(2)
		# define timer
		timer = Timer()

		# train
		net.train()

		# set batch accumulator
		accum_cnt = 0
		accum = 0
		loop = tqdm(train_iter, leave=True)

		for batch_idx, (X, y) in enumerate(loop):
			timer.start()

			X, y = X.to(devices[0]), y.to(devices[0])
			yhat = net(X)
			
			loss_val = yolo_loss(yhat, y)
			# print(loss_val)

			# backward to accumulate gradients
			loss_val.sum().backward()
			# step
			optimizer.step()
			# clear
			optimizer.zero_grad()


			# update metrics
			with torch.no_grad():
				metrics.add(loss_val.sum().cpu(), X.shape[0])
			train_l = metrics[0] / metrics[1]

			timer.stop()

			# log & visualization
			if (batch_idx + 1) % (num_batches // 5) == 0 or batch_idx == num_batches - 1:
				print_and_log("epoch: %d, batch: %d / %d, loss: %.4f, time: %.4f" % (epoch, batch_idx + 1, num_batches, train_l.item(), timer.sum()), log_file)

		# redefine metrics: test loss, test sample count
		metrics = Accumulator(2)
		# redefine timer
		timer = Timer()
		# test
		net.eval()

		with torch.no_grad():
			
			test_loop = tqdm(test_iter, leave=True)
			for batch_idx, (X, y) in enumerate(test_loop):
				timer.start()

				X, y = X.to(devices[0]), y.to(devices[0])
				yhat = net(X)

				loss_val = yolo_loss(yhat, y)
				metrics.add(loss_val.sum().cpu(), X.shape[0])
				test_l = metrics[0] / metrics[1]
				timer.stop()
				if (batch_idx + 1) % (num_batches // 5) == 0 or batch_idx == num_batches - 1:
					print_and_log("epoch: %d, batch: %d / %d, test loss: %.4f, time: %.4f" % (epoch, batch_idx + 1, num_batches, test_l.item(), timer.sum()), log_file)

		# save model
		torch.save(net.state_dict(), os.path.join(save_path, f'./{time.time_ns()}-epoch-{epoch}.pth'))


if __name__ == "__main__":

	# get coco dataset
	dataset = load_coco(15000)

	classes = dataset.distinct(
		"ground_truth.detections.label"
		)

	train_data = dataset.match_tags("train")
	test_data = dataset.match_tags("test")
	val_data = dataset.match_tags("validation")

	train_dataset = FiftyOneTorchDataset(train_data, transforms=Transformtsz(resize=(448, 448)), classes=classes)
	val_dataset_test = FiftyOneTorchDataset(val_data, transforms=Transformtsz(resize=(448, 448)), classes=classes)
	test_dataset_test = FiftyOneTorchDataset(test_data, transforms=Transformtsz(resize=(448, 448)), classes=classes)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=collate)#, sampler=train_sampler)
	val_loader = torch.utils.data.DataLoader(val_dataset_test, batch_size=8, shuffle=False, collate_fn=collate)#, sampler=train_sampler)
	test_loader = torch.utils.data.DataLoader(test_dataset_test, batch_size=8, shuffle=False, collate_fn=collate)#, sampler=train_sampler)

	# resnet18 = torchvision.models.resnet18(pretrained=True)
	# net = Yolo() # classical YoloV1 with our backbone
	# resnet 18 backbone
	# remove avg pool and fc
	resnet18 = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
	backbone = nn.Sequential(*list(resnet18.children())[:-2])
	for param in backbone.parameters():
		param.requires_grad = False
	net = Yolo(backbone, backbone_out_channels=512)
	train(net, train_iter=train_loader, test_iter=test_loader, num_epochs=20, lr=0.0001)

