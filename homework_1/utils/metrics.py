from functools import cmp_to_key
import json
from enum import Enum
import numpy as np
import torch
from matplotlib import pyplot as plt
from nms import nms


class InterpolationMethod(Enum):
	Interpolation_11 = 1
	Interpolation_101 = 2


class CalculationMetrics():
	def __init__(self, IoU: float, confidence: float, mustbe_FP: bool):#, is_difficult: bool):

		self.IoU = IoU
		self.confidence = confidence
		self.mustbe_FP = mustbe_FP
		#self.is_difficult = is_difficult


def compare_metrics(metrics1: CalculationMetrics, metrics2: CalculationMetrics):
	if metrics1.confidence == metrics2.confidence:
		return metrics2.IoU - metrics1.IoU
	return metrics2.confidence - metrics1.confidence


class ObjectDetectionMetricsCalculator():

	def __init__(self, num_classes: int, confidence_thres: float):

		# initialize data
		self.data = [{"data": [], "detection": 0, "truth": 0} for _ in range(num_classes)]
		self.confidence_thres = confidence_thres


	def add_image_data(self, pred: torch.Tensor, truth: str):

		pred = pred.reshape(-1, 30)
		truth = json.loads(truth)

		choose_truth_index = [None for _ in range(pred.shape[0])]
		iou = [0 for _ in range(pred.shape[0])]

		for i in range(pred.shape[0]):
			score, cat = pred[i][10:30].max(dim=0)
			confidence = pred[i][4]
			# filter by confidence threshold
			if confidence * score < self.confidence_thres: continue
			
			x, y, w, h = pred[i][0:4]
			# calculate cell index
			xidx = i % 7
			yidx = i // 7
			# transform cell relative coordinates to image relative coordinates
			xhat = (x + xidx) / 7.0
			yhat = (y + yidx) / 7.0

			xmin_hat = xhat - w / 2
			xmax_hat = xhat + w / 2
			ymin_hat = yhat - h / 2
			ymax_hat = yhat + h / 2

			for j in range(len(truth)):
				bbox = truth[j]
				# judge whether is same class
				if cat != bbox['category']: continue
				# calculate IoU
				xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
				wi = min(xmax, xmax_hat) - max(xmin, xmin_hat)
				wi = max(wi, 0)
				hi = min(ymax, ymax_hat) - max(ymin, ymin_hat)
				hi = max(hi, 0)
				intersection = wi * hi
				union = (xmax - xmin) * (ymax - ymin) + (xmax_hat - xmin_hat) * (ymax_hat - ymin_hat) - intersection
				this_iou = intersection / (union + 1e-6)
				# determine whether to choose this ground truth
				if iou[i] is None: choose = True
				elif iou[i] < this_iou: choose = True
				else: choose = False
				# if choose, assign value
				if choose:
					iou[i] = this_iou
					choose_truth_index[i] = j
		# init a bool array for judging mustbe_FP later
		truth_chosen = [False for _ in range(len(truth))]
		# sort according to IoU
		sort_idx = np.argsort(iou)[::-1]
		# add into metrics
		for i in sort_idx:
			score, cat = pred[i][10:30].max(dim=0)
			confidence = pred[i][4]
			# filter by confidence threshold
			if confidence * score < self.confidence_thres: continue

			truth_index = choose_truth_index[i]
			if truth_index == None: 
				mustbe_FP = True
				is_difficult = False
			elif truth_chosen[truth_index]:
				mustbe_FP = True
				#is_difficult = truth[choose_truth_index[i]]['difficult']
			else: 
				mustbe_FP = False
				truth_chosen[choose_truth_index[i]] = True
				#is_difficult = truth[choose_truth_index[i]]['difficult']

			self.data[cat]['data'].append(CalculationMetrics(iou[i], float(confidence * score), mustbe_FP))#, is_difficult))

			# update detection statistics
			self.data[cat]['detection'] += 1
		# update ground truth statistics
		for bbox in truth:
            #if bbox['difficult']: continue
			self.data[bbox['category']]['truth'] += 1


	def calculate_precision_recall(self, iou_thres: float, class_idx: int) -> list:

		ret = []
		# retrieve count
		truth_cnt = self.data[class_idx]['truth']
		# accumulated TP
		acc_TP = 0
		# accumulated difficult count
		acc_difficult = 0
		# sort metrics by confidence
		data = sorted(self.data[class_idx]['data'], key=cmp_to_key(compare_metrics))
		for i, metrics in enumerate(data):
			if metrics.IoU >= iou_thres and not metrics.mustbe_FP and not metrics.is_difficult:
				acc_TP += 1
			if metrics.is_difficult:
				acc_difficult += 1
			if i + 1 - acc_difficult > 0:
				ret.append({
					'precision': acc_TP / (i + 1 - acc_difficult),
					'recall': acc_TP / truth_cnt
				})
		
		return ret


	def calculate_average_precision(self, iou_thres: float, class_idx: int, itpl_option: InterpolationMethod) -> float:
		"""Calculate Average Precision (AP)

		Args:
			iou_thres (float): IoU Threshold
			class_idx (int): Class Index
			itpl_option (InterpolationMethod): Interpolation Method

		Returns:
			float: AP of specified class using provided interpolation method
		"""
		prl = self.calculate_precision_recall(iou_thres=iou_thres, class_idx=class_idx)

		if itpl_option == InterpolationMethod.Interpolation_11:
			intp_pts = [0.1 * i for i in range(11)]
		elif itpl_option == InterpolationMethod.Interpolation_101:
			intp_pts = [0.01 * i for i in range(101)]
		else:
			raise Exception('Unknown Interpolation Method')

		max_dict = {}
		gmax = 0

		for pr in prl[::-1]:
			gmax = max(gmax, pr['precision'])
			max_dict[pr['recall']] = gmax

		if len(max_dict) < 1: return 0.

		max_keys = max_dict.keys()
		max_keys = sorted(max_keys)

		key_ptr = len(max_keys) - 2
		last_key = max_keys[-1]

		AP = 0

		for query in intp_pts[::-1]:
			if key_ptr < 0:
				if query > last_key:
					ans = 0
				else:
					ans = max_dict[last_key]
			else:
				if query > last_key:
					ans = 0
				elif query > max_keys[key_ptr]:
					ans = max_dict[last_key]
				else:
					while key_ptr >= 0:
						if query > max_keys[key_ptr]:
							break
						last_key = max_keys[key_ptr]
						key_ptr -= 1
					ans = max_dict[last_key]
			AP += ans

		AP /= len(intp_pts)
		return AP


	def calculate_mAP(self, iou_thres: float, itpl_option: InterpolationMethod) -> float:
		mAP = 0
		for c in range(len(self.data)):
			mAP += self.calculate_average_precision(iou_thres, c, itpl_option)
		mAP /= len(self.data)

		return mAP




def test_and_draw_mAP(net, test_iter_raw, device):
        net.eval()
        print("Changed to eval")
        net.to(device)
        calc = ObjectDetectionMetricsCalculator(80, 0.1)
        for i, (X, YRaw) in enumerate(test_iter_raw):
            if i % 1000 = 0:
                print(f'Calculating {i}...')
            #to_tensor = torchvision.transforms.ToTensor()
            #X = to_tensor(img).unsqueeze_(0).to(device)
            X = X.to(device)
            YHat = net(X)
            for yhat, yraw in zip(YHat, YRaw):
                yhat = nms(yhat)
                calc.add_image_data(yhat.cpu(), yraw)
        print("Finished calculating")
        print("mAP on validation:", calculate_mAP(0.5, InterpolationMethod.Interpolation_11))
        #for i in range(80):
            #draw_precision_recall(calc.calculate_precision_recall(0.5, i), i)
