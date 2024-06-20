import os
from PIL import Image
import torch
import torchvision
import torch.nn as nn
from IPython import display
from enum import Enum
import json

from dataloader import RawDataforTest
from model import Yolo
from utils import nms, test_and_draw_mAP
from metrics import InterpolationMethod, CalculationMetrics, compare_metrics, ObjectDetectionMetricsCalculator, 

model_weight_path = ''
categories = []
test_index = ''
image_path = ''
results_dir = ''

val_data = ''

if __name__ == '__main__':
  
  resnet18 = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
  backbone = nn.Sequential(*list(resnet18.children())[:-2])
  net = Yolo(backbone, backbone_out_channels=512)

  net.to('cuda')
  net.load_state_dict(torch.load(model_weight_path))
  net.eval()
  raw = RawDataforTest(val_data)
  iter_raw = get_loader(raw)
  test_and_draw_mAP(net, iter_raw, torch.device('cuda'))	

  
  

