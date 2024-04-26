# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 10:28:28 2024

@author: M
"""

import torch
import torch.optim as optim
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset.dataset import Coco
#from src.dataset.preprocessing import Compose #-
from src.metric.loss import YoloLoss
from src.metric.utils import get_bboxes, mean_average_precision
from src.model.model import YoloV1

LEARNING_RATE = 2e-5
DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # 64 in original paper but resource exhausted error otherwise.
WEIGHT_DECAY = 0
EPOCHS = 1
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = f"{os.getcwd()}\src\hw_1_model.pth"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


def load_checkpoint(checkpoint, model, optimizer):
    print("...Loading_checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")

if __name__ == "__main__":
    model = YoloV1(split_size=7, num_boxes=2, num_classes=80).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=0.1, patience=3, mode="max", verbose=True
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


    test_dataset = Coco(
        transform=transform,
        files_dir="./src/dataset/val2017",
        ann_path="./src/dataset/annotations/instances_val2017.json",
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    for epoch in range(EPOCHS):
        
        model.eval()
        train_fn(test_loader, model, optimizer, loss_fn)

        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"mAP: {mean_avg_prec}")

        
