# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:27:43 2024

@author: M
"""

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class Coco(Dataset):
    def __init__(self, ann_path, files_dir, S=7, B=2, C=80, transform=None):
        self.ann_path = ann_path
        self.files_dir = files_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

        # read json file
        with open(ann_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data["images"])
        #return 25000 - train
    
    def normalize(self, img_info, img):
        boxes = []
        #for annotation in self.data["annotations"][:25000]: - train
        for annotation in self.data["annotations"]:
            if annotation["image_id"] == img_info["id"]:
                bbox = annotation["bbox"]
                class_label = annotation["category_id"] - 1
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2
                box_width = bbox[2]
                box_height = bbox[3]

                
                img_width, img_height = img.size
                center_x = center_x / img_width
                center_y = center_y / img_height
                box_width = box_width / img_width
                box_height = box_height / img_height

                boxes.append([class_label, center_x, center_y, box_width, box_height])
        
        return boxes
    

    def __getitem__(self, index):
        image_info = self.data["images"][index]
        image_path = os.path.join(self.files_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        boxes_list = torch.tensor(self.normalize(image_info, image))

        if self.transform:
            image, boxes_list = self.transform(image, boxes_list)
        matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes_list:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = width * self.S, height * self.S

            if matrix[i, j, self.C] == 0:
                matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                matrix[i, j, self.C + 1 : self.C + 5] = box_coordinates
                matrix[i, j, class_label] = 1
        
    
        return image, matrix