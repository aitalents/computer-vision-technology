import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, ann_path, files_dir, S=7, B=2, C=80, transform=None):
        self.ann_path = ann_path
        self.files_dir = files_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

        # Load COCO data from JSON file
        with open(ann_path, "r") as f:
            self.coco_data = json.load(f)

    def __len__(self):
        return len(self.coco_data["images"])

    def __getitem__(self, index):
        img_info = self.coco_data["images"][index]
        img_path = os.path.join(self.files_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        boxes = []
        for ann in self.coco_data["annotations"]:
            if ann["image_id"] == img_info["id"]:
                bbox = ann["bbox"]
                class_label = ann["category_id"] - 1
                centerx = bbox[0] + bbox[2] / 2
                centery = bbox[1] + bbox[3] / 2
                boxwidth = bbox[2]
                boxheight = bbox[3]

                # Normalize coordinates
                img_width, img_height = image.size
                centerx /= img_width
                centery /= img_height
                boxwidth /= img_width
                boxheight /= img_height

                boxes.append([class_label, centerx, centery, boxwidth, boxheight])

        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, self.C + 1 : self.C + 5] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
