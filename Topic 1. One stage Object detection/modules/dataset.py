import json
import os

import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(
        self,
        annotation_path,
        image_folder,
        size=(448, 448),
        S=7,
        B=2,
        C=80,
    ):
        self.annotation_path = annotation_path
        self.image_folder = image_folder
        self.size = size
        self.S = S
        self.B = B
        self.C = C
        with open(annotation_path, "r") as f:
            self.coco_data = json.load(f)

    def __len__(self):
        return len(self.coco_data["images"])

    def __getitem__(self, index):
        image_info = self.coco_data["images"][index]
        image_path = os.path.join(self.image_folder, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        t = transforms.Compose([transforms.Resize(self.size), transforms.ToTensor()])
        image = t(image)
        boxes = self.get_boxes(image_info, image)
        label_matrix = self.get_label_matrix(boxes)
        return image, label_matrix

    def get_boxes(self, image_info, image):
        boxes = []
        for annotation in self.coco_data["annotations"]:
            if annotation["image_id"] == image_info["id"]:
                boxes.append(self.coco_to_yolo(annotation, image))
        return torch.tensor(boxes)

    @staticmethod
    def coco_to_yolo(annotation, image):
        bbox = annotation["bbox"]
        label = annotation["category_id"] - 1
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        width = bbox[2]
        height = bbox[3]
        image_width, image_height = image.size
        center_x /= image_width
        center_y /= image_height
        width /= image_width
        height /= image_height
        return [label, center_x, center_y, width, height]

    def get_label_matrix(self, boxes):
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            label, x, y, width, height = box.tolist()
            label = int(label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            cell_width, cell_height = width * self.S, height * self.S
            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, cell_width, cell_height]
                )
                label_matrix[i, j, self.C + 1 : self.C + 5] = box_coordinates
                label_matrix[i, j, label] = 1
        return label_matrix
