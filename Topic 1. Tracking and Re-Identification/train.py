import torch
import torch.nn as nn
import os
import json
import torch.optim as optim
import PIL
import torchvision.transforms as transforms

from utils import DetectionUtils
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model import YOLOv1


class YOLOv1Loss(nn.Module):
    """
    Расчет потерь модели YOLOv1, включая потери локализации объектов,
    уверенности в объектах и предсказания классов.
    """

    def __init__(self, detection_utils, grid_size=7, num_boxes=2, num_classes=91):
        super(YOLOv1Loss, self).__init__()
        self.detection_utils = detection_utils
        self.grid_size = grid_size  # Размер сетки изображения
        self.num_boxes = num_boxes  # Количество предсказываемых рамок на сетку
        self.num_classes = num_classes  # Количество классов объектов
        self.mse_loss = nn.MSELoss(reduction="sum")  # Среднеквадратичное отклонение

        # Коэффициенты для потерь: для несуществующих объектов и координат рамок
        self.lambda_no_object = 0.5
        self.lambda_coordinates = 5

    def forward(self, predictions, targets):
        # Преобразуем предсказания для удобства работы
        predictions = predictions.reshape(-1, self.grid_size, self.grid_size, self.num_classes + self.num_boxes * 5)

        # Вычисляем IoU для двух рамок предсказания
        iou_box1 = self._compute_iou(predictions[..., self.num_classes+1:self.num_classes+5], targets[..., self.num_classes+1:self.num_classes+5])
        iou_box2 = self._compute_iou(predictions[..., self.num_classes+6:self.num_classes+10], targets[..., self.num_classes+1:self.num_classes+5])
        ious = torch.stack([iou_box1, iou_box2])

        # Выбираем рамку с наибольшим IoU
        iou_maxes, best_boxes = torch.max(ious, dim=0)
        presence_mask = targets[..., self.num_classes].unsqueeze(3)  # Маска наличия объекта

        # Расчет потерь для координат рамок
        box_predictions = self._select_best_box(predictions, best_boxes)
        box_targets = presence_mask * targets[..., self.num_classes+1:self.num_classes+5]
        box_predictions[..., 2:4] = torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse_loss(box_predictions.view(-1, 4), box_targets.view(-1, 4))

        # Расчет потерь для уверенности в наличии объекта
        object_predictions = self._select_best_box_confidence(predictions, best_boxes)
        object_loss = self.mse_loss(object_predictions.view(-1), (presence_mask * targets[..., self.num_classes:self.num_classes+1]).view(-1))

        # Расчет потерь для фона (отсутствие объекта)
        no_object_loss = self._compute_no_object_loss(predictions, targets, presence_mask)

        # Расчет потерь для классификации объектов
        class_predictions = presence_mask * predictions[..., :self.num_classes]
        class_targets = presence_mask * targets[..., :self.num_classes]
        class_loss = self.mse_loss(class_predictions.view(-1, self.num_classes), class_targets.view(-1, self.num_classes))

        # Общие потери
        total_loss = (self.lambda_coordinates * box_loss + object_loss + self.lambda_no_object * no_object_loss + class_loss)
        return total_loss

    def _compute_iou(self, box1, box2):
        # IoU
        return self.detection_utils.calculate_iou(box1, box2)

    def _select_best_box(self, predictions, best_boxes):
        # Выбор предсказаний рамок с наибольшим IoU
        return predictions[..., self.num_classes+1:self.num_classes+5].where(best_boxes.unsqueeze(-1), predictions[..., self.num_classes+6:self.num_classes+10])

    def _select_best_box_confidence(self, predictions, best_boxes):
        # Выбор уверенности рамок с наибольшим IoU
        return predictions[..., self.num_classes:self.num_classes+1].where(best_boxes, predictions[..., self.num_classes+5:self.num_classes+6])

    def _compute_no_object_loss(self, predictions, targets, presence_mask):
        # Расчет потерь для предсказаний, где объект отсутствует
        no_object_predictions = (1 - presence_mask) * predictions[..., self.num_classes:self.num_classes+1]
        no_object_targets = (1 - presence_mask) * targets[..., self.num_classes:self.num_classes+1]
        return self.mse_loss(no_object_predictions.view(-1), no_object_targets.view(-1))


# class ImageResizer:
#     def __init__(self, target_size):
#         self.target_size = target_size

#     def __call__(self, image: Image.Image, boxes):
#         # Получаем исходные размеры изображения
#         original_width, original_height = image.size
#         # Вычисляем коэффициенты масштабирования для обеих осей
#         scale_x = self.target_size[1] / original_width
#         scale_y = self.target_size[0] / original_height

#         # Изменяем размер изображения
#         resized_image = FT.resize(image, self.target_size)

#         # Преобразуем изображение в тензор
#         image_tensor = FT.to_tensor(resized_image)

#         # Масштабируем ограничивающие рамки
#         scaled_boxes = [
#             (xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y)
#             for (xmin, ymin, xmax, ymax) in boxes
#         ]
        
#         return image_tensor, scaled_boxes

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


class COCODataset(Dataset):
    def __init__(self, annotations_file, images_directory, anchor_sizes, grid_size=7, num_boxes=2, num_classes=3, transformations=None):
        self.annotations_file = annotations_file
        self.images_directory = images_directory
        self.transformations = transformations
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.anchors = AnchorBoxes(anchor_sizes, grid_size, image_size=448)
        self.detection_utils = DetectionUtils()

        with open(self.annotations_file, 'r') as file:
            self.dataset = json.load(file)

        self.filtered_annotations = [
            ann for ann in self.dataset['annotations'] if ann['category_id'] in [1, 2, 3]
        ]

        image_ids = set(ann['image_id'] for ann in self.filtered_annotations)

        self.filtered_images = [
            img for img in self.dataset['images'] if img['id'] in image_ids
        ]

    def __len__(self):
        return len(self.filtered_images)
    
    def __getitem__(self, index):
        image_info = self.filtered_images[index]
        image_path = os.path.join(self.images_directory, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
    
        image_annotations = [
            ann for ann in self.filtered_annotations if ann['image_id'] == image_info['id']
        ]

        bounding_boxes = []
        for annotation in image_annotations:
            class_id = annotation['category_id'] - 1
            bbox = annotation['bbox']
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            box_width = bbox[2]
            box_height = bbox[3]

            img_width, img_height = image.size
            center_x /= img_width
            center_y /= img_height
            box_width /= img_width
            box_height /= img_height

            bounding_boxes.append([class_id, center_x, center_y, box_width, box_height])

        bounding_boxes = torch.tensor(bounding_boxes)

        if self.transformations:
            image, bounding_boxes = self.transformations(image, bounding_boxes)

        label_matrix = torch.zeros((self.grid_size, self.grid_size, self.num_classes + 5 * self.num_boxes))

        if len(bounding_boxes) > 0:
            assigned_anchors = self.anchors.assign_anchors(bounding_boxes[:, 1:], self.detection_utils.calculate_iou)

            for idx, (class_id, center_x, center_y, box_width, box_height) in enumerate(bounding_boxes):
                grid_x, grid_y, anchor_idx = assigned_anchors[idx].tolist()
                x_cell, y_cell = self.grid_size * center_x - grid_x, self.grid_size * center_y - grid_y
                width_cell, height_cell = box_width * self.grid_size, box_height * self.grid_size

                if label_matrix[grid_x, grid_y, self.num_classes] == 0:
                    label_matrix[grid_x, grid_y, self.num_classes] = 1
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    label_matrix[grid_x, grid_y, self.num_classes + 1 + 5 * anchor_idx:self.num_classes + 5 * (anchor_idx + 1)] = box_coordinates
                    label_matrix[grid_x, grid_y, class_id] = 1

        return image, label_matrix



def train_model(data_loader, model, optimizer, criterion, device='cuda'):
    model.train()
    total_loss = []
    progress_bar = tqdm(data_loader, leave=True)

    for _, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        total_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    average_loss = sum(total_loss) / len(total_loss)
    print(f"AV_LOSS: {average_loss:.4}")
    return average_loss

def main(detection_utils, transform):
    anchor_sizes = [(0.28, 0.23), (0.34, 0.56)]
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3, mode='max', verbose=True)
    loss_fn = YOLOv1Loss(detection_utils, grid_size=7, num_boxes=2, num_classes=3)

    if LOAD_MODEL:
        detection_utils.load_model_state(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = COCODataset(
        annotations_file='/annotations/instances_train2017.json',
        images_directory='/data/train',
        anchor_sizes=anchor_sizes,
        transformations=transform
    )

    test_dataset = COCODataset(
        annotations_file='/annotations/instances_val2017.json',
        images_directory='/data/val',
        anchor_sizes=anchor_sizes,
        transformations=transform
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    for _ in range(EPOCHS):
        train_model(train_loader, model, optimizer, loss_fn)

        pred_boxes, target_boxes = detection_utils.get_filtered_boxes(train_loader,


LEARNING_RATE = 3e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
WEIGHT_DECAY = 0
EPOCHS = 5
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "model.pth"

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
detection_utils = DetectionUtils()

main(detection_utils, transform)
