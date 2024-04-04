import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.dataset import CocoDataset
from src.dataset.preprocessing import Compose
from src.models.loss import YoloLoss, mean_average_precision
from src.models.loss_utils import get_bboxes
from src.models.utils import load_checkpoint
from src.models.yolo import YoloV1
from train import train_fn

EPOCHS = 1
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # 64 in original paper but resource exhausted error otherwise.
WEIGHT_DECAY = 0
EPOCHS = 20
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "model.pth"


if __name__ == "__main__":
    model = YoloV1(split_size=7, num_boxes=2, num_classes=80).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

    test_dataset = CocoDataset(
        transform=transform,
        files_dir="./val2017",
        ann_path="./annotations/instances_val2017.json",
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
        print(f"Test MAP: {mean_avg_prec}")
