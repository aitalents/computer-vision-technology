import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import CocoDataset
from src.dataset.preprocessing import Compose
from src.models.loss import YoloLoss, mean_average_precision
from src.models.loss_utils import get_bboxes
from src.models.utils import load_checkpoint, save_checkpoint
from src.models.yolo import YoloV1

LEARNING_RATE = 2e-5
DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # 64 in original paper but resource exhausted error otherwise.
WEIGHT_DECAY = 0
EPOCHS = 20
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "model.pth"


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

    train_dataset = CocoDataset(
        transform=transform,
        files_dir="./train2017",
        ann_path="./annotations/instances_train2017.json",
    )

    test_dataset = CocoDataset(
        transform=transform,
        files_dir="./val2017",
        ann_path="./annotations/instances_val2017.json",
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        scheduler.step(mean_avg_prec)

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
