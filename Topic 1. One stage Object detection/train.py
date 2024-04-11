import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import EfficientNetV2S_YOLOv1
from dataset import CoCoDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

seed = 42
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 6e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 24
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "data/overfit.pth.tar"
IMG_DIR = "data/val2017"
LABEL_DIR = "data/annotations/val_annot"


class Compose(object):
    def __init__(self, img_transforms, bbox_transforms=None):
        self.img_transforms = img_transforms
        self.bbox_transforms = bbox_transforms if bbox_transforms is not None else []

    def __call__(self, img, bboxes):
        for t in self.img_transforms:
            img = t(img)  # Apply image-only transforms

        for t in self.bbox_transforms:
            img, bboxes = t(img, bboxes)  # Apply transforms that need both image and bboxes

        return img, bboxes


img_transforms = [transforms.ToTensor(),
                  transforms.Resize((256, 256), antialias=True),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

bbox_transforms = []  # Add any transforms that need to manipulate both img and bboxes here

transform = Compose(img_transforms, bbox_transforms)


list_of_losses = []
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

        # update progress bar
        loop.set_postfix(loss=loss.item())


    mean_loss_compute = sum(mean_loss) / len(mean_loss) if mean_loss else 1000
    list_of_losses.append(mean_loss_compute)
    print(f"Mean loss was {mean_loss_compute}")


def main():
    model = EfficientNetV2S_YOLOv1().to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = CoCoDataset(
        "./data/annotations/val.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = CoCoDataset(
        "./data/annotations/val.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        train_fn(train_loader, model, optimizer, loss_fn)

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)


if __name__ == "__main__":
    main()
    print(list_of_losses)
