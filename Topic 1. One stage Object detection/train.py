import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from modules.config import Config
from modules.dataset import CocoDataset
from modules.utils import YoloDetectionLoss, mean_average_precision, extract_bboxes
from yolo_model import YOLOv1


config = Config("configs/train_config.yaml")


def train():
    model = YOLOv1(S=7, B=2, num_classes=80).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = YoloDetectionLoss()
    if config.load_path:
        model.load_state_dict(torch.load(config.load_path))
    train_dataset = CocoDataset(
        image_folder="./train2017",
        annotation_path="./annotations/instances_train2017.json",
        size=(448, 448),
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}")
        loss = 0.0
        model.train()
        for imgs, boxes in train_loader:
            imgs = imgs.to(config.device)
            boxes = boxes.to(config.device)
            out = model(imgs)
            loss = loss_fn(out, boxes)
            loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        p_boxes, target_boxes = extract_bboxes(
            train_loader, model, iou_threshold=0.5, prob_threshold=0.4
        )
        mean_ap = mean_average_precision(p_boxes, target_boxes, iou_threshold=0.5)
        print(f"loss: {loss / len(train_loader)}")
        print(f"mAP: {mean_ap}")
    torch.save(model.state_dict(), config.save_path)


if __name__ == "__main__":
    train()
