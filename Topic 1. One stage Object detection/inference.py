import torch
from torch.utils.data import DataLoader

from modules.config import Config
from modules.dataset import CocoDataset
from modules.utils import YoloDetectionLoss, mean_average_precision, extract_bboxes
from yolo_model import YOLOv1


config = Config("configs/val_config.yaml")


def inference():
    model = YOLOv1(S=7, B=2, num_classes=80).to(config.device)
    loss_fn = YoloDetectionLoss()
    model.load_state_dict(torch.load(config.load_path))
    val_dataset = CocoDataset(
        image_folder="./val2017",
        annotation_path="./annotations/instances_val2017.json",
        size=(448, 448),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    loss = 0.0
    model.eval()
    for imgs, boxes in val_loader:
        imgs = imgs.to(config.device)
        boxes = boxes.to(config.device)
        out = model(imgs)
        loss = loss_fn(out, boxes)
        loss += loss.item()
    p_boxes, target_boxes = extract_bboxes(
        val_loader, model, iou_threshold=0.5, prob_threshold=0.4
    )
    mean_ap = mean_average_precision(p_boxes, target_boxes, iou_threshold=0.5)
    print(f"loss: {loss / len(val_loader)}")
    print(f"mAP: {mean_ap}")


if __name__ == "__main__":
    inference()
