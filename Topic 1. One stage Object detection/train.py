from argparse import ArgumentParser
import os
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import COCODetDataset, batch_collate_fn
from model import YOLOv1
from utils import nms, render_pred_bboxes


def evaluate(model, dataloader, epoch):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for batch in tqdm(dataloader, total=len(dataloader), desc=f'Epoch #{epoch}: val'):
            imgs, dets = batch
            imgs = imgs.to(model.device)
            dets = dets.to(model.device)

            pred = model(imgs)
            batch_loss = model.compute_loss(pred, dets)

            val_loss += batch_loss.item()

    val_loss /= len(dataloader)

    return val_loss


def show_pred_example(model, img, conf_thr, iou_thr):
    model.eval()

    model_out = model(img)
    pred_boxes = model.convert_preds_to_bboxes_list(model_out)[0]
    nms_boxes = nms(pred_boxes, model.num_classes, iou_thr=iou_thr)
    
    final_boxes = [box for box in nms_boxes if box[5] >= conf_thr]
    rendered_img = render_pred_bboxes(img[0, :, :, :], final_boxes)

    return rendered_img


def train(args):
    coco_dataset_path = args.data_folder
    train_data_folder = os.path.join(coco_dataset_path, 'train')
    val_data_folder = os.path.join(coco_dataset_path, 'validation')

    train_dataset = COCODetDataset(train_data_folder)
    val_dataset = COCODetDataset(val_data_folder)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        collate_fn=batch_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        drop_last=False,
        collate_fn=batch_collate_fn,
    )

    model = YOLOv1(
        train_dataset.class_names, args.grid_size,
        (args.imgsz, args.imgsz), args.device,
    )
    device = model.device
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=2e-5, weight_decay=0
    )
    optimizer.zero_grad()

    min_val_loss = None

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        epoch_start_time = time.time()
        epoch_loss = 0.0

        model.train()

        for batch in tqdm(train_loader, total=len(train_loader), desc=f'Epoch #{epoch}: train'):
            optimizer.zero_grad()
                
            imgs, dets = batch
            imgs = imgs.to(device)
            dets = dets.to(device)

            model_out = model(imgs)
            batch_loss = model.compute_loss(model_out, dets)
            epoch_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

        epoch_loss /= len(train_loader)
        val_loss = evaluate(model, val_loader, epoch)

        epoch_time = time.time() - epoch_start_time
        print(f'Avg train loss: {epoch_loss:.3f}, avg val loss: {val_loss:.3f},  took {epoch_time / (60 * 60):.3f} hours')

        if min_val_loss is None or val_loss <= min_val_loss:
            torch.save(model.state_dict(), f"./yolo_v1_model_{epoch}_epoch.pth")
            min_val_loss = val_loss

        first_val_img, _ = val_dataset[0]
        show_pred_example(model, first_val_img[None, :, :, :], args.conf_thres, args.iou_thres)


if __name__ == '__main__':
    parser = ArgumentParser(description='YOLOv1 train pipeline')

    parser.add_argument('--data_folder', type=str, help='Path to folder with COCO-2017 data')
    parser.add_argument('--imgsz', type=int, default=448, help='Input images resize to')
    parser.add_argument('--grid_size', type=int, default=7, help='How many row and column of cells will be in model?')
    parser.add_argument('--device', type=str, default='cpu', help='Use device: `cuda` or `cpu`')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='Confidence threshold')

    args = parser.parse_args()

    train(args)
