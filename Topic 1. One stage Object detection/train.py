from argparse import ArgumentParser
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import COCODetDataset, batch_collate_fn
from model import YOLOv1


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

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        epoch_start_time = time.time()
        epoch_loss = 0.0

        model.train()

        for idx, batch in enumerate(train_loader):
            if idx < 1:
                imgs, dets = batch
                imgs = imgs.to(device)
                dets = dets.to(device)

                model_out = model(imgs)
                batch_loss = model.compute_loss(model_out, dets)

                batch_loss.backward()
                epoch_loss += batch_loss.item()

        epoch_loss /= len(train_loader)
        epoch_time = time.time() - epoch_start_time
        print(f'Avg epoch loss: {epoch_loss:.3f}, took {epoch_time / (60 * 60):.3f} hours')

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./yolo_v1_model_{epoch}_epoch.pth")


if __name__ == '__main__':
    parser = ArgumentParser(description='YOLOv1 train pipeline')

    parser.add_argument('--data_folder', type=str, help='Path to folder with COCO-2017 data')
    parser.add_argument('--imgsz', type=int, default=448, help='Input images resize to')
    parser.add_argument('--grid_size', type=int, default=7, help='How many row and column of cells will be in model?')
    parser.add_argument('--device', type=str, default='cpu', help='Use device: `cuda` or `cpu`')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    train(args)
