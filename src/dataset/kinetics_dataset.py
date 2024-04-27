import os

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class KineticsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str):
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.dataset_dir = dataset_dir
        self.classes = sorted(os.listdir(dataset_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            class_dir = os.path.join(dataset_dir, cls)
            for video_file in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, video_file), self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        N_frames = 250

        if len(frames) < N_frames:
            while len(frames) < N_frames:
                frames.append(torch.zeros_like(frames[0]))
        elif len(frames) > N_frames:
            frames = frames[:N_frames]

        frames = torch.stack(frames)
        frames = frames.permute(1, 0, 2, 3)  # (channels, num_frames, height, width)

        return frames, label
