import numpy as np
import cv2
import glob
import torch
import torchvision
from torch.utils.data import Dataset

class CustomVideoDataset(Dataset):
    def __init__(self, labels, transform=None, target_transform=None, train_flag = True):
        self.labels = labels
        if train_flag:
            self.paths = glob.glob('DATA/train/*/*.mp4')
        else:
            self.paths = glob.glob('DATA/val/*/*.mp4')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.read_img(self.paths[idx])/255
        for i in range(len(self.labels)):
            if self.labels[i] in self.paths[idx]:
                label = i
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return np.moveaxis(image, -1, 1), label

    def read_img(self, path):
        cap = cv2.VideoCapture(path)
        imgs_list = []
        i = 0
        while cap.isOpened():
            ret, img = cap.read()
            if ret == True:
                imgs_list.append(cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR))
            else:
                break
        imgs_list = np.asarray(imgs_list)[np.linspace(0, len(imgs_list)-1, 10).astype(np.int8)]
        return imgs_list

class Resnet_model(torch.nn.Module):
  def __init__(self, resnet):
    super(Resnet_model, self).__init__()
    self.resnet = resnet
    self.lin = torch.nn.Linear(400, 4)
    self.sm = torch.nn.Softmax(dim = 1)


  def forward(self, x):
    x = self.resnet(x)
    x = self.lin(x)
    x = self.sm(x)
    return x
