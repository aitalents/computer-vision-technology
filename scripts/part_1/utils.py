import numpy as np
import cv2
import glob
import tqdm
import torch
import torchvision
from torch.utils.data import Dataset

def read_img(path):
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

def prepare_dataset(labels, train = True):
    images = []
    labels_list = []
    for label in labels:
        print('Label = ', label)
        if train:
            paths = glob.glob('Data/train/'+label+'/*.mp4')
        else:
            paths = glob.glob('Data/val/'+label+'/*.mp4')
        for path in tqdm(paths):
            img = read_img(path)
            if len(img) == 10:
                images.append(img)
                labels_list.append(labels.index(label))
            else:
                print("Expected len 10 but got len {} at path: ".format(len(img)), path)
    return np.asarray(images), np.asarray(labels_list)

class CustomImageDataset(Dataset):
    def __init__(self, imgs, labels, transform=None, target_transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

#model
# модель с предобученным бэкбоном VGG16
class VGG16_bn(torch.nn.Module):
    def __init__(self):
        super(VGG16_bn, self).__init__()
        self.vgg16_backbone = torchvision.models.vgg16_bn(weights='DEFAULT').features
        self.flatten = torch.nn.Flatten()
        self.classif = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(256, 4),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.vgg16_backbone(x)
        x = self.flatten(x)
        x = self.classif(x)
        return x

def rotation_pts_tf(img, rotation='180'):
    image = img.numpy()
    if rotation == '180':
        image = cv2.rotate(image, cv2.ROTATE_180)
    if rotation == '-90':
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation =='+90':
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return torch.from_numpy(image)

def batch_augmentation(images):
    # get idxs for img augmentation
    aug_idxs = np.argwhere(np.random.randint(3, size=len(images))==2).reshape(-1)
    aug_types = ['180', '-90', '+90']

    for el in aug_idxs:
        images[el] = rotation_pts_tf(images[el], rotation=aug_types[np.random.randint(3)])
    return images

class CustomVideoDataset(Dataset):
    def __init__(self, labels, transform=None, target_transform=None, train_flag = True):
        self.labels = labels
        if train_flag:
            self.paths = glob.glob('Data/train/*/*.mp4')
        else:
            self.paths = glob.glob('Data/val/*/*.mp4')
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

class GRU_embs(torch.nn.Module):
    def __init__(self):
        super(GRU_embs, self).__init__()
        #self.lstm1 = torch.nn.LSTM(2048, 32)
        self.lstm1 = torch.nn.GRU(2048, 32)
        self.lstm2 = torch.nn.GRU(32, 32)
        self.dropout = torch.nn.Dropout(p = 0.4)
        self.lin1 = torch.nn.Linear(32*10, 4)
        self.softmax1 = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x, hn = self.lstm1(x)
        x, _ = self.lstm2(x, hn)
        x = self.dropout(x.view(len(x), -1))
        #x = self.lin1(x.view(len(x), -1))
        x = self.lin1(x)
        x = self.softmax1(x)
        return x
