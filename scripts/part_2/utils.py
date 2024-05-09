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
            imgs_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else: 
            break
    imgs_list = [imgs_list[el] for el in np.linspace(0, len(imgs_list)-1, 10).astype(np.int8)]
    return imgs_list

def get_results_lms(results):
    x = []
    y = []
    z = []
    if results.pose_landmarks != None:
        for lm in results.pose_landmarks.landmark:
            x.append(lm.x)
            y.append(lm.y)
            z.append(lm.z)
        lms = np.stack([x, y ,z])
        lms = lms.reshape(-1)
    else:
        lms = np.zeros(99)
    return lms
    
def prep_data(labels, train_flag=True):
    if train_flag:
        paths = glob.glob('Data/train/*/*.mp4')
    else:
        paths = glob.glob('Data/val/*/*.mp4')

    lms_list = []
    labels_list = []
    for i in tqdm(range(len(paths))):
        image = read_img(paths[i])
        for j in range(len(labels)):
            if labels[j] in paths[i]:
                label = j
        lms = []
        for img in image:
            lms.append(get_results_lms(pose.process(img)))
        lms = np.asarray(lms)
        
        lms_list.append(lms)
        labels_list.append(label)
    return np.asarray(lms_list), np.asarray(labels_list)

class CustomLMSDataset(Dataset):
    def __init__(self, lms, labels):
        self.lms = lms
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        lm = self.lms[idx]
        label = self.labels[idx]
        return lm, label

class LSTM_model(torch.nn.Module):
    def __init__(self):
        super(LSTM_model, self).__init__()
        self.lstm1 = torch.nn.LSTM(99, 128)
        self.lstm2 = torch.nn.LSTM(128, 64)
        self.dropout = torch.nn.Dropout(p = 0.4)
        self.lin1 = torch.nn.Linear(64*10, 256)
        self.lin2 = torch.nn.Linear(256, 4)
        self.softmax1 = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x.view(len(x), -1))
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.softmax1(x)
        return x

