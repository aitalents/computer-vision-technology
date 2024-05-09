import cv2
import glob 
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
import torch
import torchvision
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
torch.manual_seed(0)

labels = glob.glob('DATA/train/*')
labels = [el[11:] for el in labels]

try:
    device = torch.device("cuda")
except:
    device = torch.device("cpu")

test_ds = CustomVideoDataset(labels, train_flag=False)
train_ds = CustomVideoDataset(labels)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32,
                                          shuffle=False)

resnet = torchvision.models.video.r2plus1d_18(pretrained=False, progress=True)

resnet_model = Resnet_model(resnet)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.005)

#Train loop
print("Start training")
loss_hist = []
val_loss_hist = []
patience = 0
resnet_model.to(device)

for epoch in range(8):
    resnet_model.train(True)
    running_loss = 0.0
    iters = 0
    train_acc = 0
    train_f1 = 0
    print('_____________________')
    print('EPOCH: ', epoch+1)
    for i, (img, label) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        res = resnet_model(img.float().moveaxis(2, 1).to(device))
        loss = criterion(res, label.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        iters += 1
        train_acc += accuracy_score(label, res.cpu().argmax(dim=1))
        train_f1 += f1_score(label, res.cpu().argmax(dim=1), average='weighted')
    loss_hist.append(running_loss/iters)
    train_acc = train_acc/iters
    train_f1 = train_f1/iters

    #show results per epoch
    print('train_loss = ', running_loss/iters)
    print(f'Train: accuracy = {train_acc}, F1 = {train_f1}')

    #validating
    running_loss = 0.0
    iters = 0
    test_acc = 0
    test_f1 = 0
    resnet_model.eval()
    with torch.no_grad():
        for i ,(test_img, test_label) in enumerate(test_loader, 0):
            val_res = resnet_model(test_img.float().moveaxis(2, 1).to(device))
            val_loss = criterion(val_res, test_label.to(device))
            running_loss += val_loss.item()
            iters += 1
            test_acc += accuracy_score(test_label, val_res.cpu().argmax(dim=1))
            test_f1 += f1_score(test_label, val_res.cpu().argmax(dim=1), average='weighted')
        val_loss_hist.append(running_loss/iters)
        test_acc = test_acc/iters
        test_f1 = test_f1/iters

    #show val results per epoch
    print('val_loss = ', running_loss/iters)
    print(f'Test: accuracy = {test_acc}, F1 = {test_f1}')
    print('\n')

    #early stop
    if epoch > 3:
        if val_loss_hist[-1] > val_loss_hist[-2]:
            patience += 1
        else:
            patience = 0
    if patience == 4:
        break
        #save best model
    if epoch == 0:
        best_f1 = test_f1
    else:
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(resnet_model, 'models/resnet_model_best.pth')
print('Finish! Num epochs: ', epoch+1)