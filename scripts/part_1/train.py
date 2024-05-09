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

train_vid, train_lab = prepare_dataset(labels, train = True)
test_vid, test_lab = prepare_dataset(labels, train = False)

try:
    device = torch.device("cuda")
except:
    device = torch.device("cpu")

ed_test_lab = []
for el in test_lab:
    ed_test_lab.append([el for _ in range(10)])
ed_test_lab = np.asarray(ed_test_lab, dtype=np.float32).reshape(194*10)

ed_train_lab = []
for el in train_lab:
    ed_train_lab.append([el for _ in range(10)])
ed_train_lab = np.asarray(ed_train_lab, dtype=np.float32).reshape(len(ed_train_lab)*10)

dataset_train = CustomImageDataset(train_vid.reshape(len(train_vid)*10, 64, 64, 3), ed_train_lab)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=256, 
                                           shuffle=True)

dataset_train = CustomImageDataset(train_vid.reshape(len(train_vid)*10, 64, 64, 3), ed_train_lab)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=256, 
                                           shuffle=True)

model_base = VGG16_bn()
model_base.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_base.parameters(), lr=0.0005)

#Train loop
print("Start training")
loss_hist = []
val_loss_hist = []
patience = 0

for epoch in range(8):
    model_base.train(True)
    running_loss = 0.0
    iters = 0
    train_acc = 0
    train_f1 = 0
    print('_____________________')
    print('EPOCH: ', epoch+1)
    for i, (img, label) in enumerate(tqdm(train_loader)):
        img = batch_augmentation(img)
        img = img.moveaxis(-1, 1)/255
        optimizer.zero_grad()

        res = model_base.forward(img.to(device))
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
    model_base.eval()
    with torch.no_grad():
        for i ,(test_img, test_label) in enumerate(test_loader, 0):
            test_img = test_img.moveaxis(-1, 1)/255
            val_res = model_base.forward(test_img.to(device))
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
print('Finish! Num epochs: ', epoch+1)

torch.save(model_base, '../../models/base_model.pth')

test_ds = CustomVideoDataset(labels, train_flag=False)
train_ds = CustomVideoDataset(labels)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, 
                                          shuffle=False)
                
model_base = VGG16_bn()
model_base = torch.load('../../models/base_model.pth')

embs_preparing = model_base.vgg16_backbone.float()

GRU_model = GRU_embs()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(GRU_model.parameters(), lr=0.0005)

#Train loop
print("Start training")
loss_hist = []
val_loss_hist = []
patience = 0
GRU_model.to(device)
embs_preparing.to(device)

for epoch in range(15):
    GRU_model.train(True)
    running_loss = 0.0
    iters = 0
    train_acc = 0
    train_f1 = 0
    print('_____________________')
    print('EPOCH: ', epoch+1)
    for i, (imgs, label) in enumerate(tqdm(train_loader)):
        embs = []
        with torch.no_grad():
            for i in range(len(imgs)):
                embs.append(embs_preparing(imgs[i].float().to(device)).reshape(10, -1))
            embs = torch.stack(embs)

        optimizer.zero_grad()

        res = GRU_model(embs)
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
    GRU_model.eval()
    with torch.no_grad():
        for i ,(test_imgs, test_label) in enumerate(test_loader, 0):
            embs = []
            for i in range(len(test_imgs)):
                embs.append(embs_preparing(test_imgs[i].float().to(device)).reshape(10, -1))
            embs = torch.stack(embs)
            
            val_res = GRU_model.forward(embs)
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
            torch.save(GRU_model, 'models/GRU_model_best.pth')
print('Finish! Num epochs: ', epoch+1)