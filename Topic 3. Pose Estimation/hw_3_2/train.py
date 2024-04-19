#!/usr/bin/env python

from scripts.sequence_data import train_dataset, valid_dataset
from torch.utils.data import DataLoader
import torch
import os
import torch.nn as nn
import torch.optim as optim
from config import BATCH_SIZE, SEED, EPOCHS
from models import LSTMModel
# from torch.optim.lr_scheduler import CosineAnnealingLR
import clearml


def seed_everything(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = LSTMModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss = nn.CrossEntropyLoss()


def train(model, loss_func, device, train_loader, optimizer):
    model.train()
    loss_value = 0
    correct_predictions = 0
    total_samples = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_func(out, y.to(torch.long))
        loss_value += loss.item()
        _, predicted = torch.max(out, 1)
        correct_predictions += (predicted == y).sum().item()
        loss.backward()
        optimizer.step()
        total_samples += len(x)
    avg_loss = loss_value / len(train_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def test(model, loss_func, device, valid_loader):
    model.eval()
    loss_value = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_func(out, y.to(torch.long))
            loss_value += loss.item()
            _, predicted = torch.max(out, 1)
            correct_predictions += (predicted == y).sum().item()
            total_samples += len(x)

    avg_loss = loss_value / len(valid_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


task = clearml.Task.init(
    project_name="Action recognition",
    task_name="LSTM training",
    output_uri=True
)


logger = clearml.Logger.current_logger()
best_val_metric = float("-inf")
counter = 0
num_iter_early_stop = 15


for epoch in range(EPOCHS):
    train_loss, train_accuracy = train(model, loss, device, train_dataloader, optimizer)
    valid_loss, valid_accuracy = test(model, loss, device, valid_dataloader)

    task.logger.report_scalar("Loss", "train_loss", iteration=epoch, value=train_loss)
    task.logger.report_scalar("Loss", "valid_loss", iteration=epoch, value=valid_loss)

    task.logger.report_scalar("Metric", "train_accuracy", iteration=epoch, value=train_accuracy)
    task.logger.report_scalar("Metric", "valid_accuracy", iteration=epoch, value=valid_accuracy)

    if valid_accuracy > best_val_metric:
        best_val_metric = valid_accuracy
        counter = 0
    else:
        counter += 1
        if counter >= num_iter_early_stop:
            logger.report_text(f"Early stopping at epoch {epoch}")
            break

model.eval()
torch.save(model.state_dict(), "LSTM.pt")
