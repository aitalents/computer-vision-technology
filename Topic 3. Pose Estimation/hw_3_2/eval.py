import json
from scripts.sequence_data import test_dataset
from config import BATCH_SIZE
from models import LSTMModel
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, recall_score, precision_score


test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = LSTMModel().to(device)
model.load_state_dict(torch.load("LSTM.pt"))


def collect_predictions(model, test_dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in test_dataloader:
            out = model(x.to(device))
            _, predicted = torch.max(out, 1)

            y_true.extend(y.numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
    return y_true, y_pred


def calculate_classification_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    metrics_dict = {
        'precision': precision,
        'recall': recall,
        'f1-score': f1
    }
    return metrics_dict


y_true, y_pred = collect_predictions(model, test_dataloader, device)
metrics = calculate_classification_metrics(y_true, y_pred)
metrics_json = json.dumps(metrics, indent=4)

# Save to file
file_path = "lstm_metrics.json"
with open(file_path, "w") as file:
    file.write(metrics_json)
