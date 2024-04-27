import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import fiftyone.zoo as foz

from model import YoloV1, YoloLoss, FiftyOneTorchDataset
from utils import mean_average_precision, get_bboxes, save_checkpoint, load_checkpoint, MyTransform

# обучим на 5 классах, больше ресурсов не хватает
classes_list = ["traffic light", "stop sign", "car", "bus", "truck"]

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"]
    , classes=classes_list
    # , max_samples=500,
)

classes = dataset.distinct(
    "ground_truth.detections.label"
)
id2name, name2id = {}, {}
classes_id_list = []
for class_id, class_name in enumerate(classes):
    if class_name in classes_list:
    # if True:
      classes_id_list.append(class_id)
    id2name[class_id] = class_name
    name2id[class_name] = class_id

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # 64 in original paper but resource exhausted error otherwise.
WEIGHT_DECAY = 0
EPOCHS = 20
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "model.pth"

torch_dataset = FiftyOneTorchDataset(dataset, transform=MyTransform(resize=(448, 448)))
train_dataset, test_dataset = train_test_split(torch_dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Train loop
def main():
    model = YoloV1(split_size=7, num_boxes=2, num_classes=5).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3, mode='max', verbose=True)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # Списки для хранения метрик
    losses = []
    map_scores = []

    for epoch in range(EPOCHS):
        mean_loss = train_fn(train_loader, model, optimizer, loss_fn)
        losses.append(mean_loss)

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE # on 'cpu' try
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        map_scores.append(mean_avg_prec)
        print(f"Epoch {epoch}/{EPOCHS}, Train mAP: {mean_avg_prec:.8f}, Mean loss: {mean_loss:.4f}")

        scheduler.step(mean_avg_prec)

    checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)

    # Построение графиков
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Mean Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(map_scores, label='mAP Score')
    plt.title('mAP Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mAP Score')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()