import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import timm


class TripletDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Инициализация датасета.
        Параметры:
            root (str): Путь к директории с данными (например, 'images/train' или 'images/val').
            transform: Трансформации, применяемые к изображениям.
        """
        self.dataset = datasets.ImageFolder(root=root, transform=transform)
        self.transform = transform

        # Создаем словарь, где для каждого класса хранится список индексов изображений данного класса.
        # Это понадобится для выбора позитивного (positive) примера.
        self.class_to_indices = {}

        for idx, (_, label) in enumerate(self.dataset.imgs):  # self.dataset.imgs содержит пары (путь_к_изображению, метка)
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Возвращает кортеж из четырёх элементов: (anchor, positive, negative, anchor_label).
        Anchor и positive принадлежат одному классу, negative – другому.
        anchor_label используется для расчёта recall@k.
        """
        # Загружаем опорное изображение (anchor) и его метку
        anchor_img, anchor_label = self.dataset[index]

        # Выбираем позитивный пример: случайное изображение того же класса (но не само anchor)
        positive_index = index
        # Если в классе только одно изображение, этот цикл может зациклиться – в реальных задачах стоит добавить проверку
        while positive_index == index:
            positive_index = random.choice(self.class_to_indices[anchor_label])
        positive_img, _ = self.dataset[positive_index]

        # Выбираем негативный пример: изображение из другого класса
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.class_to_indices.keys()))
        negative_index = random.choice(self.class_to_indices[negative_label])
        negative_img, _ = self.dataset[negative_index]

        return anchor_img, positive_img, negative_img, anchor_label


# Модель-эмбеддер, которая использует бэкбон из timm и дополнительный FC слой для получения эмбеддингов нужной размерности
class EmbeddingNet(nn.Module):
    def __init__(self, backbone_name='resnet18', embedding_dim=128, pretrained=True):
        """
        Параметры:
            backbone_name (str): Имя модели из timm (например, 'resnet18').
            embedding_dim (int): Размерность выходного эмбеддинга.
            pretrained (bool): Использовать ли предобученные веса.
        """
        super(EmbeddingNet, self).__init__()
        # Загружаем модель-бэкбон из timm без классификационной головы (num_classes=0)
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        # Получаем размерность выходных признаков от бэкбона
        backbone_features = self.backbone.num_features
        # Полносвязный слой для проекции признаков в пространство эмбеддингов заданной размерности
        self.fc = nn.Linear(backbone_features, embedding_dim)

    def forward(self, x):
        # Получаем признаки от бэкбона
        x = self.backbone(x)
        # Пропускаем признаки через полносвязный слой
        x = self.fc(x)
        # Нормализуем эмбеддинги (L2-норма), что часто улучшает качество в задачах Metric Learning
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


# Функция для одной эпохи обучения
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # Переводим модель в режим обучения
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # Если датасет возвращает 4 элемента, игнорируем четвертый (label)
        if len(batch) == 4:
            anchor, positive, negative, _ = batch
        else:
            anchor, positive, negative = batch

        # Перемещаем данные на устройство (CPU или GPU)
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()  # Обнуляем градиенты

        # Получаем эмбеддинги для всех изображений
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        # Вычисляем тройную потерю (Triplet Margin Loss)
        loss = criterion(anchor_out, positive_out, negative_out)

        loss.backward()  # Обратное распространение ошибки
        optimizer.step()  # Обновляем параметры модели

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.4f}')

    avg_loss = running_loss / len(dataloader)
    return avg_loss


# Функция для валидации (оценка модели без обратного распространения ошибки) по потере
def validate(model, dataloader, criterion, device):
    model.eval()  # Переводим модель в режим оценки
    running_loss = 0.0

    # Отключаем вычисление градиентов для экономии памяти и ускорения вычислений
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                anchor, positive, negative, _ = batch
            else:
                anchor, positive, negative = batch

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


# Функция для валидации по метрике recall@k.
# Для каждого изображения-запроса (anchor) вычисляются эмбеддинги, затем производится поиск top-k ближайших соседей
# (исключая само изображение). Если среди них найден хотя бы один с таким же классом, считается попадание.
def validate_recall_at_k(model, dataloader, k, device):
    model.eval()
    embeddings_list = []
    labels_list = []

    # Собираем эмбеддинги и метки для всех изображений из валидационного набора
    with torch.no_grad():
        for batch in dataloader:
            # Из батча используем только anchor и его метку
            if len(batch) == 4:
                anchor, _, _, labels = batch
            else:
                # Если меток нет, валидация по recall@k невозможна
                continue
            anchor = anchor.to(device)
            emb = model(anchor)
            embeddings_list.append(emb)
            labels_list.append(labels.to(device))

    # Объединяем батчи в один тензор
    embeddings_all = torch.cat(embeddings_list, dim=0)  # размер (N, embedding_dim)
    labels_all = torch.cat(labels_list, dim=0)            # размер (N,)

    # Вычисляем попарные евклидовы расстояния
    distances = torch.cdist(embeddings_all, embeddings_all, p=2)  # размер (N, N)
    # Сортируем расстояния по возрастанию
    sorted_indices = torch.argsort(distances, dim=1)

    hits = 0
    N = embeddings_all.size(0)
    for i in range(N):
        # Первый сосед – это само изображение, поэтому берем следующие k соседей
        neighbors = sorted_indices[i, 1:k+1]
        # Если среди top-k соседей есть хотя бы один с той же меткой, считаем попадание
        if (labels_all[neighbors] == labels_all[i]).any():
            hits += 1

    recall_at_k = hits / N
    return recall_at_k


# Основная функция, где происходит настройка данных, модели и запуск обучения
def main():
    # Определяем устройство: GPU, если доступен, иначе CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Используем устройство: {device}')

    # Определяем трансформации для изображений (изменение размера, преобразование в тензор, нормализация)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Нормировка с использованием средних и стандартных отклонений ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Пути к директориям с обучающими и валидационными данными
    train_dir = 'images/train'
    val_dir = 'images/val'

    # Создаем датасеты для обучения и валидации
    train_dataset = TripletDataset(root=train_dir, transform=transform)
    val_dataset = TripletDataset(root=val_dir, transform=transform)

    # Создаем DataLoader'ы для удобного перебора данных
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Создаем модель-эмбеддер с выбранным бэкбоном
    model = EmbeddingNet(backbone_name='resnet18', embedding_dim=128, pretrained=True)
    model.to(device)

    # Определяем оптимизатор (Adam) и функцию потерь (Triplet Margin Loss)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    num_epochs = 10  # Задаем количество эпох обучения
    k = 5          # Параметр для recall@k

    for epoch in range(num_epochs):
        print(f'Эпоха {epoch + 1}/{num_epochs}')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        recall_at_k = validate_recall_at_k(model, val_loader, k, device)
        print(f'Средняя потеря на обучении: {train_loss:.4f}, на валидации: {val_loss:.4f}, Recall@{k}: {recall_at_k:.4f}')

        # Сохраняем модель после каждой эпохи (можно сохранять только при улучшении валидационной метрики)
        os.makedirs('train_1', exist_ok=True)
        torch.save(model.state_dict(), f'train_1/model_epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    main()
