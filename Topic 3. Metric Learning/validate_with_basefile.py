import torch
import timm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


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


def compute_base_embeddings(model, dataloader, device):
    """
    Формирует бейз-файл — усреднённые эмбеддинги для каждого класса.
    
    Параметры:
        model: обученная модель-эмбеддер.
        dataloader: DataLoader для обучающего датасета.
        device: устройство (CPU/GPU).

    Возвращает:
        base_embeddings: словарь, где ключ — метка класса, значение — усреднённый эмбеддинг (тензор).
    """
    model.eval()
    # Словари для накопления суммарных эмбеддингов и количества примеров для каждого класса
    sums = {}
    counts = {}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model(images)  # размер (batch_size, embedding_dim)
            labels = labels.to(device)
            for emb, label in zip(embeddings, labels):
                label = label.item()
                if label not in sums:
                    sums[label] = emb.clone()
                    counts[label] = 1
                else:
                    sums[label] += emb
                    counts[label] += 1

    base_embeddings = {}
    for label in sums:
        # Усредняем эмбеддинги для каждого класса
        base_embeddings[label] = sums[label] / counts[label]

    return base_embeddings


def validate_classification(model, base_embeddings, dataloader, device):
    """
    Выполняет классификацию на валидационном датасете, используя бейз-файл.
    Для каждого изображения выбирается класс с ближайшим по евклидову расстоянию эмбеддингом.
    
    Параметры:
        model: обученная модель-эмбеддер.
        base_embeddings: словарь с усредненными эмбеддингами для каждого класса.
        dataloader: DataLoader для валидационного датасета.
        device: устройство (CPU/GPU).
        
    Возвращает:
        accuracy: метрика точности классификации.
    """
    model.eval()
    total = 0
    correct = 0

    # Преобразуем бейз-файл в два списка: метки и эмбеддинги
    base_labels = []
    base_embs = []
    for label, emb in base_embeddings.items():
        base_labels.append(label)
        base_embs.append(emb.unsqueeze(0))  # добавляем размерность для конкатенации
    base_embs = torch.cat(base_embs, dim=0)  # размер (num_classes, embedding_dim)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images)  # размер (batch_size, embedding_dim)
            # Вычисляем евклидовы расстояния до бейз-эмбеддингов
            # Расстояния: (batch_size, num_classes)
            dists = torch.cdist(embeddings, base_embs, p=2)
            # Выбираем ближайший класс для каждого изображения
            preds = torch.argmin(dists, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = correct / total
    return accuracy


def main():
    # Параметры и пути
    train_dir = 'images/train'
    val_dir = 'images/val'
    model_path = 'train_2/model_epoch_2.pth'  # путь к сохранённой модели
    backbone_name = 'resnet18'
    embedding_dim = 128
    batch_size = 32

    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # Трансформации (аналогично обучению)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Загружаем обучающий датасет (ImageFolder структура)
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Загружаем валидационный датасет
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Создаём модель и загружаем веса
    model = EmbeddingNet(backbone_name=backbone_name, embedding_dim=embedding_dim, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Формирование бейз-файла: усреднённые эмбеддинги для каждого класса по обучающему датасету
    print("Формирование бейз-эмбеддингов для каждого класса...")
    base_embeddings = compute_base_embeddings(model, train_loader, device)

    # Производим классификацию валидационного датасета
    print("Валидация модели по задаче классификации...")
    accuracy = validate_classification(model, base_embeddings, val_loader, device)

    print(f"Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()
