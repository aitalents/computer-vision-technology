import torch
import torch.nn as nn

# Установка случайного зерна для воспроизводимости результатов
torch.manual_seed(123)

# Конфиг для модели
# Кортежи - слои
# Maxpool - он самый :)
LAYER_CONFIG = [
    (7, 64, 2, 3),
    "MAXPOOL",
    (3, 192, 1, 1),
    "MAXPOOL",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "MAXPOOL",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "MAXPOOL",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class ConvolutionalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        # Создание слоя свёртки без bias'а
        self.convolution = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        # Добавление слоя пакетной нормализации
        self.batch_normalization = nn.BatchNorm2d(output_channels)
        # Добавление функции активации LeakyReLU
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, input_tensor):
        x = self.convolution(input_tensor)
        x = self.batch_normalization(x)
        return self.activation(x)
    
class YOLOv1(nn.Module):
    def __init__(self, input_channels=3, **kwargs):
        super().__init__()
        self.input_channels = input_channels
        # Создание свёрточных слоёв сети
        self.conv_layers = self.create_conv_layers(LAYER_CONFIG)
        # Создание полносвязных слоёв сети
        self.fully_connected_layers = self.create_fully_connected(**kwargs)
        
    def forward(self, input_tensor):
        x = self.conv_layers(input_tensor)
        x = torch.flatten(x, start_dim=1)
        return self.fully_connected_layers(x)
    
    def create_conv_layers(self, architecture):
        layers = []
        channels = self.input_channels
        
        for element in architecture:
            if isinstance(element, tuple):
                # Добавление блока свёртки
                layers.append(ConvolutionalBlock(channels, element[1], kernel_size=element[0], stride=element[2], padding=element[3]))
                channels = element[1]
            elif element == "MAXPOOL":
                # Добавление слоя максимального пулинга
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif isinstance(element, list):
                # Добавление повторяющихся блоков свёртки
                for _ in range(element[2]):
                    layers.append(ConvolutionalBlock(channels, element[0][1], kernel_size=element[0][0], stride=element[0][2], padding=element[0][3]))
                    layers.append(ConvolutionalBlock(element[0][1], element[1][1], kernel_size=element[1][0], stride=element[1][2], padding=element[1][3]))
                    channels = element[1][1]
                    
        return nn.Sequential(*layers)
    
    def create_fully_connected(self, split_size, num_boxes, num_classes):
        # Расчёт выходного размера для полносвязных слоёв
        output_size = split_size * split_size * (num_classes + num_boxes * 5)
        return nn.Sequential(
            nn.Linear(1024 * split_size * split_size, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, output_size)
        )