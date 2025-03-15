import os
import shutil
from sklearn.model_selection import train_test_split
import torchvision


def prepare_oxford_pet_dataset():
    # Путь для загрузки датасета Oxford-IIIT Pet
    dataset_root = './data/oxford_pet'
    
    # Загружаем датасет Oxford-IIIT Pet.
    # Параметр target_types='category' означает, что в качестве меток будут использоваться категориальные индексы.
    dataset = torchvision.datasets.OxfordIIITPet(root=dataset_root, download=True, target_types='category')
    
    # Получаем список путей к изображениям и их меток.
    # Обратите внимание, что атрибуты _images и _labels являются внутренними, но для демонстрационных целей их можно использовать.
    image_paths = dataset._images    # Список полных путей к изображениям
    labels = dataset._labels         # Список меток (числовых индексов)
    
    # Получаем список имен классов (например, ['Abyssinian', 'American Shorthair', ...]).
    classes = dataset.classes
    
    # Создаем список индексов для всех изображений
    indices = list(range(len(image_paths)))
    # Разбиваем индексы на обучающую (80%) и валидационную (20%) выборки с учетом стратификации по классам
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    
    # Определяем корневую директорию для нового датасета в нужной структуре
    dest_root = './images'
    train_dest = os.path.join(dest_root, 'train')
    val_dest = os.path.join(dest_root, 'val')
    
    # Вспомогательная функция для копирования изображений в соответствующую директорию
    def copy_images(indices, split_dest):
        for idx in indices:
            src_path = image_paths[idx]
            # Получаем имя класса по метке (например, 'Bengal' или 'Siamese')
            class_name = classes[labels[idx]]
            # Определяем конечную директорию для данного класса
            class_dir = os.path.join(split_dest, class_name)
            os.makedirs(class_dir, exist_ok=True)
            # Копируем изображение в соответствующий каталог
            shutil.copy(src_path, class_dir)
    
    print("Копирование обучающих изображений...")
    copy_images(train_idx, train_dest)
    
    print("Копирование валидационных изображений...")
    copy_images(val_idx, val_dest)
    
    print("Датасет Oxford-IIIT Pet успешно подготовлен!")
    print(f"Структура данных расположена в папке: {dest_root}")


if __name__ == '__main__':
    prepare_oxford_pet_dataset()
