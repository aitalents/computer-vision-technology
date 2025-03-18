import fiftyone as fo
import fiftyone.zoo as foz
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Загружаем датасет Caltech256
dataset = foz.load_zoo_dataset("caltech256")

# 2. Получаем список id всех сэмплов
dataset_ids = [sample.id for sample in dataset]

# 3. Разделяем на train (80%) и val (20%) с фиксированным seed
train_ids, val_ids = train_test_split(dataset_ids, test_size=0.2, random_state=51)

# 4. Назначаем сплиты каждому сэмплу
for sample in dataset:
    if sample.id in train_ids:
        sample["split"] = "train"
    else:
        sample["split"] = "val"
    sample.save()

# 5. Выводим статистику по количеству сэмплов в каждом сплите
train_count = len(train_ids)
val_count = len(val_ids)
print("Количество сэмплов в train сплите:", train_count)
print("Количество сэмплов в val сплите:", val_count)

# 6. Сохраняем локальные имена файлов для сэмплов из сплита val в CSV
val_samples = dataset.match({"split": "val"})
val_file_names = [os.path.basename(sample.filepath) for sample in val_samples]
df = pd.DataFrame(val_file_names, columns=["filename"])
df.to_csv("val.csv", index=False)
print("Локальные имена файлов из val сплита сохранены в 'val.csv'.")
