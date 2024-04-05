# Topic 1. One Stage Object detection

[Запись занятия]()

Команде студентов необходимо реализовать и обучить модель YOLO первой версии, используя фреймворк pytorch, на датасете [COCO-2017](https://docs.voxel51.com/user_guide/dataset_zoo/datasets.html#dataset-zoo-coco-2017)

Критерии оценки:
- Обучение модели 3 балла
- Добавить самописную реализацию NMS и якорей(Anchors) 4 балла
- Добавить реализацию метрики mAP 5 баллов

Решение необходимо продоставить в формате Pull Request в данную ветку, код рещения должен быть написан в скриптах, без использования Jupyter.
Зависимости зафиксированы в файле requirements.txt.

### Оформление Pull Request
название PR доожно быть в виде: "team #номер или название команды"
в описании PR указать:
- состав команды: фамилии имена участников команды
- что из критериев сделали
- метрики получившейся модели на сплите `validation`

# Скачивание датасета

Создадим папку для хранения датасета MS COCO 2017 для детекции:
```
mkdir detection_coco2017
cd mkdir detection_coco2017
```

Скачаем нужные нам файлы на сервер с помощью `wget` и `txum`.
Откроем в терминале сессию `tmux`:
```
tmux
```

## Скачаем необходимые файлы:
```
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
```

## Разархивируем данные:
```
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

Приводим разархивированный датасет к виду:
```
dataset
├── train
│   ├── data
│   └── labels.json
└── validation
    ├── data
    └── labels.json
```



