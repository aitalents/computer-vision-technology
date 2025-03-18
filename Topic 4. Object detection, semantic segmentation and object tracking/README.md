# Object Detection, Semantic Segmentation and Tracking

[Презентация по теме Semantic Segmentation и Object Detection](https://docs.google.com/presentation/d/1ZIwYsRG-zBn1LNJ3-zRzubozFyFG1Pi8uy5GwXcZOas/edit?usp=sharing)

[Презентация по теме Tracking](https://docs.google.com/presentation/d/1DACS7VJl-uKhhfsHwEdzsxJpVZkEGAyjT9ksXgIbu4M/edit?usp=sharing)

## Домашнее задание по теме Object Detection

Написать реализацию YOLO([ https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640) ) с нуля для распознавания свиней и людей.

Данные -[ https://disk.yandex.ru/d/qXFgvtO3y-ey_A](https://disk.yandex.ru/d/qXFgvtO3y-ey_A)

1. Подготовить датасет для обучения - 2 балла
2. Реализовать архитектуру YOLO и методы для её обучения - 10 баллов
3. Реализовать NMS - 2 балла
4. Реализовать метрику mAP - 2 балла
5. Подобрать оптимальные гиперпараметры - 2 балла
6. Залогировать результаты экспериментов (метрики, гиперпараметры, визуализации) - 2 балла

Для защиты домашнего задания нужно предоставить исходный код решения и отчет обо всей проделанной работе

## Домашнее задание по теме Tracking

Реализовать методы tracker_soft, tracker_strong и метод подсчета метрик для трекинга country balls в песочнице https://github.com/thegoldenbeetle/object-tracking-assignment/. Сравните результаты tracker_soft и tracker_strong для разного количества объектов и различных значений параметров среды. Подробности в [Readme](https://github.com/thegoldenbeetle/object-tracking-assignment/blob/main/README.md).		. 

Критерии оценки дз:

1. Реализован метод tracker_soft - 10 баллов
2. Реализован метод tracker_strong - 5 баллов
3. Реализован способ подсчета метрик - 5 баллов

Для защиты домашнего задания нужно предоставить исходный код решения и отчет обо всей проделанной работе. В отчете необходимо в свободном стиле привести описание реализованных алгоритмов трекинга и метода оценки качества трекеров, привести сравнительную таблицу реализованных трекеров, включая результаты проделанных экспериментов, сделать вывод.