# Topic 6. Action Recognition

[Презентация](https://docs.google.com/presentation/d/e/2PACX-1vRVdQiEjyCGt_mS8zyabFRbxTyp3IBrVzm3h0o2PdKr9ns5glrQzxwQzpjYowGTLwCW3oPUMG32zACr/pub?start=false&loop=false&delayms=60000)

[UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)

[MMAction](https://github.com/open-mmlab/mmaction2)

[PyTorchVideo](https://pytorchvideo.org/)

[PyTorch Video Classification Models](https://pytorch.org/vision/0.9/models.html#video-classification)

## Домашнее задание
Скачать из датасета [Kinetics](https://github.com/cvdfoundation/kinetics-dataset) 700-2020 видео с классами содержащими слово dancing

Нельзя использовать веса предобученные на Kinetics!

 1. Обучить модель на отдельных кадрах и провести сравнение - 3 балла
 2. Обучить модель классификации этих видео на основе Pose Estimation - 4 балла
 3. Построить с нуля и обучить модель классификации видео на основе 3D свёрток или трансформеров - 5 баллов
