# Курс Технологии компьютерного зрения

# ДЗ 3
## Exp 1
Обучим сверточную нейронную сеть для классификации по отдельным кадрам видео

<img width="689" alt="image" src="https://github.com/SvyatoslavMilovidov/computer-vision-technology/assets/92350053/c55ec7a0-82d8-45e7-b698-7ad6239c4c0c">


Таким образом удалось достичь метрик: 
Test: accuracy = 0.6146669130067568, F1 = 0.6117995443460864

Затем используем Feature extractor модели для получения эмбеддингов.
Подготовим LSTM модель, получающую на вход последовательность эмбеддингов с кадров видео.

Обучим модель
<img width="581" alt="image" src="https://github.com/SvyatoslavMilovidov/computer-vision-technology/assets/92350053/d3c2cf1e-7e12-4085-baa1-9fb3790c98a2">

Полученные метрики:
 <img width="449" alt="image" src="https://github.com/SvyatoslavMilovidov/computer-vision-technology/assets/92350053/8124f7dd-5938-4673-9180-56cf4d81b0ba">

 ## Exp 2
Обучим LSTM модель, получающую на вход ключевые точки. Точки будут получены с помощью mediapipe.

<img width="576" alt="image" src="https://github.com/SvyatoslavMilovidov/computer-vision-technology/assets/92350053/7a12a852-db92-4f39-a02f-1c8f4e872720">

Полученные метрики:
Test: accuracy = 0.5824742268041238, F1 = 0.5793110610931782

 ## Exp 3
 Обучим Resnet (2+1)D.

 Для обучения модели использовался google colab, но к сожалению, из-за размеров модели не удалось произвести полное обучение. Русурсов Colab хватило тольок на 3 полных эпохи.

 <img width="553" alt="image" src="https://github.com/SvyatoslavMilovidov/computer-vision-technology/assets/92350053/310f0057-3afa-466a-9f65-3d84b3c08dc5">




