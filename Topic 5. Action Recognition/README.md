![Снимок экрана 2024-04-26 084323](https://github.com/Sergey-Kit/computer-vision-technology/assets/82327055/17fcfa60-1b05-4a54-a48f-088d41c63f1d)# Topic 5. Action Recognition
[Лекция](https://disk.yandex.ru/d/JHwZXroTqgW-HA/%D0%A2%D0%B5%D1%85%D0%BD%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D0%B8%20CV/2023_05_24_17_14_%D0%A2%D0%B5%D1%85%D0%BD%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D0%B8_CV%2C_%D0%98%D0%9C%D0%9E%2C_%D0%B2%D0%B5%D1%81%D0%BD%D0%B0_2023.mp4)

## Домашнее задание
Скачать из датасета Kinetics 700-2020 видео с классами содержащими слово dancing
https://www.deepmind.com/open-source/kinetics

В репозитории представлены скрипты: 
 download_data.py - для скачивания видео
 videos_to_frames.py - для создания фреймов из видео (из 15 гб видео получилось 80 гб фреймов)
 action-recognition.ipynb - ноутбук с классами моделей и экспериментами
 
Всего 8 классов для различных видов танцев:

 0) - belly dancing
 1) - breakdancing
 2 - country line dancing
 3 - dancing ballet
 4 - dancing charleston
 5 - dancing gangnam style
 6 - dancing macarena
 7 - jumpstyle dancing

 1. Пострена с нуля и обучена модель классификации танцев на видео на основе 3D-свёрток

 ![image](https://github.com/Sergey-Kit/computer-vision-technology/assets/82327055/633c405d-5c5e-4886-a8db-b2b5043ff798)

Обучение целиком занимает меного времени, поэтому в отчете вариант обучения на ~25 минут на 2 эпохах на 200 видео. Всего 8 классов. Обучение на rtx3090

![Снимок экрана 2024-04-26 084323](https://github.com/Sergey-Kit/computer-vision-technology/assets/82327055/818e0e00-6a53-4eaa-b656-db33aaed24ec)
 
 2. Обучена модель классификации этих видео на основе Pose Estimation 

Пайплайн для предобработки с помощью MediaPipe написан. Далее используется сеть с LSTM-блоками.
Однако класс всегда определяется один. 

 3. Представлено применение предобученной модели mvit_base_16x4:

![image](https://github.com/Sergey-Kit/computer-vision-technology/assets/82327055/ee434960-e2a9-43d5-abb1-26cc7a815375)

В целом видно, что применение готовых моделей ускоряет процесс валидации. Самописная модель основе 3D-свёрток работает хорошо, а модель с двумя ступенями работает долго в связи с неоптимизированной обработкой видео и подачей для обучения второй модели. 