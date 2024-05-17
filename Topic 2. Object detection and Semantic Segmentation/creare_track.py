import numpy as np
import random
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# количество объектов
tracks_amount = 20
# на сколько пикселей рамка объектов может ложно смещаться (эмуляция не идеальной детекции)
random_range = 20
# с какой вероятностью объект на фрейме может быть не найдет детектором
bb_skip_percent = 0.25

width = 1000
height = 800
tracks = []
i = 0
cb_width = 120
cb_height = 100

def get_point_on_random_side(width, height):
    side = random.randint(0, 4)
    if side == 0:
        x = random.randint(0, width)
        y = 0
    elif side == 1:
        x = random.randint(0, width)
        y = height
    elif side == 2:
        x = 0
        y = random.randint(0, height)
    else:
        x = width
        y = random.randint(0, height)
    return x, y

def fun(x, a, b, c, d):
    return a * x + b * x ** 2 + c * x ** 3 + d

def objective(x, a, b, c, d, e, f):
    return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + f

def check_track(track):
    if all(el['x'] == track[0]['x'] for el in track):
        return False
    if all(el['y'] == track[0]['y'] for el in track):
        return False
    if not all(el['x'] >= 0 and el['x'] <= width for el in track):
        return False
    if not all(el['y'] >= 0 and el['y'] <= height for el in track):
        return False
    if (2 > track[0]['x'] > (width - 2) and 2 > track[0]['y'] > (width - 2)) or (2 > track[-1]['x'] > (width - 2) and 2 > track[-1]['y'] > (width - 2)):
        return False
    return True

def add_track_to_tracks(track, tracks, id):
    for i, p in enumerate(track):
        # a chance that detector missed object
        if random.random() < bb_skip_percent:
            bounding_box = []
        else:
            bounding_box = [
                              p['x'] - int(cb_width/2) + random.randint(-random_range, random_range),
                              p['y'] - cb_height + random.randint(-random_range, random_range),
                              p['x'] + int(cb_width/2) + random.randint(-random_range, random_range),
                              p['y'] + random.randint(-random_range, random_range)
                            ]
        if i < len(tracks):
            tracks[i]['data'].append({'cb_id': id, 'bounding_box': bounding_box,
                                      'x': p['x'], 'y': p['y'], 'track_id': None})
        else:
            tracks.append(
                {
                    'frame_id': len(tracks)+1,
                    'data': [{'cb_id': id, 'bounding_box': bounding_box,
                              'x': p['x'], 'y': p['y'], 'track_id': None}]
                }
            )
    return tracks

while i < tracks_amount:
    x, y = np.array([]), np.array([])
    p = get_point_on_random_side(width, height)
    x = np.append(x, p[0])
    y = np.append(y, p[1])
    x = np.append(x, random.randint(200, width - 200))
    y = np.append(y, random.randint(200, height - 200))
    x = np.append(x, random.randint(200, width - 200))
    y = np.append(y, random.randint(200, height - 200))
    p = get_point_on_random_side(width, height)
    x = np.append(x, p[0])
    y = np.append(y, p[1])
    num = random.randint(20, 50)

    coef, _ = curve_fit(fun, x, y)
    track = [{'x': int(x), 'y': int(y)} for x, y in
             zip(np.linspace(x[0], x[-1], num=num), fun(np.linspace(x[0], x[-1], num=num), *coef))]

    if check_track(track):
        plt.plot(x, y, 'o', label='Original points')
        plt.plot(np.linspace(x[0], x[-1]), fun(np.linspace(x[0], x[-1]), *coef), '*-')
        tracks = add_track_to_tracks(track, tracks, i)
        i += 1
#plt.show()

print(f'country_balls_amount = {tracks_amount}')
print(f'track_data = {tracks}')
