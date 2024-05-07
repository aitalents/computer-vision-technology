from fastapi import FastAPI, WebSocket
from track_3 import track_data, country_balls_amount
import asyncio
import glob
from transformers import AutoModel
from tracker_soft import Tracker as Hungarian
from tracker_strong import Tracker as DeepSORT
from eval import id_f1

app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
frames = glob.glob('screens/cb20/*')
model_ckpt = "nateraw/vit-base-beans"
model = AutoModel.from_pretrained(model_ckpt)
print('Started')


def tracker_soft(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """
    tracker = Hungarian(160, 30, 5, 0)
    detections = [
        {
            'center': (obj['x'], int(obj['y'] - obj['bounding_box'][1] / 2))
        } for obj in el['data'] if len(obj['bounding_box']) > 0
    ]
    tracker.Update(detections)
    for i in range(len(tracker.tracks)):
        el['data'][i]['track_id'] = int(tracker.tracks[i].track_id)
    return el


def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true для первого прогона, на повторном прогоне можете читать фреймы из папки
    и по координатам вырезать необходимые регионы.
    """
    tracker = DeepSORT(160, 30, 5, 0, model)
    detections = [
        {
            'img': frames[el['frame_id'] - 1],
            'bbox': obj['bounding_box'],
            'center': (obj['x'], int(obj['y'] - obj['bounding_box'][1] / 2))
        } for obj in el['data'] if len(obj['bounding_box']) > 0
    ]
    tracker.Update(detections)
    for i in range(len(tracker.tracks)):
        el['data'][i]['track_id'] = int(tracker.tracks[i].track_id)
    return el


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    await websocket.send_text(str(country_balls))
    preds = []
    for el in track_data:
        await asyncio.sleep(0.5)
        # TODO: part 1
        # el = tracker_soft(el)
        # TODO: part 2
        preds.append(el)
        el = tracker_strong(el)
        # отправка информации по фрейму
        await websocket.send_json(el)
    print(id_f1(preds, track_data))
    print('Bye..')
