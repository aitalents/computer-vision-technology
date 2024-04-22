from track_3 import track_data, country_balls_amount
from PIL import Image
import glob

imgs = glob.glob('imgs/*')
country_balls = [imgs[x % len(imgs)] for x in range(country_balls_amount)]

def make_screenshot():
    cb_width = 120
    cb_height = 100
    width = 1000
    height = 800
    for frame in track_data:
        canvas = Image.new('RGB', (width, height), 'white')
        for cb in frame['data']:
            img = Image.open(country_balls[cb['cb_id']])
            img = img.convert('RGB')
            img = img.resize((cb_width, cb_height))
            canvas.paste(img, (int(cb['x'] - cb_width / 2), int(cb['y'] - cb_height)))
        fr_id = frame['frame_id']
        canvas.save(f'screens/cb20/frame{fr_id}.png')

make_screenshot()