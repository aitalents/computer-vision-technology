from matplotlib import pyplot as plt
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np
from glob import glob
import pickle


model = YOLO('yolov8n-pose.pt')


def check_intersection(item: np.ndarray, image_shape: tuple[int]):

    height, width, _ = image_shape
    x, y = width // 2, height // 2
    eps_x = width // 7
    eps_y = height // 7

    coord_x, coord_y = item[:, 0], item[:, 1]

    mask_x = coord_x != 0
    mask_y = coord_y != 0

    coord_x = coord_x[mask_x]
    coord_y = coord_y[mask_y]

    if len(coord_x) == 0 or len(coord_y) == 0:
        return False

    x_min, x_max, y_min, y_max = np.min(coord_x), np.max(coord_x), np.min(coord_y), np.max(coord_y)

    bool_x = x_min - eps_x <= x <= x_max + eps_x
    bool_y = y_min - eps_y <= y <= y_max + eps_y

    return all([bool_x, bool_y])


def save_frames(array: np.ndarray, video_id: str) -> None:
    os.makedirs(f"data/frames/{video_id}", exist_ok=True)
    for i in range(len(array)):
        tensor_np = array[i]
        image = Image.fromarray(tensor_np)
        image.save(f"data/frames/{video_id}/{i}.png")


def sort_path(frames: list[str]) -> dict[int, str]:
    frames = [(item, int(os.path.basename(item).replace(".png", ""))) for item in frames]
    frames = sorted(frames, key=lambda x: x[1])
    frames = {item[1]: item[0] for item in frames}
    return frames


def get_keypoints(video_id: str) -> None:
    frames = glob(f"tmp/{video_id}/*.png")
    frames = sort_path(frames)
    for key in frames:
        results = model(f"{frames[key]}")
        instances = results[0].keypoints.xy
        instances = instances.cpu().numpy()

        image = plt.imread(f"{frames[key]}")
        height, width, _ = image.shape
        x, y = width // 2, height // 2

        for points in instances:
            if check_intersection(points, x, y):
                with open(f"tmp/{video_id}/frame_{key}.pkl", "wb") as f:
                    pickle.dump(points, f)
                break
