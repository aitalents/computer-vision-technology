import cv2
import numpy as np


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def batch_preprocessing(img_list: list) -> np.ndarray:
    batch = []
    
    for img in img_list:
        img = letterbox(img)[0]
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        img /= 255
        batch.append(img)
    return np.array(batch)


def plot_detections(frame: np.ndarray, det_results: np.ndarray):
    img1 = letterbox(frame.copy())[0]
    for frame_res in det_results:
        if frame_res != []:
            for i, res in enumerate(frame_res):
                x1, y1 = int(res[0]), int(res[1])
                x2, y2 = int(res[2]), int(res[3])

                label = f"person_{i}, conf={round(res[4], 3)}"
                img1 = cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return img1