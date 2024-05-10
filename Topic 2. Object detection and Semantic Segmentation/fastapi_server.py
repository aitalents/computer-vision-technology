from fastapi import FastAPI, WebSocket
import asyncio
import glob
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from scipy.optimize import linear_sum_assignment
from track_14 import country_balls_amount, track_data, name, param
import cv2

app = FastAPI(title='Tracker assignment')

images = glob.glob('imgs/*')
country_balls = [{'cb_id': idx, 'img': images[idx % len(images)]} for idx in range(country_balls_amount)]
print('Server started')

def calculate_euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def create_distance_matrix(previous_centers, current_centers):
    distance_matrix = np.zeros((len(previous_centers), len(current_centers)))
    for i, prev_center in enumerate(previous_centers):
        for j, curr_center in enumerate(current_centers):
            distance_matrix[i, j] = calculate_euclidean_distance(prev_center, curr_center)
    return distance_matrix

def convert_to_opencv_format(xyxy_bbox):
    x1, y1, x2, y2 = xyxy_bbox
    return [x1, y1, x2 - x1, y2 - y1]

def convert_to_yolo_format(xyxy_bbox):
    x1, y1, x2, y2 = xyxy_bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]

# Инициализация свободных идентификаторов треков
free_track_ids = list(range(50))


# Реализация алгоритма трекера (soft)
def soft_tracker(current_frame, previous_frame):
    FP = 0
    FN = 0
    IDSW = 0

    if not previous_frame:
        for obj in current_frame['data']:
            if obj['bounding_box']:
                obj['track_id'] = free_track_ids.pop(0)
            else:
                obj['track_id'] = None
        return current_frame, FP, FN, IDSW

    for obj in current_frame['data']:
        if not obj['bounding_box']:
            obj['track_id'] = None

    for obj in previous_frame['data']:
        if not obj['bounding_box']:
            obj['track_id'] = None

    previous_centers = [convert_to_yolo_format(obj['bounding_box'])[:2] for obj in previous_frame['data'] if obj['bounding_box']]
    current_centers = [convert_to_yolo_format(obj['bounding_box'])[:2] for obj in current_frame['data'] if obj['bounding_box']]

    distance_matrix = create_distance_matrix(previous_centers, current_centers)
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    assigned_ids = set()
    for row, col in zip(row_indices, col_indices):
        if distance_matrix[row, col] < 50:
            current_frame['data'][col]['track_id'] = previous_frame['data'][row]['track_id']
            assigned_ids.add(previous_frame['data'][row]['track_id'])
        else:
            FP += 1

    for obj in current_frame['data']:
        if obj['bounding_box'] and obj['track_id'] is None:
            obj['track_id'] = free_track_ids.pop(0)
            FP += 1

    for obj in previous_frame['data']:
        if obj['bounding_box'] and obj['track_id'] not in assigned_ids:
            FN += 1

    used_ids = [obj['track_id'] for obj in current_frame['data'] if obj['bounding_box']]
    available_ids = list(range(100))

    if len(current_centers) < len(previous_centers):
        for track_id in available_ids:
            if track_id not in used_ids and track_id not in free_track_ids:
                free_track_ids.append(track_id)

    free_track_ids.sort()
    return current_frame, FP, FN, IDSW

# Реализация алгоритма трекера (strong)
def strong_tracker(current_frame, deepsort_tracker):
    FP = 0
    FN = 0
    IDSW = 0

    frame_path = f'./frames/{name}/{current_frame["frame_id"]}.png'
    image = cv2.imread(frame_path)
    if image is None or image.size == 0:
        raise ValueError(f"Frame is empty or not loaded correctly from {frame_path}")

    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes = [(convert_to_opencv_format(obj['bounding_box']), 1, 1) for obj in current_frame['data'] if
              obj['bounding_box']]
    tracks = deepsort_tracker.update_tracks(bboxes, frame=frame)

    track_dict = {track.track_id: convert_to_yolo_format(track.to_ltrb())[:2] for track in tracks}

    detected_track_ids = set(track_dict.keys())
    gt_track_ids = {obj['track_id'] for obj in current_frame['data'] if obj['bounding_box']}

    # Calculate FP and FN
    for track_id in detected_track_ids:
        if track_id not in gt_track_ids:
            FP += 1

    for gt_track_id in gt_track_ids:
        if gt_track_id not in detected_track_ids:
            FN += 1

    for obj in current_frame['data']:
        if obj['bounding_box']:
            bbox_center = convert_to_yolo_format(obj['bounding_box'])[:2]
            closest_track_id = min(track_dict,
                                   key=lambda track_id: calculate_euclidean_distance(track_dict[track_id], bbox_center))
            if obj['track_id'] != closest_track_id:
                IDSW += 1
            obj['track_id'] = closest_track_id

    return current_frame, FP, FN, IDSW


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Client connected')
    await websocket.accept()
    deepsort_tracker = DeepSort(max_age=5)
    tracking_ids = {i: [] for i in range(country_balls_amount)}
    await websocket.send_text(str(country_balls))
    previous_frame = None

    total_FP = 0
    total_FN = 0
    total_IDSW = 0
    total_GT = 0

    for current_frame in track_data:
        await asyncio.sleep(0.1)

        # Выбор трекера на основе условия: 0 - soft, 1 - strong
        tracker_choice = 0  # Здесь вы можете задать условие для выбора трекера

        if tracker_choice == 0:
            tracker_name = "soft"
            current_frame, FP, FN, IDSW = soft_tracker(current_frame, previous_frame)
            previous_frame = current_frame
        elif tracker_choice == 1:
            tracker_name = "strong"
            current_frame, FP, FN, IDSW = strong_tracker(current_frame, deepsort_tracker)

        for obj in current_frame['data']:
            if obj['track_id'] is not None:
                tracking_ids[obj['cb_id']].append(obj['track_id'])

        total_FP += FP
        total_FN += FN
        total_IDSW += IDSW
        total_GT += len([obj for obj in current_frame['data'] if obj['bounding_box']])

        await websocket.send_json(current_frame)

    print(tracking_ids)

    # Вычисление MOTA
    MOTA = 1 - (total_FP + total_FN + total_IDSW) / total_GT if total_GT > 0 else 0

    print(f"Total Tracker Metrics:")
    print(f"total_FP: {total_FP}")
    print(f"total_FN: {total_FN}")
    print(f"total_IDSW: {total_IDSW}")
    print(f"total_GT: {total_GT}")
    print(f"MOTA: {MOTA}")

    print('Connection closed')
