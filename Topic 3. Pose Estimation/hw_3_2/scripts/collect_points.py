import os
import pickle
from ultralytics import YOLO
from glob import glob
from utils import check_intersection


def extract_points(results):
    points_obj = []
    id_obj = -1
    stop = False
    found = False

    for r in results:
        if stop:
            break

        if r.boxes.id is None:
            continue

        ids = r.boxes.id.numpy()
        all_points = r.keypoints.xy.cpu().numpy()
        assert len(ids) == len(all_points)

        if id_obj == -1:
            for i, points in enumerate(all_points):
                if check_intersection(points, r.orig_img.shape):
                    id_obj = ids[i]
                    found = True
                    points_obj.append(points)
                    break

        if found and id_obj != -1:
            found = False
            continue

        elif not found and id_obj != -1:
            for i, points in enumerate(all_points):
                if id_obj not in ids:
                    stop = True
                    break
                elif ids[i] == id_obj:
                    points_obj.append(points)

    if points_obj:
        return points_obj


model = YOLO("yolov8n-pose.pt")

def main():
    videos = glob("data/train/videos/*.mp4")

    for video in videos:
        video_id = os.path.basename(video).split(".mp4")[0].replace("video_", "")
        results = model.track(source=video, stream=True, verbose=False)
        points_obj = extract_points(results)
        if points_obj is not None:
            with open(f"data/points/{video_id}.pkl", "wb") as f:
                pickle.dump(points_obj, f)


if __name__ == "__main__":
    main()
