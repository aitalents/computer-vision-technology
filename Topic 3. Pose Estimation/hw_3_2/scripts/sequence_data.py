from glob import glob
import os
import pickle
import numpy as np
import pandas as pd
import torch
from config import dance_styles


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_points() -> dict:
    files = glob("data/points/*.pkl")

    points_data = {}
    for file in files:
        points = load_pickle(file)
        if len(points) >= 45:
            points = np.stack(points[:45], axis=0)
            points = points.reshape(45, -1)
            video_id = os.path.basename(file).split(".pkl")[0]
            points_data[video_id] = points
    return points_data


points_data = collect_points()

train = pd.read_csv("data/train_split.csv")
valid = pd.read_csv("data/valid_split.csv")
test = pd.read_csv("data/test_split.csv")


def filter_df(df):
    accessed_video = set(points_data.keys())
    df_video = set(df["video_id"])
    accessed_video = accessed_video & df_video
    df = df[df["video_id"].isin(accessed_video)]
    return df


train = filter_df(train)
valid = filter_df(valid)
test = filter_df(test)


class PointsDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data = dict(zip(df["video_id"], df["label"]))
        self.id = list(self.data.keys())


    def __len__(self):
        return len(self.id)


    def __getitem__(self, idx):
        video_id = self.id[idx]
        x = points_data[video_id]
        x = torch.from_numpy(x)

        dance = self.data[video_id]
        y = torch.tensor(dance_styles[dance])
        return x, y


train_dataset = PointsDataset(train)
valid_dataset = PointsDataset(valid)
test_dataset = PointsDataset(test)
