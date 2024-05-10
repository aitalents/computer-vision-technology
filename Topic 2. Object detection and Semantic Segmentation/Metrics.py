import pickle
import numpy as np
import pandas as pd
import os

with open('./metrics_dicts/track_12_strong_ids.pkl', 'rb') as f:
    metrics_dict = pickle.load(f)


def get_avg_jumps(ids_dict):
    jump_ratios = []

    for key in ids_dict.keys():
        track = np.array(ids_dict[key]).astype(int)
        if len(track) == 0:
            jump_ratios.append(0)
            continue

        changes = np.diff(track)
        num_jumps = len(changes[changes != 0])
        jump_ratio = num_jumps / len(track)
        jump_ratios.append(jump_ratio)
    return sum(jump_ratios) / len(jump_ratios)

def get_metrics_dataset(path_to_folder):
    names_of_files = os.listdir(path_to_folder)
    names_of_files.sort()
    dict_of_metrics = {}
    for name_of_file in names_of_files:
        path_to_file = path_to_folder + '/' + name_of_file
        with open(path_to_file, 'rb') as f:
            track_ids_dict = pickle.load(f)
        dict_of_metrics[name_of_file] = [get_avg_jumps(track_ids_dict)]
    return pd.DataFrame(dict_of_metrics).T

print(get_avg_jumps(metrics_dict))