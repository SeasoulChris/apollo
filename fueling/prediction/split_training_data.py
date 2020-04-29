import glob
import os

import numpy as np

file_prefix = "/data/kinglong_train_clean/"
target_prefix = "/data/kinglong_train_clean_split/"

all_file_paths = glob.glob(file_prefix + "**/training_data.npy", recursive=True)
for file_path in all_file_paths:
    file_content = np.load(file_path, allow_pickle=True).tolist()
    os.makedirs(os.path.dirname(file_path.replace(file_prefix, target_prefix)))
    writing_scene = []
    writing_idx = 0
    for scene_id, scene in enumerate(file_content):
        writing_scene += [scene]
        if (len(writing_scene) >= 100):
            np.save(file_path.replace(file_prefix, target_prefix).replace('.npy', '.{}.npy'.format(writing_idx)), np.array(writing_scene))
            writing_scene.clear()
            writing_idx += 1
    np.save(file_path.replace(file_prefix, target_prefix).replace('.npy', '.{}.npy'.format(writing_idx)), np.array(writing_scene))
