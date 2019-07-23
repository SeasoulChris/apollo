import argparse
import numpy as np
import os


def merge_dicts(dirpath, dict_name='future_status'):
    '''
    Merge all dictionaries directly under a directory
    '''
    list_of_files = os.listdir(dirpath)
    dict_merged = dict()
    for filename in list_of_files:
        full_path = os.path.join(dirpath, filename)
        if filename.endswith(dict_name + '.npy'):
            dict_curr = np.load(full_path).item()
            dict_merged.update(dict_curr)
    np.save(os.path.join(dirpath, 'merged_' + dict_name + '.npy'), dict_merged)
    return dict_merged

def GetListOfFiles(dirpath):
    list_of_files = os.listdir(dirpath)
    all_files = []

    for file in list_of_files:
        full_path = os.path.join(dirpath, file)
        if os.path.isdir(full_path):
            all_files = all_files + GetListOfFiles(full_path)
        else:
            all_files.append(full_path)

    return all_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge all label_dicts in each terminal folder.')
    parser.add_argument('dirpath', type=str, help='Path of terminal folder.')
    args = parser.parse_args()

    all_files = GetListOfFiles(args.dirpath)
    all_child_folders = set()
    for filename in all_files:
        all_child_folders.add(os.path.dirname(filename))

    for dirname in all_child_folders:
        merge_dicts(dirname, dict_name='visited_lane_segment')
