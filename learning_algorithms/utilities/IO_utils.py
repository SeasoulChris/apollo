#!/usr/bin/env python
"""IO utils."""

import os


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
