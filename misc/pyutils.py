import numpy as np
import os
import random
import glob


def seed_random(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_paths(image_folder_path, suffix='*.png'):
    """
    :param image_folder_path: str
    :param suffix: str
    :return: list
    """
    paths = sorted(glob.glob(os.path.join(image_folder_path, suffix)))
    return paths


def get_paths_from_list(image_folder_path, list):
    out = []
    for item in list:
        path = os.path.join(image_folder_path,item)
        out.append(path)
    return sorted(out)


