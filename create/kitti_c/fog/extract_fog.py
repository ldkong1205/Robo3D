__author__  = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import os
import csv
import argparse

import numpy as np

from tqdm import tqdm
from typing import List
from pathlib import Path

MIN_DIST = 1.75         # in m
MAX_DIST = 10           # in m

MIN_HEIGHT = -1         # in m
MAX_HEIGHT = np.inf     # in m

SPLIT = 'test_dense_fog_night'
TOPIC = 'lidar_hdl64_strongest'



def get_recordings(split: str) -> List[str]:

    splits_folder = Path(__file__).parent.absolute() / 'SeeingThroughFog' / 'splits'

    splits = sorted(os.listdir(splits_folder))

    assert f'{split}.txt' in splits, f'{split} is undefined'

    recordings = []

    with open(f'{splits_folder / split}.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            recordings.append(f'{row[0]}_{row[1]}.png')

    return sorted(recordings)


def extract_fog(arguments: argparse.Namespace, recordings: List[str]) -> None:

    points_before_sum = 0
    points_after_sum = 0

    avg_num_point_before = 0
    avg_num_point_after = 0

    prog_bar = tqdm(recordings)

    save_dir = Path(arguments.root_path) / f'{arguments.topic}_fog_extraction'
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, recording in enumerate(prog_bar):

        file_name = recording.replace('.png', '.bin')
        path = Path(arguments.root_path) / arguments.topic / file_name

        pc = np.fromfile(path, dtype=np.float32)
        pc = pc.reshape((-1, 5))

        points_before = len(pc)

        pc = filter_ego_point(pc)
        pc = filter_by_distance(pc)
        pc = filter_by_height(pc)

        points_after = len(pc)

        points_before_sum += points_before
        points_after_sum += points_after

        avg_num_point_before = (avg_num_point_before * i + points_before) / (i + 1)
        avg_num_point_after = (avg_num_point_after * i + points_after) / (i + 1)

        save_path =  save_dir / file_name
        pc.astype(np.float32).tofile(save_path)

        prog_bar.set_description(f'{int(avg_num_point_after)}/{int(avg_num_point_before)}')

    num_recordings = len(recordings)

    avg_num_point_before = points_before_sum / num_recordings
    avg_num_point_after = points_after_sum / num_recordings

    print(f'average points before: {avg_num_point_before}')
    print(f'average points after:   {avg_num_point_after}')


def filter_ego_point(pc: np.ndarray, l: float = 5.116, w: float = 1.899, h: float = 1.496) -> np.ndarray:

    # default dimensions are dimensions of W222 taken from wikipedia

    x_mask_lower = -l / 2 < pc[:, 0]
    x_mask_upper = pc[:, 0] < l / 2
    x_mask = (x_mask_lower == 1) & (x_mask_upper == 1)

    y_mask_lower = -w / 2 < pc[:, 1]
    y_mask_upper = pc[:, 1] < w / 2
    y_mask = (y_mask_lower == 1) & (y_mask_upper == 1)

    z_mask_lower = -h < pc[:, 2]
    z_mask_upper = pc[:, 2] < -h / 2
    z_mask = (z_mask_lower == 1) & (z_mask_upper == 1)

    inside_mask = (x_mask == 1) & (y_mask == 1) & (z_mask == 1)
    outside_mask = ~inside_mask

    pc = pc[outside_mask, :]

    return pc


def filter_by_distance(pc: np.ndarray, min_dist: float = MIN_DIST, max_dist: float = MAX_DIST) -> np.ndarray:

    min_dist_mask = np.linalg.norm(pc[:, 0:3], axis=1) > min_dist
    pc = pc[min_dist_mask, :]

    max_dist_mask = np.linalg.norm(pc[:, 0:3], axis=1) < max_dist
    pc = pc[max_dist_mask, :]

    return pc


def filter_by_height(pc: np.ndarray, min_height: float = MIN_HEIGHT, max_height: float = MAX_HEIGHT) -> np.ndarray:

    min_height_mask = pc[:, 2] > min_height
    pc = pc[min_height_mask, :]

    max_height_mask = pc[:, 2] < max_height
    pc = pc[max_height_mask, :]

    return pc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default=str(Path.home() / 'datasets/DENSE/SeeingThroughFog'),
                        help='path to SeeingThroughFog dataset')
    parser.add_argument('-s', '--split', type=str, default=SPLIT)
    parser.add_argument('-t', '--topic', type=str, default=TOPIC)

    args = parser.parse_args()
    recs = get_recordings(args.split)

    extract_fog(args, recs)
