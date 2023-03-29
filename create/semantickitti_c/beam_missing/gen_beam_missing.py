import argparse
import os
import random
import time
import glob
import numpy as np
import multiprocessing as mp
import copy
from pathlib import Path
from tqdm import tqdm

seed = 1205
random.seed(seed)
np.random.seed(seed)

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def get_kitti_ringID(points):
        scan_x = points[:, 0]
        scan_y = points[:, 1]

        yaw = -np.arctan2(scan_y, -scan_x)
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
        proj_y = np.zeros_like(proj_x)
        proj_y[new_raw] = 1
        ringID = np.cumsum(proj_y)
        ringID = np.clip(ringID, 0, 63)
        return ringID

def parse_arguments():

    parser = argparse.ArgumentParser(description='LiDAR beam missing')

    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default= mp.cpu_count())
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset', type=str,
                        default='./data_root/SemanticKITTI/sequences')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset', type=str,
                        default='./save_root/beam_missing/light')  # ['light','moderate','heavy']
    parser.add_argument('-b', '--num_beam_to_drop', help='number of beam to be dropped', type=int, default=16)

    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    args = parse_arguments()
    # beam lost (light: 16, moderate: 32, heavy: 48)
    num_beam_to_drop = args.num_beam_to_drop
    print(num_beam_to_drop)

    print('')
    print(f'using {args.n_cpus} CPUs')

    src_folder =args.root_folder
    all_files = []
    all_files += absoluteFilePaths('/'.join([src_folder, str('08').zfill(2), 'velodyne']))
    all_files.sort()
    all_paths =  copy.deepcopy(all_files)
    dst_folder = args.dst_folder
    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    def _map(i: int) -> None:
        points = np.fromfile(all_paths[i], dtype=np.float32)
        points = points.reshape((-1, args.n_features))
        assert points is not None

        label = np.fromfile(all_paths[i].replace('velodyne', 'labels')[:-3] + 'label', dtype=np.uint32
            ).reshape((-1, 1))
        assert label is not None

        # get beam id
        beam_id = get_kitti_ringID(points)
        beam_id = beam_id.astype(np.int64)

        drop_range = np.arange(4, 59, 1)
        to_drop = np.random.choice(drop_range, num_beam_to_drop)
        for id in to_drop:
            points_to_drop = beam_id == id

            points = np.delete(points, points_to_drop, axis=0)
            label = np.delete(label, points_to_drop, axis=0)
            beam_id = np.delete(beam_id, points_to_drop, axis=0)

        lidar_save_path = os.path.join(dst_folder,'velodyne', all_files[i].split('/')[-1])
        if not os.path.exists(os.path.dirname(lidar_save_path)):
            os.makedirs(os.path.dirname(lidar_save_path))
        points.astype(np.float32).tofile(lidar_save_path)


        label_save_path1 =  os.path.join(dst_folder,'labels', all_files[i].split('/')[-1].replace('bin', 'label'))
        if not os.path.exists(os.path.dirname(label_save_path1)):
            os.makedirs(os.path.dirname(label_save_path1))
        label = label.astype(np.uint32)
        label.tofile(label_save_path1)

    n = len(all_files)

    with mp.Pool(args.n_cpus) as pool:

        l = list(tqdm(pool.imap(_map, range(n)), total=n))