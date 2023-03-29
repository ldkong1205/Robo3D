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

def lidar_crosstalk_noise(pointcloud, percentage):
    N, C = pointcloud.shape  # [m,], 4 (xyzi)
    c = int(percentage * N)
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.normal(size=(c, C)) * 3.0
    return pointcloud, index

def parse_arguments():
    parser = argparse.ArgumentParser(description='LiDAR crosstalk')
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default= mp.cpu_count())
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset', type=str,
                        default='./data_root/SemanticKITTI/sequences')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset', type=str,
                        default='./save_root/crosstalk/light')  # ['light','moderate','heavy']
    parser.add_argument('-p', '--percentage', help='crosstalk ratio', type=float, default=0.006)

    arguments = parser.parse_args()

    return arguments





if __name__ == '__main__':
    args = parse_arguments()
    percentage = args.percentage
    # light: 0.006, moderate: 0.008, heavy: 0.01
    print(percentage)

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
            ).reshape(-1)
        label = label & 0xFFFF
        assert label is not None
        crosstalk_scan, index = lidar_crosstalk_noise(points, percentage=percentage)

        label[index] = 23  # ignored (crosstalk)
        label = label.reshape(-1,1)

        lidar_save_path = os.path.join(dst_folder,'velodyne', all_files[i].split('/')[-1])
        if not os.path.exists(os.path.dirname(lidar_save_path)):
            os.makedirs(os.path.dirname(lidar_save_path))
        crosstalk_scan.astype(np.float32).tofile(lidar_save_path)


        label_save_path1 =  os.path.join(dst_folder,'labels', all_files[i].split('/')[-1].replace('bin', 'label'))
        if not os.path.exists(os.path.dirname(label_save_path1)):
            os.makedirs(os.path.dirname(label_save_path1))
        label = label.astype(np.uint32)
        label.tofile(label_save_path1)

    n = len(all_files)

    with mp.Pool(args.n_cpus) as pool:

        l = list(tqdm(pool.imap(_map, range(n)), total=n))