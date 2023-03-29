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

def parse_arguments():

    parser = argparse.ArgumentParser(description='LiDAR motion blur')

    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default= mp.cpu_count())
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset', type=str,
                        default='./data_root/SemanticKITTI/sequences')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset', type=str,
                        default='./save_root/motion_blur/light')  # ['light','moderate','heavy']
    parser.add_argument('-t', '--trans_std', help='jitter score', type=float, default=0.2)

    arguments = parser.parse_args()

    return arguments




if __name__ == '__main__':
    args = parse_arguments()
    # motion (jitter score)
    trans_std = [args.trans_std]*3             
    # light:[0.2, 0.2, 0.2]; moderate:[0.25, 0.25, 0.25]; heavy:[0.3, 0.3, 0.3]    
    print(trans_std)

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
        label = label & 0xFFFF
        assert label is not None

        noise_translate = np.array([
            np.random.normal(0, trans_std[0], 1),
            np.random.normal(0, trans_std[1], 1),
            np.random.normal(0, trans_std[2], 1),
            ]).T

        points[:, 0:3] += noise_translate
        num_points = points.shape[0]
        jitters_x = np.clip(np.random.normal(loc=0.0, scale=trans_std[0]*0.1, size=num_points), -3 * trans_std[0], 3 * trans_std[0])
        jitters_y = np.clip(np.random.normal(loc=0.0, scale=trans_std[1]*0.1, size=num_points), -3 * trans_std[1], 3 * trans_std[1])
        jitters_z = np.clip(np.random.normal(loc=0.0, scale=trans_std[2]*0.05, size=num_points), -3 * trans_std[2], 3 * trans_std[2])

        points[:, 0] += jitters_x
        points[:, 1] += jitters_y
        points[:, 2] += jitters_z

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