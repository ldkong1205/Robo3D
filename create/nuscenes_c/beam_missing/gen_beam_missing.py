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
from nuscenes import NuScenes
import pickle

seed = 1205
random.seed(seed)
np.random.seed(seed)


def parse_arguments():

    parser = argparse.ArgumentParser(description='LiDAR beam missing')
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default= mp.cpu_count())
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='./data_root/nuScenes')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset',
                        default='./save_root/beam_missing/light') # ['light','moderate','heavy']
    parser.add_argument('-b', '--num_beam_to_drop', help='number of beam to be dropped', type=int, default=8)
    arguments = parser.parse_args()

    return arguments




if __name__ == '__main__':
    args = parse_arguments()
    # beam lost (light: 8, moderate: 16, heavy: 24)
    num_beam_to_drop = args.num_beam_to_drop
    print(num_beam_to_drop)
    print('')
    print(f'using {args.n_cpus} CPUs')

    all_files = []
    nusc_info = NuScenes(version='v1.0-trainval', dataroot=args.root_folder, verbose=False)
    imageset = os.path.join(args.root_folder,"nuscenes_infos_val.pkl")
    with open(imageset, 'rb') as f:
            infos = pickle.load(f)
    all_files = infos['infos']

    all_paths =  copy.deepcopy(all_files)
    dst_folder = args.dst_folder

    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    lidar_save_root = os.path.join(dst_folder, 'samples/LIDAR_TOP')
    if not os.path.exists(lidar_save_root):
        os.makedirs(lidar_save_root)
    label_save_root = os.path.join(dst_folder, 'lidarseg/v1.0-trainval')
    if not os.path.exists(label_save_root):
        os.makedirs(label_save_root)
    def _map(i: int) -> None:
        info = all_paths[i]
        lidar_path = info['lidar_path'][16:]
        points = np.fromfile(os.path.join(args.root_folder, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5]) 

        lidar_sd_token = nusc_info.get('sample', info['token'])['data']['LIDAR_TOP']
        label_path = nusc_info.get('lidarseg', lidar_sd_token)['filename']
        lidarseg_labels_filename = os.path.join(args.root_folder, label_path)
        labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])

        # get beam id
        beam_id = points[:,-1]
        beam_id = beam_id.astype(np.int64)

        drop_range = np.arange(2, 29, 1)
        to_drop = np.random.choice(drop_range, num_beam_to_drop)

        for id in to_drop:
            points_to_drop = beam_id == id

            points = np.delete(points, points_to_drop, axis=0)
            labels = np.delete(labels, points_to_drop, axis=0)

            beam_id = np.delete(beam_id, points_to_drop, axis=0)
       

        lidar_save_path = os.path.join(dst_folder, lidar_path)

        label_save_path = os.path.join(dst_folder, label_path )

        points.astype(np.float32).tofile(lidar_save_path)
        labels.astype(np.uint8).tofile(label_save_path)

    n = len(all_files)

    with mp.Pool(args.n_cpus) as pool:

        l = list(tqdm(pool.imap(_map, range(n)), total=n))









