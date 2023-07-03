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

    parser = argparse.ArgumentParser(description='LiDAR motion blur')
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default= mp.cpu_count())
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='/data_root/nuScenes')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset', type=str,
                        default='./save_root/motion_blur/light')  # ['light','moderate','heavy']
    parser.add_argument('-t', '--trans_std', help='jitter score', type=float, default=0.2)
    arguments = parser.parse_args()

    return arguments





if __name__ == '__main__':
    args = parse_arguments()
    # motion (jitter score)
    trans_std = [args.trans_std]*3              
    print(trans_std)
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
        points = np.fromfile(os.path.join(args.root_folder, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])    #[:, :4]
        
        lidar_sd_token = nusc_info.get('sample', info['token'])['data']['LIDAR_TOP']
        label_path = nusc_info.get('lidarseg', lidar_sd_token)['filename']
        lidarseg_labels_filename = os.path.join(args.root_folder, label_path)
        sem_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape(-1, 1)

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

        lidar_save_path = os.path.join(dst_folder, lidar_path)
        label_save_path = os.path.join(dst_folder, label_path )

        points.astype(np.float32).tofile(lidar_save_path)

        sem_label = sem_label.reshape(-1,1)
        sem_label.astype(np.uint8).tofile(label_save_path)

    n = len(all_files)

    with mp.Pool(args.n_cpus) as pool:

        l = list(tqdm(pool.imap(_map, range(n)), total=n))