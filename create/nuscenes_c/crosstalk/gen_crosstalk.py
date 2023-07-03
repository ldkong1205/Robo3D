import argparse
import os
import time
import glob
import numpy as np
import multiprocessing as mp
import copy
from pathlib import Path
from tqdm import tqdm
from nuscenes import NuScenes
import pickle
import random

seed = 1205
random.seed(seed)
np.random.seed(seed)
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description='LiDAR crosstalk')
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default=mp.cpu_count())
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='./data_root/nuScenes')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset', type=str,
                        default='./save_root/crosstalk/light')  # ['light','moderate','heavy']
    parser.add_argument('-p', '--percentage', help='crosstalk ratio', type=float, default=0.03)
    arguments = parser.parse_args()

    return arguments


def lidar_crosstalk_noise(pointcloud, percentage):

    N, C = pointcloud.shape  # [m,], 4 (xyzi)
    c = int(percentage * N)
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.normal(size=(c, C)) * 3.0

    return pointcloud, index




if __name__ == '__main__':
    args = parse_arguments()
    percentage = args.percentage
    print(percentage)
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
        scan = np.fromfile(os.path.join(args.root_folder, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5]) 

        lidar_sd_token = nusc_info.get('sample', info['token'])['data']['LIDAR_TOP']
        label_path = nusc_info.get('lidarseg', lidar_sd_token)['filename']
        lidarseg_labels_filename = os.path.join(args.root_folder, label_path)
        sem_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape(-1)


        crosstalk_scan, index = lidar_crosstalk_noise(scan[:,:-1], percentage=percentage)
        crosstalk_scan = np.concatenate((crosstalk_scan, scan[:, -1].reshape(-1,1)), axis=1)
        assert crosstalk_scan.shape[1] == 5

        sem_label[index] = 43  # ignored (crosstalk)

        sem_label = sem_label.reshape(-1,1)

        lidar_save_path = os.path.join(dst_folder, lidar_path)
        label_save_path = os.path.join(dst_folder, label_path )

        crosstalk_scan.astype(np.float32).tofile(lidar_save_path)

        sem_label = sem_label.reshape(-1,1)
        sem_label.astype(np.uint8).tofile(label_save_path)




    n = len(all_files)

    with mp.Pool(args.n_cpus) as pool:

        l = list(tqdm(pool.imap(_map, range(n)), total=n))





