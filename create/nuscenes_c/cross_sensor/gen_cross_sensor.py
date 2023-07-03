import argparse
import os
import random
import time
import glob
import multiprocessing as mp
import copy
from pathlib import Path
from tqdm import tqdm
from nuscenes import NuScenes
import pickle
import numpy as np

seed = 1205
random.seed(seed)
np.random.seed(seed)




def parse_arguments():

    parser = argparse.ArgumentParser(description='LiDAR cross_sensor')
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default= mp.cpu_count())
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='./data_root/nuScenes')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset', type=str,
                        default='./save_root/cross_sensor/light')  # ['light','moderate','heavy']
    parser.add_argument('-n', '--num_beam_to_drop', help='number of beam to be dropped', type=int, default=8)
    arguments = parser.parse_args()

    return arguments





if __name__ == '__main__':
    args = parse_arguments()
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
        scan = np.fromfile(os.path.join(args.root_folder, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5]) 

        lidar_sd_token = nusc_info.get('sample', info['token'])['data']['LIDAR_TOP']
        label_path = nusc_info.get('lidarseg', lidar_sd_token)['filename']
        lidarseg_labels_filename = os.path.join(args.root_folder, label_path)
        label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape(-1)

        # get beam id
        beam_id = scan[:,-1]
        beam_id = beam_id.astype(np.int64)

        if num_beam_to_drop == 8:
            to_drop = np.arange(1, 32, 4)
            assert len(to_drop) == 8
        
        elif num_beam_to_drop == 16:
            to_drop = np.arange(1, 32, 2)
            assert len(to_drop) == 16

        elif num_beam_to_drop == 24:
            to_drop = np.arange(1, 32, 1.33)
            to_drop = to_drop.astype(int)
            assert len(to_drop) == 24

        to_keep = [i for i in np.arange(0, 32, 1) if i not in to_drop]
        assert len(to_drop) + len(to_keep) == 32


        for id in to_drop:
            points_to_drop = beam_id == id

            scan = np.delete(scan, points_to_drop, axis=0)
            label = np.delete(label, points_to_drop, axis=0)
            assert len(scan) == len(label)

            beam_id = np.delete(beam_id, points_to_drop, axis=0)


        scan = scan[::2, :]
        label = label[::2]

        assert len(scan) == len(label)     

        lidar_save_path = os.path.join(dst_folder, lidar_path)
        label_save_path = os.path.join(dst_folder, label_path )

        scan.astype(np.float32).tofile(lidar_save_path)
        label.astype(np.uint8).tofile(label_save_path)

    n = len(all_files)

    with mp.Pool(args.n_cpus) as pool:

        l = list(tqdm(pool.imap(_map, range(n)), total=n))


