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
import yaml
seed = 1205
random.seed(seed)
np.random.seed(seed)


with open("./nuscenes.yaml", 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
learning_map = semkittiyaml['learning_map']

def parse_arguments():

    parser = argparse.ArgumentParser(description='LiDAR Incomplete Echo')
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default=mp.cpu_count())
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='./data_root/nuScenes')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset',
                        default='./save_root/incomplete_echo/light')  # ['light','moderate','heavy']
    parser.add_argument('-t', '--drop_ratio', help='drop ratio of instance points', type=float, default=0.75)
    arguments = parser.parse_args()

    return arguments






if __name__ == '__main__':
    args = parse_arguments()
    # incomplete echo (light: 0.75, moderate: 0.85, heavy: 0.95)
    drop_ratio = args.drop_ratio
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
        label = np.vectorize(learning_map.__getitem__)(sem_label)

        # bicycle
        pix_bicycle = label == 2  
        if np.sum(pix_bicycle) > 10:
            idx_pix_bicycle = np.squeeze(np.argwhere(pix_bicycle))

            # to drop
            num_pix_to_drop = int(len(idx_pix_bicycle) * drop_ratio)
            idx_pix_bicycle_to_drop = np.random.choice(idx_pix_bicycle, num_pix_to_drop, replace=False)

            scan = np.delete(scan, idx_pix_bicycle_to_drop, axis=0)
            label = np.delete(label, idx_pix_bicycle_to_drop, axis=0)  # label mapped to 0-16
            sem_label = np.delete(sem_label, idx_pix_bicycle_to_drop, axis=0)  # original mapping
            assert len(scan) == len(label)
            assert len(scan) == len(sem_label)

        # bus
        pix_bus = label == 3  
        if np.sum(pix_bus) > 10:
            idx_pix_bus = np.squeeze(np.argwhere(pix_bus))

            # to drop
            num_pix_to_drop = int(len(idx_pix_bus) * drop_ratio)
            idx_pix_bus_to_drop = np.random.choice(idx_pix_bus, num_pix_to_drop, replace=False)

            scan = np.delete(scan, idx_pix_bus_to_drop, axis=0)
            label = np.delete(label, idx_pix_bus_to_drop, axis=0)  # label mapped to 0-16
            sem_label = np.delete(sem_label, idx_pix_bus_to_drop, axis=0)  # original mapping
            assert len(scan) == len(label)
            assert len(scan) == len(sem_label)

        # car
        pix_car = label == 4 
        if np.sum(pix_car) > 10:
            idx_pix_car = np.squeeze(np.argwhere(pix_car))

            # to drop
            num_pix_to_drop = int(len(idx_pix_car) * drop_ratio)
            print(num_pix_to_drop)
            idx_pix_car_to_drop = np.random.choice(idx_pix_car, num_pix_to_drop, replace=False)

            scan = np.delete(scan, idx_pix_car_to_drop, axis=0)
            label = np.delete(label, idx_pix_car_to_drop, axis=0)  # label mapped to 0-16
            sem_label = np.delete(sem_label, idx_pix_car_to_drop, axis=0)  # original mapping
            assert len(scan) == len(label)
            assert len(scan) == len(sem_label)

        # construction_vehicle
        pix_cv = label == 5  
        if np.sum(pix_cv ) > 10:
            idx_pix_cv  = np.squeeze(np.argwhere(pix_cv))

            # to drop
            num_pix_to_drop = int(len(idx_pix_cv) * drop_ratio)
            idx_pix_cv_to_drop = np.random.choice(idx_pix_cv, num_pix_to_drop, replace=False)

            scan = np.delete(scan, idx_pix_cv_to_drop, axis=0)
            label = np.delete(label, idx_pix_cv_to_drop, axis=0)  # label mapped to 0-16
            sem_label = np.delete(sem_label, idx_pix_cv_to_drop, axis=0)  # original mapping
            assert len(scan) == len(label)
            assert len(scan) == len(sem_label)

        # motorcycle
        pix_motorcycle = label == 6  
        if np.sum(pix_motorcycle) > 10:
            idx_pix_motorcycle = np.squeeze(np.argwhere(pix_motorcycle))

            # to drop
            num_pix_to_drop = int(len(idx_pix_motorcycle) * drop_ratio)
            idx_pix_motorcycle_to_drop = np.random.choice(idx_pix_motorcycle, num_pix_to_drop, replace=False)

            scan = np.delete(scan, idx_pix_motorcycle_to_drop, axis=0)
            label = np.delete(label, idx_pix_motorcycle_to_drop, axis=0)  # label mapped to 0-16
            sem_label = np.delete(sem_label, idx_pix_motorcycle_to_drop, axis=0)  # original mapping
            assert len(scan) == len(label)
            assert len(scan) == len(sem_label)       

        # trailer
        pix_trailer = label == 9  
        if np.sum(pix_trailer) > 10:
            idx_pix_trailer = np.squeeze(np.argwhere(pix_trailer))

            # to drop
            num_pix_to_drop = int(len(idx_pix_trailer) * drop_ratio)
            idx_pix_trailer_to_drop = np.random.choice(idx_pix_trailer, num_pix_to_drop, replace=False)

            scan = np.delete(scan, idx_pix_trailer_to_drop, axis=0)
            label = np.delete(label, idx_pix_trailer_to_drop, axis=0)  # label mapped to 0-16
            sem_label = np.delete(sem_label, idx_pix_trailer_to_drop, axis=0)  # original mapping
            assert len(scan) == len(label)
            assert len(scan) == len(sem_label)  

        # truck
        pix_truck = label == 10  
        if np.sum(pix_truck) > 10:
            idx_pix_truck = np.squeeze(np.argwhere(pix_truck))

            # to drop
            num_pix_to_drop = int(len(idx_pix_truck) * drop_ratio)
 
            idx_pix_truck_to_drop = np.random.choice(idx_pix_truck, num_pix_to_drop, replace=False)

            scan = np.delete(scan, idx_pix_truck_to_drop, axis=0)
            label = np.delete(label, idx_pix_truck_to_drop, axis=0)  # label mapped to 0-16
            sem_label = np.delete(sem_label, idx_pix_truck_to_drop, axis=0)  # original mapping
            assert len(scan) == len(label)
            assert len(scan) == len(sem_label)  
        lidar_save_path = os.path.join(dst_folder, lidar_path)
        label_save_path = os.path.join(dst_folder, label_path )

        scan.astype(np.float32).tofile(lidar_save_path)
        sem_label = sem_label.reshape(-1,1)
        sem_label.astype(np.uint8).tofile(label_save_path)


    n = len(all_files)

    with mp.Pool(args.n_cpus) as pool:

        l = list(tqdm(pool.imap(_map, range(n)), total=n))
