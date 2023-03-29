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
import yaml

seed = 1205
random.seed(seed)
np.random.seed(seed)

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

with open("semantic-kitti.yaml", 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
learning_map = semkittiyaml['learning_map']

def parse_arguments():
    parser = argparse.ArgumentParser(description='LiDAR incomplete echo')
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default= mp.cpu_count())
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='./data_root/SemanticKITTI/sequences')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset',
                        default='./save_root/incomplete_echo/light')  # ['light','moderate','heavy']
    parser.add_argument('-t', '--drop_ratio', help='drop ratio of instance points', type=float, default=0.75)
    arguments = parser.parse_args()

    return arguments





if __name__ == '__main__':
    args = parse_arguments()
    # incomplete echo (light: 0.75, moderate: 0.85, heavy: 0.95)
    drop_ratio = args.drop_ratio
    print(drop_ratio)

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
        scan = np.fromfile(all_paths[i], dtype=np.float32)
        scan = scan.reshape((-1, args.n_features))
        assert scan is not None

        sem_label = np.fromfile(all_paths[i].replace('velodyne', 'labels')[:-3] + 'label', dtype=np.uint32
            ).reshape(-1)
        sem_label = sem_label & 0xFFFF
        assert sem_label is not None
        label = np.vectorize(learning_map.__getitem__)(sem_label)

        # car
        pix_car = label == 1  # cls: 1 (car)
        if np.sum(pix_car) > 10:
            idx_pix_car = np.squeeze(np.argwhere(pix_car))

            # to drop
            num_pix_to_drop = int(len(idx_pix_car) * drop_ratio)
            idx_pix_car_to_drop = np.random.choice(idx_pix_car, num_pix_to_drop, replace=False)

            scan = np.delete(scan, idx_pix_car_to_drop, axis=0)
            label = np.delete(label, idx_pix_car_to_drop, axis=0)  # label mapped to 0-19
            sem_label = np.delete(sem_label, idx_pix_car_to_drop, axis=0)  # original mapping
            assert len(scan) == len(label)
            assert len(scan) == len(sem_label)
        
        # bicycle
        pix_bicycle = label == 2  # cls: 2 (bicycle)
        if np.sum(pix_bicycle) > 10:
            idx_pix_bicycle = np.squeeze(np.argwhere(pix_bicycle))

            # to drop
            num_pix_to_drop = int(len(idx_pix_bicycle) * drop_ratio)
            idx_pix_bicycle_to_drop = np.random.choice(idx_pix_bicycle, num_pix_to_drop, replace=False)

            scan = np.delete(scan, idx_pix_bicycle_to_drop, axis=0)
            label = np.delete(label, idx_pix_bicycle_to_drop, axis=0)  # label mapped to 0-19
            sem_label = np.delete(sem_label, idx_pix_bicycle_to_drop, axis=0)  # original mapping
            assert len(scan) == len(label)
            assert len(scan) == len(sem_label)

        # motorcycle
        pix_motorcycle = label == 3  # cls: 3 (motorcycle)
        if np.sum(pix_motorcycle) > 10:
            idx_pix_motorcycle = np.squeeze(np.argwhere(pix_motorcycle))
            # to drop
            num_pix_to_drop = int(len(idx_pix_motorcycle) * drop_ratio)
            idx_pix_motorcycle_to_drop = np.random.choice(idx_pix_motorcycle, num_pix_to_drop, replace=False)

            scan = np.delete(scan, idx_pix_motorcycle_to_drop, axis=0)
            label = np.delete(label, idx_pix_motorcycle_to_drop, axis=0)  # label mapped to 0-19
            sem_label = np.delete(sem_label, idx_pix_motorcycle_to_drop, axis=0)  # original mapping
            assert len(scan) == len(label)
            assert len(scan) == len(sem_label)

        # truck
        pix_truck = label == 4  # cls: 4 (truck)
        # print(np.sum(pix_truck))
        if np.sum(pix_truck) > 10:
            idx_pix_truck = np.squeeze(np.argwhere(pix_truck))

            # to drop
            num_pix_to_drop = int(len(idx_pix_truck) * drop_ratio)
            idx_pix_truck_to_drop = np.random.choice(idx_pix_truck, num_pix_to_drop, replace=False)

            scan = np.delete(scan, idx_pix_truck_to_drop, axis=0)
            label = np.delete(label, idx_pix_truck_to_drop, axis=0)  # label mapped to 0-19
            sem_label = np.delete(sem_label, idx_pix_truck_to_drop, axis=0)  # original mapping
            assert len(scan) == len(label)
            assert len(scan) == len(sem_label)

        # other-vehicle
        pix_other_vehicle = label == 5  # cls: 5 (other-vehicle)
        if np.sum(pix_other_vehicle) > 10:
            idx_pix_other_vehicle = np.squeeze(np.argwhere(pix_other_vehicle))

            # to drop
            num_pix_to_drop = int(len(idx_pix_other_vehicle) * drop_ratio)
            idx_pix_other_vehicle_to_drop = np.random.choice(idx_pix_other_vehicle, num_pix_to_drop, replace=False)

            scan = np.delete(scan, idx_pix_other_vehicle_to_drop, axis=0)
            label = np.delete(label, idx_pix_other_vehicle_to_drop, axis=0)  # label mapped to 0-19
            sem_label = np.delete(sem_label, idx_pix_other_vehicle_to_drop, axis=0)  # original mapping
            assert len(scan) == len(label)
            assert len(scan) == len(sem_label)



        lidar_save_path = os.path.join(dst_folder,'velodyne', all_files[i].split('/')[-1])
        if not os.path.exists(os.path.dirname(lidar_save_path)):
            os.makedirs(os.path.dirname(lidar_save_path))
        scan.astype(np.float32).tofile(lidar_save_path)

        sem_label = sem_label.reshape(-1,1)
        label_save_path1 =  os.path.join(dst_folder,'labels', all_files[i].split('/')[-1].replace('bin', 'label'))
        if not os.path.exists(os.path.dirname(label_save_path1)):
            os.makedirs(os.path.dirname(label_save_path1))
        sem_label = sem_label.astype(np.uint32)
        sem_label.tofile(label_save_path1)

    n = len(all_files)

    with mp.Pool(args.n_cpus) as pool:

        l = list(tqdm(pool.imap(_map, range(n)), total=n))