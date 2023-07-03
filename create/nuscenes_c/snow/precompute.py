__author__ = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import copy
import numpy as np

from tqdm import tqdm
from pathlib import Path
from simulation import augment
from lib.OpenPCDet.pcdet.utils import calibration_kitti
from sampling import compute_occupancy, snowfall_rate_to_rainfall_rate


SPLIT_FOLDER = Path(__file__).parent.parent.parent.resolve() / 'lib' / 'LiDAR_fog_sim' / 'SeeingThroughFog' / 'splits'
LIDAR_FOLDER = Path.home() / 'datasets' / 'DENSE' / 'SeeingThroughFog' / 'lidar_hdl64_strongest'

SPLIT = SPLIT_FOLDER / 'train_clear.txt'

SNOWFALL_RATES = [0.5, 1.0, 2.0, 2.5, 1.5]       # mm/h
TERMINAL_VELOCITIES = [2.0, 1.6, 2.0, 1.6, 0.6]  # m/s


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def get_calib(sensor: str = 'hdl64'):
    calib_file = Path(__file__).parent.parent.parent.resolve() / \
                 'lib' / 'OpenPCDet' / 'data' / 'dense' / f'calib_{sensor}.txt'
    assert calib_file.exists(), f'{calib_file} not found'
    return calibration_kitti.Calibration(calib_file)


def get_fov_flag(pts_rect, img_shape, calib):

    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag


if __name__ == '__main__':

    assert SPLIT.exists(), f'{SPLIT} does not exist'

    assert len(SNOWFALL_RATES) == len(TERMINAL_VELOCITIES), f'you need to provide an equal amount of ' \
                                                            f'snowfall_rates and terminal velocities'
    rainfall_rates = []
    occupancy_ratios = []

    for j in range(len(SNOWFALL_RATES)):
        rainfall_rates.append(snowfall_rate_to_rainfall_rate(SNOWFALL_RATES[j], TERMINAL_VELOCITIES[j]))
        occupancy_ratios.append(compute_occupancy(SNOWFALL_RATES[j], TERMINAL_VELOCITIES[j]))

    combos = np.column_stack((rainfall_rates, occupancy_ratios))

    sample_id_list = sorted(['_'.join(x.strip().split(',')) for x in open(SPLIT).readlines()])

    reversed_first_half = list(split(sample_id_list, 2))[0]
    reversed_first_half.reverse()
    second_half = list(split(sample_id_list, 2))[1]

    new_list = second_half + reversed_first_half

    for mode in ['gunn', 'sekhon']:

        p_bar = tqdm(new_list, desc=mode)

        for sample_idx in p_bar:

            lidar_file = LIDAR_FOLDER / f'{sample_idx}.bin'

            points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)
            calibration = get_calib()

            for combo in combos:

                rainfall_rate, occupancy_ratio = combo

                save_dir = LIDAR_FOLDER.parent / 'snowfall_simulation' / mode / f'{LIDAR_FOLDER.name}_' \
                                                                                f'rainrate_{int(rainfall_rate)}'
                save_dir.mkdir(parents=True, exist_ok=True)

                save_path = save_dir / f'{sample_idx}.bin'

                if save_path.is_file():
                    continue

                pc = copy.deepcopy(points)

                pts_rectified = calibration.lidar_to_rect(pc[:, 0:3])
                fov_flag = get_fov_flag(pts_rectified, (1024, 1920), calibration)

                pc = pc[fov_flag]

                snowflake_file_prefix = f'{mode}_{rainfall_rate}_{occupancy_ratio}'

                stats, aug_pc = augment(pc=pc, particle_file_prefix=snowflake_file_prefix,
                                        beam_divergence=float(np.degrees(3e-3)))

                aug_pc.astype(np.float32).tofile(save_path)