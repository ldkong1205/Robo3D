__author__ = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import copy
import yaml
import random
import itertools
import functools
from pathlib import Path
import argparse
import os 
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor
import numpy as np
import multiprocessing as mp
from typing import Dict, List, Tuple
from tqdm.contrib.concurrent import process_map
from scipy.constants import speed_of_light as c     # in m/s
from scipy.stats import linregress
import multiprocessing as mp
import geometry as g
import random
seed = 1205
random.seed(seed)
np.random.seed(seed)
PI = np.pi
DEBUG = False
EPSILON = np.finfo(float).eps


def compute_occupancy(snowfall_rate: float, terminal_velocity: float, snow_density: float=0.1) -> float:
    """
    :param snowfall_rate:           Typically ranges from 0 to 2.5                          [mm/h]
    :param terminal_velocity:       Varies from 0.2 to 2                                    [m/s]
    :param snow_density:            Varies from 0.01 to 0.2 depending on snow wetness       [g/cm³]
    :return:                        Occupancy ratio.
    """
    water_density = 1.0

    return (water_density * snowfall_rate) / ((3.6 * 10 ** 6) * (snow_density * terminal_velocity))

def snowfall_rate_to_rainfall_rate(snowfall_rate: float, terminal_velocity: float,
                                   snowflake_density: float = 0.1, snowflake_diameter: float = 0.003) -> float:
    """
    :param snowfall_rate:       Typically ranges from 0 to 2.5                          [mm/h]
    :param terminal_velocity:   Varies from 0.2 to 2                                    [m/s]
    :param snowflake_density:   Varies from 0.01 to 0.2 depending on snow wetness       [g/cm^3]
    :param snowflake_diameter:  Varies from 1 to 10                                     [m]

    :return:
    rainfall_rate:              Varies from 0.5 (slight rain) to 50 (violent shower)    [mm/h]
    """

    rainfall_rate = np.sqrt((snowfall_rate / (487 * snowflake_density * snowflake_diameter * terminal_velocity))**3)

    return rainfall_rate

def estimate_laser_parameters(pointcloud_planes, calculated_indicent_angle, power_factor=15, noise_floor=0.7,
                              debug=True, estimation_method='linear'):
    """
    :param pointcloud_planes: Get all points which correspond to the ground
    :param calculated_indicent_angle: The calculated incident angle for each individual point
    :param power_factor: Determines, how much more Power is available compared to a groundplane reflection.
    :param noise_floor: What are the minimum intensities that could be registered
    :param debug: Show additional Method
    :param estimation_method: Method to fit to outputted laser power.
    :return: Fits the laser outputted power level and noiselevel for each point based on the assumed ground floor reflectivities.
    """
    # normalize intensitities
    normalized_intensitites = pointcloud_planes[:, 3] / np.cos(calculated_indicent_angle)
    distance = np.linalg.norm(pointcloud_planes[:, :3], axis=1)

    # linear model
    p = None
    stat_values = None
    if len(normalized_intensitites) < 3:
        return None, None, None, None
    if estimation_method == 'linear':
        reg = linregress(distance, normalized_intensitites)
        w = reg[0]
        h = reg[1]
        p = [w, h]
        stat_values = reg[2:]
        relative_output_intensity = power_factor * (p[0] * distance + p[1])

    elif estimation_method == 'poly':
        # polynomial 2degre fit
        p = np.polyfit(np.linalg.norm(pointcloud_planes[:, :3], axis=1),
                       normalized_intensitites, 2)
        relative_output_intensity = power_factor * (
                p[0] * distance ** 2 + p[1] * distance + p[2])


    # estimate minimum noise level therefore get minimum reflected intensitites
    hist, xedges, yedges = np.histogram2d(distance, normalized_intensitites, bins=(50, 2555),
                                          range=((10, 70), (5, np.abs(np.max(normalized_intensitites)))))
    idx = np.where(hist == 0)
    hist[idx] = len(pointcloud_planes)
    ymins = np.argpartition(hist, 2, axis=1)[:, 0]
    min_vals = yedges[ymins]
    idx = np.where(min_vals > 5)
    min_vals = min_vals[idx]
    idx1 = [i + 1 for i in idx]
    x = (xedges[idx] + xedges[idx1]) / 2

    if estimation_method == 'poly':
        pmin = ransac_polyfit(x, min_vals, order=2)
        adaptive_noise_threshold = noise_floor * (
                pmin[0] * distance ** 2 + pmin[1] * distance + pmin[2])
    elif estimation_method == 'linear':
        if len(min_vals) > 3:
            pmin = linregress(x, min_vals)
        else:
            pmin = p
        adaptive_noise_threshold = noise_floor * (
                pmin[0] * distance + pmin[1])
    # Guess that noise level should be half the road relfection

    if debug:
        plt.plot(distance, normalized_intensitites, 'x')
        plt.plot(distance, relative_output_intensity, 'x')
        plt.plot(distance, adaptive_noise_threshold, 'x')
        plt.title('Estimated Lidar Parameters')
        plt.ylabel('Intensity')
        plt.xlabel('distance')
        plt.legend(['Input Intensities', 'Total Power', 'Noise Level'])
        plt.show()

    return relative_output_intensity, adaptive_noise_threshold, p, stat_values
def calculate_plane(pointcloud, standart_height=-1.55):
    """
    caluclates plane from loaded pointcloud
    returns the plane normal w and lidar height h.
    :param pointcloud: binary with x,y,z, coordinates
    :return:           w, h
    """

    # Filter points which are close to ground based on mounting position
    valid_loc = (pointcloud[:, 2] < -1.55) & \
                (pointcloud[:, 2] > -1.86 - 0.01 * pointcloud[:, 0]) & \
                (pointcloud[:, 0] > 10) & \
                (pointcloud[:, 0] < 70) & \
                (pointcloud[:, 1] > -3) & \
                (pointcloud[:, 1] < 3)
    # valid_loc = (pointcloud[:, 2] < -1.5) & \
    #         (pointcloud[:, 2] > -3) & \
    #         (pointcloud[:, 0] > -50) & \
    #         (pointcloud[:, 0] < 50) & \
    #         (pointcloud[:, 1] > -50) & \
    #         (pointcloud[:, 1] < 50)
                
    pc_rect = pointcloud[valid_loc]

    if pc_rect.shape[0] <= pc_rect.shape[1]:
        w = [0, 0, 1]
        # Standard height from vehicle mounting position in dense
        h = standart_height
    else:
        try:
            reg = RANSACRegressor(loss='squared_loss', max_trials=1000).fit(pc_rect[:, [0, 1]], pc_rect[:, 2])
            w = np.zeros(3)
            w[0] = reg.estimator_.coef_[0]
            w[1] = reg.estimator_.coef_[1]
            w[2] = -1.0
            h = reg.estimator_.intercept_
            w = w / np.linalg.norm(w)

        except:
            # If error occurs fall back to flat earth assumption
            print('Was not able to estimate a ground plane. Using default flat earth assumption')
            w = [0, 0, 1]
            # Standard height from vehicle mounting position
            h = standart_height

    return w, h


def get_calib(sensor: str = 'hdl64'):
    calib_file = Path(__file__).parent.parent.parent.absolute() / \
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


def process_single_channel(root_path: str, particle_file_prefix: str, orig_pc: np.ndarray, beam_divergence: float,
                           order: List[int], channel_infos: List, channel: int) -> Tuple:
    """
    :param root_path:               Needed for training on GPU cluster.
    :param particle_file_prefix:    Path to file where sampled particles are stored (x, y, r).
    :param orig_pc:                 N-by-5 array containing original pointcloud (x, y, z, intensity, channel).
    :param beam_divergence:         Equivalent to the total beam opening angle (in degree).
    :param order:                   Order of the particle disks.
    :param channel_infos            List of Dicts containing sensor calibration info.

    :param channel:                 Number of the LiDAR channel [0, 63].

    :return:                        Tuple of
                                    - intensity_diff_sum,
                                    - idx,
                                    - the augmented points of the current LiDAR channel.
    """
    
    intensity_diff_sum = 0

    index = order[channel]

    min_intensity = 0  #channel_infos[channel].get('min_intensity', 0)  # not all channels contain this info

    focal_distance = channel_infos[channel]['focal_distance'] * 100
    focal_slope = channel_infos[channel]['focal_slope']
    focal_offset = (1 - focal_distance / 13100) ** 2                # from velodyne manual

    particle_file = f'{particle_file_prefix}_{index + 1}.npy'

    channel_mask = orig_pc[:, 4] == channel
    idx = np.where(channel_mask == True)[0]

    pc = orig_pc[channel_mask]
    N = pc.shape[0]

    x, y, z, intensity, _ = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], pc[:, 4]

    distance = np.linalg.norm([x, y, z], axis=0)

    center_angles = np.arctan2(y, x)  # in range [-PI, PI]
    center_angles[center_angles < 0] = center_angles[center_angles < 0] + 2 * PI  # in range [0, 2*PI]

    beam_angles = -np.ones((N, 2))

    beam_angles[:, 0] = center_angles - np.radians(beam_divergence / 2)  # could lead to negative values
    beam_angles[:, 1] = center_angles + np.radians(beam_divergence / 2)  # could lead to values above 2*PI

    # put beam_angles back in range [0, 2*PI]
    beam_angles[beam_angles < 0] = beam_angles[beam_angles < 0] + 2 * PI
    beam_angles[beam_angles > 2 * PI] = beam_angles[beam_angles > 2 * PI] - 2 * PI

    occlusion_list = get_occlusions(beam_angles=beam_angles, ranges_orig=distance, beam_divergence=beam_divergence,
                                    root_path=root_path, particle_file=particle_file)

    lidar_range = 120                       # in meter
    intervals_per_meter = 10                # => 10cm discretization
    beta_0 = 1 * 10 ** -6 / PI
    tau_h = 1e-8                            #  value 10ns taken from HDL64-S1 specsheet

    M = lidar_range * intervals_per_meter

    M_extended = int(np.ceil(M + c * tau_h * intervals_per_meter))
    lidar_range_extended = lidar_range + c * tau_h

    R = np.round(np.linspace(0, lidar_range_extended, M_extended), len(str(intervals_per_meter)))

    for j, beam_dict in enumerate(occlusion_list):

        d_orig = distance[j]
        i_orig = intensity[j]

        if channel in [53, 55, 56, 58]:
            max_intensity = 230
        else:
            max_intensity = 255

        i_adjusted = i_orig - 255 * focal_slope * np.abs(focal_offset - (1 - d_orig/120)**2)
        i_adjusted = np.clip(i_adjusted, 0, max_intensity)      # to make sure we don't get negative values

        CA_P0 = i_adjusted * d_orig ** 2 / beta_0

        if len(beam_dict.keys()) > 1:                           # otherwise there is no snowflake in the current beam

            i = np.zeros(M_extended)

            for key, tuple_value in beam_dict.items():

                if key != -1:                                   # if snowflake
                    i_orig = 0.9 * max_intensity                # set i to 90% reflectivity
                    CA_P0 = i_orig / beta_0                     # and do NOT normalize with original range

                r_j, ratio = tuple_value

                start_index = int(np.ceil(r_j * intervals_per_meter))
                end_index = int(np.floor((r_j + c * tau_h) * intervals_per_meter) + 1)

                for k in range(start_index, end_index):
                    i[k] += received_power(CA_P0, beta_0, ratio, R[k], r_j, tau_h)

            max_index = np.argmax(i)
            i_max = i[max_index]
            d_max = (max_index / intervals_per_meter) - (c * tau_h / 2)

            i_max += max_intensity * focal_slope * np.abs(focal_offset - (1 - d_max/120)**2)
            i_max = np.clip(i_max, min_intensity, max_intensity)

            if abs(d_max - d_orig) < 2 * (1 / intervals_per_meter):  # only alter intensity

                pc[j, 4] = 1

                new_i = int(i_max)

                if new_i > (i_orig + 1) and DEBUG:
                    print(f'\nnew intensity ({new_i}) in channel {channel} bigger than before ({i_orig}) '
                          f'=> clipping to {i_orig}')
                    new_i = np.clip(new_i, min_intensity, i_orig)
                    pc[j, 4] = 0

                intensity_diff_sum += i_orig - new_i

            else:  # replace point of hard target with snowflake

                pc[j, 4] = 2

                d_scaling_factor = d_max / d_orig

                pc[j, 0] = pc[j, 0] * d_scaling_factor
                pc[j, 1] = pc[j, 1] * d_scaling_factor
                pc[j, 2] = pc[j, 2] * d_scaling_factor  

                new_i = int(i_max)

            assert new_i >= 0, f'new intensity is negative ({new_i})'

            clipped_i = np.clip(new_i, min_intensity, max_intensity)

            pc[j, 3] = clipped_i

        else:

            pc[j, 4] = 0

    return intensity_diff_sum, idx, pc


def binary_angle_search(angles: List[float], low: int, high: int, angle: float) -> int:
    """
    Adapted from https://www.geeksforgeeks.org/python-program-for-binary-search

    :param angles:                  List of individual endpoint angles.
    :param low:                     Start index.
    :param high:                    End index.
    :param angle:                   Query angle.

    :return:                        Index of angle if present in list of angles, else -1
    """

    # Check base case
    if high >= low:

        mid = (high + low) // 2

        # If angle is present at the middle itself
        if angles[mid] == angle:
            return mid

        # If angle is smaller than mid, then it can only be present in left sublist
        elif angles[mid] > angle:
            return binary_angle_search(angles, low, mid - 1, angle)

        # Else the angle can only be present in right sublist
        else:
            return binary_angle_search(angles, mid + 1, high, angle)

    else:
        # Angle is not present in the list
        return -1


def compute_occlusion_dict(beam_angles: Tuple[float, float], intervals: np.ndarray,
                           current_range: float, beam_divergence: float) -> Dict:
    """
    :param beam_angles:         Tuple of angles (left, right).
    :param intervals:           N-by-3 array of particle tangent angles and particle distance from origin.
    :param current_range:       Range to the original hard target.
    :param beam_divergence:     Equivalent to the total beam opening angle (in degree).

    :return:
    occlusion_dict:             Dict containing a tuple of the distance and the occluded angle by respective particle.
                                e.g.
                                0: (distance to particle, occlusion ratio [occluded angle / total angle])
                                1: (distance to particle, occlusion ratio [occluded angle / total angle])
                                3: (distance to particle, occlusion ratio [occluded angle / total angle])
                                7: (distance to particle, occlusion ratio [occluded angle / total angle])
                                ...
                                -1: (distance to original target, unocclusion ratio [unoccluded angle / total angle])

                                all (un)occlusion ratios always sum up to the value 1
    """

    try:
        N = intervals.shape[0]
    except IndexError:
        N = 1

    right_angle, left_angle = beam_angles

    # Make everything properly sorted in the corner case of phase discontinuity.
    if right_angle > left_angle:
        right_angle = right_angle - 2*PI
        right_left_order_violated = intervals[:, 0] > intervals[:, 1]
        intervals[right_left_order_violated, 0] = intervals[right_left_order_violated, 0] - 2*PI

    endpoints = sorted(set([right_angle] + list(itertools.chain(*intervals[:, :2])) + [left_angle]))
    diffs = np.diff(endpoints)
    n_intervals = diffs.shape[0]

    assignment = -np.ones(n_intervals)

    occlusion_dict = {}

    for j in range(N):

        a1, a2, distance = intervals[j]

        i1 = binary_angle_search(endpoints, 0, len(endpoints), a1)
        i2 = binary_angle_search(endpoints, 0, len(endpoints), a2)

        assignment_made = False

        for k in range(i1, i2):

            if assignment[k] == -1:
                assignment[k] = j
                assignment_made = True

        if assignment_made:
            ratio = diffs[assignment == j].sum() / np.radians(beam_divergence)
            occlusion_dict[j] = (distance, np.clip(ratio, 0, 1))

    ratio = diffs[assignment == -1].sum() / np.radians(beam_divergence)
    occlusion_dict[-1] = (current_range, np.clip(ratio, 0, 1))

    return occlusion_dict


def get_occlusions(beam_angles: np.ndarray, ranges_orig: np.ndarray, root_path: str, particle_file: str,
                   beam_divergence: float) -> List:
    """
    :param beam_angles:         M-by-2 array of beam endpoint angles, where for each row, the value in the first column
                                is lower than the value in the second column.
    :param ranges_orig:         M-by-1 array of original ranges corresponding to beams (in m).
    :param root_path:           Needed for training on GPU cluster.

    :param particle_file:       Path to N-by-3 array of all sampled particles as disks,
                                where each row contains abscissa and ordinate of the disk center and disk radius (in m).
    :param beam_divergence:     Equivalent to the opening angle of an individual LiDAR beam (in degree).

    :return:
    occlusion_list:             List of M Dicts.
                                Each Dict contains a Tuple of
                                If key == -1:
                                - distance to the original hard target
                                - angle that is not occluded by any particle
                                Else:
                                - the distance to an occluding particle
                                - the occluded angle by this particle

    """

    M = np.shape(beam_angles)[0]
    # print("M shape is :()".format(M))

    if root_path:
        path = Path(root_path) / 'training' / 'snowflakes' / 'npy' / particle_file
    else:
        path = Path(__file__).parent.absolute() / 'npy' / particle_file

    all_particles = np.load(str(path))
    x, y, _ = all_particles[:, 0], all_particles[:, 1], all_particles[:, 2]

    all_particle_ranges = np.linalg.norm([x, y], axis=0)                                                        # (N,)
    all_beam_limits_a, all_beam_limits_b = g.angles_to_lines(beam_angles)                                       # (M, 2)

    occlusion_list = []

    # Main loop over beams.
    for i in range(M):

        current_range = ranges_orig[i]                                                                          # (K,)

        right_angle = beam_angles[i, 0]
        left_angle = beam_angles[i, 1]

        in_range = np.where(all_particle_ranges < current_range)

        particles = all_particles[in_range]                                                                     # (K, 3)

        x, y, particle_radii = particles[:, 0], particles[:, 1], particles[:, 2]

        particle_angles = np.arctan2(y, x)                                                                      # (K,)
        particle_angles[particle_angles < 0] = particle_angles[particle_angles < 0] + 2 * PI

        tangents_a, tangents_b = g.tangents_from_origin(particles)                                              # (K, 2)

        ################################################################################################################
        # Determine whether centers of the particles lie inside the current beam,
        # which is first sufficient condition for intersection.
        standard_case = np.logical_and(right_angle <= particle_angles, particle_angles <= left_angle)
        seldom_case = np.logical_and.reduce((right_angle - 2 * PI <= particle_angles, particle_angles <= left_angle,
                                             np.full_like(particle_angles, right_angle > left_angle, dtype=bool)))
        seldom_case_2 = np.logical_and.reduce((right_angle <= particle_angles, particle_angles <= left_angle + 2 * PI,
                                               np.full_like(particle_angles, right_angle > left_angle, dtype=bool)))

        center_in_beam = np.logical_or.reduce((standard_case, seldom_case, seldom_case_2))  # (K,)
        ################################################################################################################

        ################################################################################################################
        # Determine whether distances from particle centers to beam rays are smaller than the radii of the particles,
        # which is second sufficient condition for intersection.
        beam_limits_a = all_beam_limits_a[i, np.newaxis].T                                                      # (2, 1)
        beam_limits_b = all_beam_limits_b[i, np.newaxis].T                                                      # (2, 1)
        beam_limits_c = np.zeros((2, 1))  # origin                                                              # (2, 1)

        # Get particle distances to right and left beam limit.
        distances = g.distances_of_points_to_lines(particles[:, :2],
                                                   beam_limits_a, beam_limits_b, beam_limits_c)                 # (K, 2)

        radii_intersecting = distances < np.column_stack((particle_radii, particle_radii))                      # (K, 2)

        intersect_right_ray = g.do_angles_intersect_particles(right_angle, particles[:, 0:2]).T                 # (K, 1)
        intersect_left_ray = g.do_angles_intersect_particles(left_angle, particles[:, 0:2]).T                   # (K, 1)

        right_beam_limit_hit = np.logical_and(radii_intersecting[:, 0], intersect_right_ray[:, 0])
        left_beam_limit_hit = np.logical_and(radii_intersecting[:, 1], intersect_left_ray[:, 0])

        ################################################################################################################
        # Determine whether particles intersect the current beam by taking the disjunction of the above conditions.
        particles_intersect_beam = np.logical_or.reduce((center_in_beam,
                                                         right_beam_limit_hit, left_beam_limit_hit))            # (K,)

        ################################################################################################################

        intersecting_beam = np.where(particles_intersect_beam)

        particles = particles[intersecting_beam]  # (L, 3)
        particle_angles = particle_angles[intersecting_beam]
        tangents_a = tangents_a[intersecting_beam]
        tangents_b = tangents_b[intersecting_beam]
        tangents = (tangents_a, tangents_b)
        right_beam_limit_hit = right_beam_limit_hit[intersecting_beam]
        left_beam_limit_hit = left_beam_limit_hit[intersecting_beam]

        # Get the interval angles from the tangents.
        tangent_angles = g.tangent_lines_to_tangent_angles(tangents, particle_angles)                           # (L, 2)

        # Correct tangent angles that do exceed beam limits.
        interval_angles = g.tangent_angles_to_interval_angles(tangent_angles, right_angle, left_angle,
                                                              right_beam_limit_hit, left_beam_limit_hit)        # (L, 2)

        ################################################################################################################
        # Sort interval angles by increasing distance from origin.
        distances_to_origin = np.linalg.norm(particles[:, :2], axis=1)                                          # (L,)

        intervals = np.column_stack((interval_angles, distances_to_origin))                                     # (L, 3)
        ind = np.argsort(intervals[:, -1])
        intervals = intervals[ind]                                                                              # (L, 3)

        occlusion_list.append(compute_occlusion_dict((right_angle, left_angle),
                                                     intervals,
                                                     current_range,
                                                     beam_divergence))

    return occlusion_list

def get_channel_info(points):
    proj_fov_up =  3.0
    proj_fov_down = -25.0
    proj_H = 64
    # laser parameters
    fov_up = proj_fov_up / 180.0 * np.pi  # field of view up in rad
    fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)  # [m,]: range (depth)

    # get scan components
    scan_x = points[:, 0]  # [m,]
    scan_y = points[:, 1]  # [m,]
    scan_z = points[:, 2]  # [m,]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)  # [m,]
    pitch = np.arcsin(scan_z / depth)  # [m,]

    # get projections in image coords
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_y *= proj_H  # in [0.0, H]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0, H-1]

    channel = proj_y.reshape(-1, 1)
    points = np.concatenate((points, channel), axis=1)
    points[:, 3] = points[:, 3] * 255   # scale intensity to 0 ~ 255
    return points



def augment(pc: np.ndarray, particle_file_prefix: str, beam_divergence: float, shuffle: bool = True,
            show_progressbar: bool=False, only_camera_fov: bool=True, noise_floor: float=0.7,
            root_path: str=None) -> Tuple:
    """
    :param pc:                      N-by-5 array containing original pointcloud (x, y, z, intensity, channel).
    :param particle_file_prefix:    Path to file where sampled particles are stored (x, y, r).
    :param beam_divergence:         Beam divergence in degrees.
    :param shuffle:                 Flag if order of sampled snowflakes should be shuffled.
    :param show_progressbar:        Flag if tqdm should display a progessbar.
    :param only_camera_fov:         Flag if the camera field of view (FOV) filter should be applied.
    :param noise_floor:             Noise floor threshold.
    :param root_path:               Optional root path, needed for training on GPU cluster.

    :return:                        Tuple of
                                    - Tuple of the following statistics
                                        - num_attenuated,
                                        - avg_intensity_diff
                                    - N-by-4 array of the augmented pointcloud.
    """
    pc = get_channel_info(pc)
    assert pc.shape[1] == 5

    w, h = calculate_plane(pc)
    ground = np.logical_and(np.matmul(pc[:, :3], np.asarray(w)) + h < 0.5,
                            np.matmul(pc[:, :3], np.asarray(w)) + h > -0.5)
    pc_ground = pc[ground]

    calculated_indicent_angle = np.arccos(np.divide(np.matmul(pc_ground[:, :3], np.asarray(w)),
                                                    np.linalg.norm(pc_ground[:, :3], axis=1) * np.linalg.norm(w)))

    relative_output_intensity, adaptive_noise_threshold, _, _ = estimate_laser_parameters(pc_ground,
                                                                                          calculated_indicent_angle,
                                                                                          noise_floor=noise_floor,
                                                                                          debug=False)

    adaptive_noise_threshold *= np.cos(calculated_indicent_angle)

    ground_distances = np.linalg.norm(pc_ground[:, :3], axis=1)
    distances = np.linalg.norm(pc[:, :3], axis=1)

    p = np.polyfit(ground_distances, adaptive_noise_threshold, 2)

    relative_output_intensity = p[0] * distances ** 2 + p[1] * distances + p[2]

    orig_pc = copy.deepcopy(pc)
    aug_pc = copy.deepcopy(pc)

    sensor_info = './calib/20171102_64E_S3.yaml'

    with open(sensor_info, 'r') as stream:
        sensor_dict = yaml.safe_load(stream)

    channel_infos = sensor_dict['lasers']
    num_channels = sensor_dict['num_lasers']

    channels = range(num_channels)
    order = list(range(num_channels))
    
    if shuffle:
        random.shuffle(order)

    channel_list = [None] * num_channels
        
    if show_progressbar:

        channel_list[:] = process_map(functools.partial(process_single_channel, root_path, particle_file_prefix,
                                                        orig_pc,beam_divergence, order, channel_infos),
                                      channels, chunksize=4)

    else:

        pool = mp.pool.ThreadPool(mp.cpu_count())

        channel_list[:] = pool.map(functools.partial(process_single_channel, root_path, particle_file_prefix, orig_pc,
                                                     beam_divergence, order, channel_infos), channels)

        pool.close()
        pool.join()

    intensity_diff_sum = 0
    snowflakes_sum = 0

    for item in channel_list:
        tmp_intensity_diff_sum, idx, pc_, p_label = item
        intensity_diff_sum += tmp_intensity_diff_sum
        aug_pc[idx] = pc_

    aug_pc[:, 3] = aug_pc[:, 3]

    scattered = aug_pc[:, 4] == 2
    above_threshold = aug_pc[:, 3] > relative_output_intensity[:]
    scattered_or_above_threshold = np.logical_or(scattered, above_threshold)
    num_removed = np.logical_not(scattered_or_above_threshold).sum()
    aug_pc = aug_pc[np.where(scattered_or_above_threshold)]

    num_attenuated = (aug_pc[:, 4] == 1).sum()

    if num_attenuated > 0:
        avg_intensity_diff = int(intensity_diff_sum / num_attenuated)
    else:
        avg_intensity_diff = 0

    # if only_camera_fov:
    #     calib = get_calib()

    #     pts_rect = calib.lidar_to_rect(aug_pc[:, 0:3])
    #     fov_flag = get_fov_flag(pts_rect, (1024, 1920), calib)

    #     num_removed += np.logical_not(fov_flag).sum()

    #     aug_pc = aug_pc[fov_flag]

    stats = num_attenuated, num_removed, avg_intensity_diff

    return  stats, aug_pc


def received_power(CA_P0: float, beta_0: float, ratio: float, r: float, r_j: float, tau_h: float) -> float:

    answer = ((CA_P0 * beta_0 * ratio * xsi(r_j)) / (r_j ** 2)) * np.sin((PI * (r - r_j)) / (c * tau_h)) ** 2

    return answer

def xsi(R: float, R_1: float = 0.9, R_2: float = 1.0) -> float:

    if R <= R_1:    # emitted ligth beam from the tansmitter is not captured by the receiver

        return 0

    elif R >= R_2:  # emitted ligth beam from the tansmitter is fully captured by the receiver

        return 1

    else:           # emitted ligth beam from the tansmitter is partly captured by the receiver

        m = (1 - 0) / (R_2 - R_1)
        b = 0 - (m * R_1)
        y = m * R + b

        return y



def parse_arguments():

    parser = argparse.ArgumentParser(description='LiDAR snow')

    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default= mp.cpu_count())
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset', type=str,
                        default='./data_root/Kitti/')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset', type=str,
                        default='./save_root/snow/light')  # ['light','moderate','heavy']
    parser.add_argument('-s', '--snowfall_rate', help='snowfall rate', type=str,
                        default= '0.5')  
    parser.add_argument('-t', '--terminal_velocity', help='terminal velocity', type=str,
                        default= '2.0')  

    arguments = parser.parse_args()

    return arguments



if __name__ == '__main__':
    args = parse_arguments()

    print('')
    print(f'using {args.n_cpus} CPUs')
    mode = 'gunn'
    beam_divergence = 0.003  # (rad)
    snowfall_rate = args.snowfall_rate  #'2.5'
    terminal_velocity = args.terminal_velocity #  '1.6'
    # light: snowfall_rate = '0.5' , terminal_velocity = '2.0'; moderate: snowfall_rate = '1.0', terminal_velocity = '1.6'; heavy: snowfall_rate = '2.5', terminal_velocity = '1.6' 
    rain_rate = snowfall_rate_to_rainfall_rate(float(snowfall_rate), float(terminal_velocity))
    occupancy = compute_occupancy(float(snowfall_rate), float(terminal_velocity))
    snowflake_file_prefix = f'{mode}_{rain_rate}_{occupancy}'  
    all_files = []
    src_folder = os.path.join(args.root_folder, 'training/velodyne')
    val_txt = os.path.join(args.root_folder, 'ImageSets/val.txt')
    with open(val_txt, 'r') as f:
        for line in f.readlines():
            all_files.append(os.path.join(src_folder,line.strip()+'.bin'))

    all_paths =  copy.deepcopy(all_files)
    dst_folder = args.dst_folder
    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    for i,_ in enumerate(all_files):
        points = np.fromfile(all_paths[i], dtype=np.float32)
        points = points.reshape((-1, args.n_features))
        assert points is not None

        stats, augmented_pointcloud = augment(points, only_camera_fov=False,
                particle_file_prefix=snowflake_file_prefix, noise_floor=0.7,
                beam_divergence=float(np.degrees(beam_divergence)),
                shuffle=True, show_progressbar=True)
        
        augmented_pointcloud[:, 3] = augmented_pointcloud[:, 3] / 255
        augmented_pointcloud = augmented_pointcloud[:, :4]

        lidar_save_path = os.path.join(dst_folder,'velodyne', all_files[i].split('/')[-1])
        if not os.path.exists(os.path.dirname(lidar_save_path)):
            os.makedirs(os.path.dirname(lidar_save_path))
        augmented_pointcloud.astype(np.float32).tofile(lidar_save_path)



