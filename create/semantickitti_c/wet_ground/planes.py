__author__ = "Mario Bijelic"
__contact__ = "mario.bijelic@t-online.de"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import os

import numpy as np

from sklearn.linear_model import RANSACRegressor


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


def save_plane(destination_path, file_name, w_in, h_in, projection_matrix):

    w, h_projected = transform_results_to_camera_extrinsics(projection_matrix, w_in, h_in)

    lines = ['# Plane', 'Width 4', 'Height 1']

    plane_file = os.path.join(destination_path, file_name)
    result_lines = lines[:3]
    result_lines.append("{:e} {:e} {:e} {:e}".format(w[0], w[1], w[2], h_projected))
    result_str = '\n'.join(result_lines)
    with open(plane_file, 'w') as f:
        f.write(result_str)


def transform_results_to_camera_extrinsics(projection_matrix, w, h):
    """
    Transform estimated groundplane values to camera coordinates
    :param projection_matrix:  image projection matrix
    :param w: plane normal
    :param h: plane height
    :return: projected plane and height
    """
    w = np.matmul(projection_matrix[0:3, 0:3], np.asarray(w).transpose())
    h = np.matmul(projection_matrix, np.transpose([0, 0, h, 1]))[:3]

    h_projected = np.matmul(w, h)

    return w, h_projected
