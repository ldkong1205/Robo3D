__author__ = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import numpy as np

from copy import deepcopy
from typing import Tuple

PI = np.pi
EPSILON = np.finfo(float).eps


def tangent_angles_to_interval_angles(angles: np.ndarray, right_angle: float, left_angle: float,
                                      right_angle_hit: np.ndarray, left_angle_hit: np.ndarray) -> np.ndarray:
    """
    :param angles:              N-by-2 array containing tangent angles.
    :param right_angle:         Right beam angle.
    :param left_angle:          Left beam angle.
    :param right_angle_hit:     Flag if right beam angle has been exceeded.
    :param left_angle_hit:      Flag if left beam angle has been exceeded.

    :return:                    N-by-2 array of corrected tangent angles that do not exceed beam limits.
    """

    angles[right_angle_hit, 0] = right_angle
    angles[left_angle_hit, 1] = left_angle

    return angles


def tangent_lines_to_tangent_angles(lines: Tuple[np.ndarray, np.ndarray], center_angles: np.ndarray) -> np.ndarray:
    """
    :param lines:               Tuple of two N-by-2 arrays holding the $a$ and $b$ coefficients of the tangents.
    :param center_angles:       N-by-1 array containing the angle to the particle center.

    :return:
    angles:                     N-by-2 array of tangent angles (right angle first, left angle second).
    """

    a_s, b_s = lines

    try:
        N = center_angles.shape[0]
    except IndexError:
        N = 1

    angles = -np.ones((N, 2))                                                                                   # (N, 2)

    ray_1_angles = np.arctan(-a_s/b_s)                                      # in range [-PI/2, PI/2]            # (N, 2)
    ray_2_angles = deepcopy(ray_1_angles) + PI                              # in range [PI/2, 3*PI/2]           # (N, 2)

    # correct value range
    ray_1_angles[ray_1_angles < 0] = ray_1_angles[ray_1_angles < 0] + 2*PI  # in range [0, 2*PI]                # (N, 2)
    ray_1_angles = np.abs(ray_1_angles)                                     # to prevent -0 value

    # catch special case if line is vertical
    ray_1_angles[b_s == 0] = PI/2
    ray_2_angles[b_s == 0] = 3*PI/2

    tangent_1_angles = np.column_stack((ray_1_angles[:, 0], ray_2_angles[:, 0]))                                # (N, 2)
    tangent_2_angles = np.column_stack((ray_1_angles[:, 1], ray_2_angles[:, 1]))                                # (N, 2)

    for i, tangent_angles in enumerate([tangent_1_angles, tangent_2_angles]):

        tangent_difference = tangent_angles - np.column_stack((center_angles, center_angles))                   # (N, 2)

        correct_ray = np.logical_or.reduce((np.abs(tangent_difference) < PI/2,
                                            np.abs(tangent_difference - 2*PI) < PI/2,
                                            np.abs(tangent_difference + 2*PI) < PI/2))                          # (N, 2)

        angles[:, i] = tangent_angles[np.where(correct_ray)]                                                    # (N, 2)

    angles.sort(axis=1)

    # swap order where discontinuity is crossed
    swap = angles[:, 1] - angles[:, 0] > PI
    angles[swap, 0], angles[swap, 1] = angles[swap, 1], angles[swap, 0]

    return angles


def angles_to_lines(angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param angles:              M-by-2 array of angles, where for each row, the value in the first column
                                is lower than the value in the second column.
    :return:
    a_s:                        N-by-2 array holding the $a$ coefficients of the tangent rays of particles passing from
                                the origin. Each row corresponds to a_s particle.
    b_s:                        N-by-2 array holding the $b$ coefficients of the tangent rays of particles passing from
                                the origin. Each row corresponds to a_s particle.
    """

    tan_directions = np.tan(angles)                                                                             # (M, 2)

    directions_vertical = np.logical_or(angles == PI/2, angles == 3 * PI/2)
    directions_not_vertical = np.logical_not(directions_vertical)

    a_s = np.zeros_like(angles)
    b_s = np.zeros_like(angles)

    a_s[np.where(directions_vertical)] = 1
    b_s[np.where(directions_vertical)] = 0

    a_s[np.where(directions_not_vertical)] = -tan_directions[np.where(directions_not_vertical)]
    b_s[np.where(directions_not_vertical)] = 1

    # a_s[np.abs(a_s) < EPSILON] = 0              # to prevent -0 value

    return a_s, b_s


def distances_of_points_to_lines(points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    :param points:      N-by-2 array of points, where each row contains the coordinates (abscissa, ordinate) of a point
    :param a:           M-by-1 array of $a$ coefficients of lines
    :param b:           M-by-1 array of $b$ coefficients of lines
    :param c:           M-by-1 array of $c$ coefficients of lines
                        where ax + by = c

    :return:            N-by-M array containing distances of points to lines
    """

    try:
        N = points.shape[0]
    except IndexError:
        N = 1

    abscissa, ordinate = points[:, 0, np.newaxis], points[:, 1, np.newaxis]

    numerators = np.dot(abscissa, a.T) + np.dot(ordinate, b.T) + np.dot(np.ones((N, 1)), c.T)

    denominators = np.dot(np.ones((N, 1)), np.sqrt(a ** 2 + b ** 2).T)

    return np.abs(numerators / denominators)


def tangents_from_origin(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param samples:             N-by-3 array of sampled particles as disks, where each row contains abscissa and
                                ordinate of disk center and disk radius (in meters).
    :return:
    a:                          N-by-2 array holding the $a$ coefficients of the tangent rays of particles passing from
                                the origin. Each row corresponds to a particle.
    b:                          N-by-2 array holding the $b$ coefficients of the tangent rays of particles passing from
                                the origin. Each row corresponds to a particle.
    """

    # Solve systems of equations that encode the following information:
    # 1) rays include origin,
    # 2) rays are tangent to the circles corresponding to the particles, i.e., they intersect with the circles at
    # exactly one point.

    x_s, y_s, r_s = samples[:, 0], samples[:, 1], samples[:, 2]

    try:
        N = samples.shape[0]
    except IndexError:
        N = 1

    discriminants = r_s * np.sqrt(x_s ** 2 + y_s ** 2 - r_s ** 2)

    case_1 = np.abs(x_s) - r_s == 0  # One of the two lines is vertical.
    case_2 = np.logical_not(case_1)  # Both lines are not vertical.

    a_1_case_1, b_1_case_1 = np.ones(N), np.zeros(N)
    a_2_case_1, b_2_case_1 = (y_s ** 2 - x_s ** 2) / (2 * x_s * y_s), - np.ones(N)

    a_1_case_2 = (-x_s * y_s + discriminants) / (r_s ** 2 - x_s ** 2)
    a_2_case_2 = (-x_s * y_s - discriminants) / (r_s ** 2 - x_s ** 2)
    b_1_case_2 = -np.ones(N)
    b_2_case_2 = -np.ones(N)

    # Compute the coefficients by distinguishing the two cases.
    a_1, a_2, b_1, b_2 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

    a_1[case_1] = a_1_case_1[case_1]
    a_2[case_1] = a_2_case_1[case_1]
    b_1[case_1] = b_1_case_1[case_1]
    b_2[case_1] = b_2_case_1[case_1]

    a_1[case_2] = a_1_case_2[case_2]
    a_2[case_2] = a_2_case_2[case_2]
    b_1[case_2] = b_1_case_2[case_2]
    b_2[case_2] = b_2_case_2[case_2]

    a = np.column_stack((a_1, a_2))
    b = np.column_stack((b_1, b_2))

    return a, b


def do_angles_intersect_particles(angles: np.ndarray, particle_centers: np.ndarray) -> np.ndarray:
    """
    Assumption: either the ray that corresponds to an angle or its opposite ray intersects with all particles.

    :param angles:              (M,) array of angles in the range [0, 2*PI).
    :param particle_centers:    (N, 2) array of particle centers (abscissa, ordinate).

    :return:
    """
    try:
        M = angles.shape[0]
    except IndexError:
        M = 1

    try:
        N = particle_centers.shape[0]
    except IndexError:
        N = 1

    x, y = particle_centers[:, 0], particle_centers[:, 1]

    angle_to_centers = np.arctan2(y, x)
    angle_to_centers[angle_to_centers < 0] = angle_to_centers[angle_to_centers < 0] + 2*PI                      # (N, 1)

    angle_differences = np.tile(angles, (1, N)) - np.tile(angle_to_centers.T, (M, 1))                           # (M, N)

    answer = np.logical_or.reduce((np.abs(angle_differences) < PI/2,
                                   np.abs(angle_differences - 2*PI) < PI/2,
                                   np.abs(angle_differences + 2*PI) < PI/2))                                    # (M, N)

    return answer
