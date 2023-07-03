__author__ = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import itertools
import multiprocessing

import numpy as np

from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from datetime import datetime
from tqdm.contrib.concurrent import process_map


import matplotlib.pyplot as plt

PI = np.pi
SAVE_DIR = str(Path.home() / 'Downloads' / 'snowflakes')


def compute_occupancy(snowfall_rate: float, terminal_velocity: float, snow_density: float=0.1) -> float:
    """
    :param snowfall_rate:           Typically ranges from 0 to 2.5                          [mm/h]
    :param terminal_velocity:       Varies from 0.2 to 2                                    [m/s]
    :param snow_density:            Varies from 0.01 to 0.2 depending on snow wetness       [g/cm³]
    :return:                        Occupancy ratio.
    """
    water_density = 1.0

    return (water_density * snowfall_rate) / ((3.6 * 10 ** 6) * (snow_density * terminal_velocity))


def rainfall_rate_to_snowfall_rate(rainfall_rate: float, terminal_velocity: float,
                                   snowflake_density: float = 0.1, snowflake_diameter: float = 0.003) -> float:
    """
    :param rainfall_rate:       Varies from 0.5 (slight rain) to 50 (violent shower)    [mm/h]
    :param terminal_velocity:   Varies from 0.2 to 2                                    [m/s]
    :param snowflake_density:   Varies from 0.01 to 0.2 depending on snow wetness       [g/cm³]
    :param snowflake_diameter:  Varies from 1 to 10                                     [m]

    :return:
    snowfall_rate:              Typically ranges from 0 to 2.5                          [mm/h]
                                0 - 1       light snow
                                1 - 2.5     moderate snow
                                > 2.5       heavy snow
    """

    snowfall_rate = 487 * snowflake_density * snowflake_diameter * terminal_velocity * (rainfall_rate ** (2/3))

    return snowfall_rate


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


def sekhon_srivastava(precipitation_rate: float) -> float:
    """
    :param precipitation_rate:  in mm/h
    :return:                    in 1/cm
    """
    # Determine rate parameter of distribution of snowflake diameters via formula of Sekhon and Srivastava (1970).
    return 22.9 * precipitation_rate ** -0.45


def gunn_marshall(precipitation_rate: float) -> float:
    """
    :param precipitation_rate:  in mm/h
    :return:                    in 1/cm
    """
    # Determine rate parameter of distribution of snowflake diameters via formula of Marshall and Gunn (1958).
    return 25.5 * precipitation_rate ** -0.48


def dart_throwing(occupancy_ratio: float,
                  precipitation_rate: float,
                  R_0: float,
                  rng: np.random.Generator,
                  distribution: str = 'sekhon_srivastava',
                  show_progessbar: bool = False) -> np.ndarray:
    """
    :param occupancy_ratio:     Ratio of the area of the medium occupied by particles.
    :param precipitation_rate:  Measured in millimeters of equivalent liquid water per hour.
    :param R_0:                 Radius of circular disk that forms the domain of sampling (in meters).
    :param rng:                 Random number generator initialized externally with a random seed.
    :param distribution:        Distribition model of particle diameters.
    :param show_progessbar:     Flag if progressbar should be displayed.

    :return:                    N-by-3 array of sampled particles as disks, where each row contains abscissa and
                                ordinate of disk center and disk radius (in meters).
    """

    if distribution == 'sekhon':
        diameter_rate_parameter = sekhon_srivastava(precipitation_rate)
    elif distribution == 'gunn':
        diameter_rate_parameter = gunn_marshall(precipitation_rate)
    else:
        raise NotImplementedError('Distribution model unknown.')

    diameter_scale_parameter = 1 / diameter_rate_parameter              # in cm

    # Initialize samples to empty set.
    samples = np.zeros((0, 3))

    # Initialize occupied area to 0.
    area_occupied = 0.0

    # Calculate global occupied area across entire domain.
    area_occupied_global = occupancy_ratio * PI * R_0 ** 2

    large_number = 1 / occupancy_ratio
    total = area_occupied_global * large_number + 1

    if show_progessbar:

        pbar = tqdm(total=total, desc='sampling particles',
                    bar_format='{desc}: {percentage:3.0f}%|{bar}|[{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    else:

        pbar = None

    i = 0
    r_avg = 0

    # Main sampling loop.
    while area_occupied < area_occupied_global:

        # Sample center of particle.
        length = np.sqrt(rng.uniform(0, R_0 ** 2))
        angle = rng.uniform(0, 2) * PI

        x = length * np.cos(angle)
        y = length * np.sin(angle)

        particle_diameter = np.inf
        # Sample diameter of particle from exponential distribution (in millimeters).
        while particle_diameter > 20:   # limit diameter to a maximum of 2cm
            particle_diameter = rng.exponential(diameter_scale_parameter * 10)

        # Convert diameter to meters.
        particle_diameter = particle_diameter / 1000

        # Sample height of particle center relative to examined plane.
        height = rng.uniform(-particle_diameter / 2, particle_diameter / 2)

        # Calculate radius of disk that constitutes the intersection of the sampled ball with the examined plane.
        disk_radius = np.sqrt((particle_diameter / 2) ** 2 - height ** 2)

        # If the disk includes the origin, reject the sample and continue.
        if x ** 2 + y ** 2 <= disk_radius ** 2:
            continue

        # Check whether current particle overlaps with any particle that has already been sampled.
        sample_has_overlap = (samples[:, 0] - x) ** 2 + (samples[:, 1] - y) ** 2 <= (samples[:, 2] + disk_radius) ** 2

        # If yes, reject the sample and continue.
        if np.any(sample_has_overlap):
            continue

        else:

            r_avg = (r_avg * i + disk_radius) / (i+1)
            i += 1

            area = PI * disk_radius ** 2
            area_occupied += area
            samples = np.concatenate((samples, np.array([[x, y, disk_radius]])))

            if pbar:
                pbar.update(area * large_number)
                pbar.set_postfix({'n_sampled': len(samples),
                                  'r_avg': r_avg})

    if pbar:
        pbar.n = total
        pbar.close()

    return samples


def incidence_range_empirical_distribution(samples,
                                           R_0,
                                           n_directions,
                                           sample_regular_directions=True,
                                           rng=None):
    """
    :param samples:                     N-by-3 array of sampled particles as disks, where each row contains abscissa
                                        and ordinate of disk center and disk radius (in meters).
    :param R_0:                         Radius of circular disk that forms the domain of sampling (in meters).
    :param n_directions:                number of directions to sample for obtaining an empirical distribution of
                                        incidence range.
    :param sample_regular_directions:   Boolean determining whether to sample directions regularly or uniformly at
                                        random.
    :param rng:                         NumPy random number generator object.

    :return: ranges_travelled:          (n_directions,) NumPy array of ranges that rays have travelled before
                                        intersecting a particle (in meters).
             directions:                directions of rays represented as angles (in the [0, 2 * pi) interval,
                                        in radians).
    """

    n_samples = np.shape(samples)[0]

    # Create NumPy array with directions to be examined represented by angles in the [0, 2 * pi) interval.
    if sample_regular_directions:
        directions = np.linspace(0, 2*PI, n_directions, endpoint=False)
    else:
        directions = rng.uniform(0, 2*PI, n_directions)

    ranges_travelled = R_0 * np.ones(n_directions)

    for i in range(n_directions):

        direction = directions[i]
        tan_direction = np.tan(direction)

        for j in range(n_samples):
            sample = samples[j]
            x_center, y_center, radius = sample

            if direction == PI/2 or direction == 3*PI/2:
                a, b, c = 1.0, 0.0, 0.0
            else:
                a, b, c = tan_direction, -1.0, 0.0

            # Compute distance of current line from the center of the current particle.
            distance_from_sample_to_line = np.abs(a * x_center + b * y_center + c) / np.sqrt(a ** 2 + b ** 2)

            # Check whether line that corresponds to current direction incides on current particle.
            if distance_from_sample_to_line <= radius:
                # Solve the system of (line, circle) to identify their two points of intersection.
                if direction == PI/2 or direction == 3*PI/2:
                    x_intersection = np.array([0.0, 0.0])
                    y_intersection = np.array([y_center + np.sqrt(radius ** 2 - x_center ** 2),
                                               y_center - np.sqrt(radius ** 2 - x_center ** 2)])
                else:
                    discriminant = (x_center + y_center * tan_direction) ** 2\
                                   - (1 + tan_direction ** 2) * (x_center ** 2 + y_center ** 2 - radius ** 2)
                    x_intersection = np.array([(x_center + y_center * tan_direction + np.sqrt(discriminant)) /
                                               (1 + tan_direction ** 2),
                                               (x_center + y_center * tan_direction - np.sqrt(discriminant)) /
                                               (1 + tan_direction ** 2)])
                    y_intersection = tan_direction * x_intersection

                # Keep the solution with the smallest distance from the origin.
                ind_min = np.argmin(np.sum(np.array([x_intersection, y_intersection]) ** 2, axis=0))
                x_intersection_single = x_intersection[ind_min]
                y_intersection_single = y_intersection[ind_min]

                # Check whether current direction corresponds to the part of the line that intersects with the current
                # particle.
                phi_intersection = np.arctan2(y_intersection_single, x_intersection_single)
                if np.cos(phi_intersection) * np.cos(direction) > 0 or np.sin(phi_intersection) * np.sin(direction) > 0:
                    # If yes, proceed with update of range for current direction if necessary. Otherwise disregard the
                    # current particle.
                    distance_from_origin_to_intersection =\
                        np.sqrt(np.sum(x_intersection_single ** 2 + y_intersection_single ** 2))
                    if distance_from_origin_to_intersection < ranges_travelled[i]:
                        ranges_travelled[i] = distance_from_origin_to_intersection

    return ranges_travelled, directions


def save_plot(samples: np.ndarray, R_0: float, string: str, scale_factor: int = 500,
              show_progessbar: bool = False) -> None:

    fig, ax = plt.subplots(figsize=(R_0, R_0))

    if scale_factor == 1:
        filename = f'{string}.svg'
        plt.title(f'sampled particles', fontsize=150)
    else:
        assert scale_factor > 1, 'scale_factor has to be bigger than or equal to 1'
        filename = f'{string}_({scale_factor}x_increased).svg'
        plt.title(f'sampled particles (radius {scale_factor}x increased)', fontsize=150)

    plt.xlim([-R_0, R_0])
    plt.ylim([-R_0, R_0])

    if scale_factor <= 100:
        plt.xticks(np.arange(-R_0, R_0+1, step=1), fontsize=10)
        plt.yticks(np.arange(-R_0, R_0+1, step=1), fontsize=10)
        plt.grid(color = 'red', linewidth = 0.1)
    else:
        plt.xticks(np.arange(-R_0, R_0+1, step=10), fontsize=100)
        plt.yticks(np.arange(-R_0, R_0+1, step=10), fontsize=100)

    plt.xlabel('x (m)', fontsize=150)
    plt.ylabel('y (m)', fontsize=150)

    origin = plt.Circle((0, 0), 0.1, color='red')
    ax.add_patch(origin)

    if show_progessbar:
        pbar = tqdm(samples, desc='creating plot')
    else:
        pbar = samples

    for sample in pbar:
        x, y, radius = sample
        circle = plt.Circle((x, y), scale_factor * radius)  # original radius is too small to be visualized
        ax.add_patch(circle)

    fig.savefig(f'{SAVE_DIR}/{filename}')


def save_array(samples: np.ndarray, string: str) -> None:

    filepath = f'{SAVE_DIR}/{string}.npy'

    np.save(filepath, samples)


def sampling_exists(name: str) -> bool:

    filepath = Path(f'{SAVE_DIR}/{name}.npy')

    return filepath.is_file()


#A function which will process a tuple of parameters
def do_in_parallel(params: Tuple) -> None:

    dist = params[0]
    rate, ratio = params[1][0], params[1][1]
    line = params[2]

    name = f'{dist}_{rate}_{ratio}_{line}'

    if sampling_exists(name):
        print(f'\n{name} skipped')
    else:
        print(f'\n{name}')

        particles = dart_throwing(occupancy_ratio=ratio, precipitation_rate=rate, R_0=r,
                                  distribution=dist, rng=random_number_generator)

        save_array(samples=particles, string=name)

        if line < 3 and rate > 5:
            save_plot(samples=particles, R_0=r, string=name)


if __name__ == '__main__':

    date_time_obj = datetime.now()
    timestamp_str = date_time_obj.strftime("%Y-%m-%d_%H-%M-%S")

    """
    Rates of rainfall:
    ==================
    - Drizzle                   Very small droplets.
    - Slight (fine) drizzle     Detectable as droplets only on the face, car windscreens and windows.
    - Moderate drizzle          Windows and other surfaces stream with water.
    - Heavy (thick) drizzle     Impairs visibility and is measurable in a raingauge, rates up to 1 mm per hour.

    - Rain                      Drops of appreciable size and may be described as small to large drops.
                                It is possible to have rain drops within drizzle!
    - Slight rain               Less than 0.5 mm per hour.
    - Moderate rain             Greater than 0.5 mm per hour, but less than 4.0 mm per hour.
    - Heavy rain                Greater than 4 mm per hour, but less than 8 mm per hour.
    - Very heavy rain           Greater than 8 mm per hour.
    - Slight shower             Less than 2 mm per hour.
    - Moderate shower           Greater than 2 mm, but less than 10 mm per hour.
    - Heavy shower              Greater than 10 mm per hour, but less than 50 mm per hour.
    - Violent shower            Greater than 50 mm per hour.

    https://water.usgs.gov/edu/activity-howmuchrain-metric.html
    """

    r = 80.0                                                                    # m

    random_number_generator = np.random.default_rng(42)

    # Generate values for each parameter
    r_s_s = np.linspace(0.5, 2.5, 5)                                            # mm/h
    v_s_s = np.linspace(0.2, 2, 10)                                             # m/s

    p = list(itertools.product(r_s_s, v_s_s))

    r_r_s = [snowfall_rate_to_rainfall_rate(r_s, v_s) for r_s, v_s in p]        # mm/h
    ratios = [compute_occupancy(r_s, v_s) for r_s, v_s in p]

    m = ['gunn', 'sekhon']

    runs = np.column_stack((r_r_s, ratios))
    runs = runs[runs[:, 1].argsort()]
    runs = runs[::-1]

    n = range(1, 65)

    # Generate a list of tuples where each tuple is a combination of parameters.
    # The list will contain all possible combinations of parameters.
    paramlist = list(itertools.product(m, runs, n))

    # Distribute the parameter sets evenly across the cores
    process_map(do_in_parallel, paramlist, max_workers=multiprocessing.cpu_count(), chunksize=1)
