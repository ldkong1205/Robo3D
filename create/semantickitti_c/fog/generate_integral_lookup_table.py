__author__  = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import pickle
import argparse
import functools

import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from typing import List
from pathlib import Path
from theory import ParameterSet, P_R_fog_soft

from scipy.constants import speed_of_light as c     # in m/s



def parse_arguments():

    parser = argparse.ArgumentParser(description='LiDAR foggification')

    parser.add_argument('-a', '--alphas', help='list of alpha values', type=List[float])
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default=mp.cpu_count())
    parser.add_argument('-r', '--r_0_max', help='maximum range', type=int, default=200)
    parser.add_argument('-n', '--n_steps', help='number of steps in range interval', type=int)
    parser.add_argument('--shift', help='toggle to incorporate shift', type=bool, default=False)
    parser.add_argument('-s', '--save_path', help='path to ',
                        default=str(Path(__file__).parent.absolute() / 'integral_lookup_tables_seg_light_0.008beta/original'))

    arguments = parser.parse_args()

    if arguments.n_steps is None:
        arguments.n_steps = 10 * arguments.r_0_max      # decimeter accuracy

    if arguments.alphas is None:
        arguments.alphas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.06]

    return arguments


def P_R_fog_soft_wrapper(p, R: float, n: int = None) -> float:

    if R > p.r_0:
        return 0            # skip unnecessary computation
    else:
        return P_R_fog_soft(p, R, n)


def generate_integral_lookup_tables(arguments) -> None:

    for alpha in arguments.alphas:

        n = arguments.n_steps
        r_0_max = arguments.r_0_max

        save_path = Path(arguments.save_path)
        pool = mp.Pool(arguments.n_cpus)
        granularity = r_0_max / n

        filename = f'integral_0m_to_{r_0_max}m_stepsize_{granularity}m_tau_h_20ns_alpha_{alpha}.pickle'
        filepath = save_path / filename

        print(f'generating {filepath}')

        p = ParameterSet(n=n, r_range=r_0_max, alpha=alpha)
        print(p.beta * 1)

        integral = {}
        r_0 = 0

        steps = int(r_0_max / granularity)

        for _ in tqdm(range(steps + 1)):

            p.r_0 = round(r_0, 2)

            x_list = np.linspace(0, p.r_range, p.n)
            y_list = pool.map(functools.partial(P_R_fog_soft_wrapper, p), x_list)

            if arguments.shift:
                # shift x so that the peak of the transmitted pulse is at t=0
                # and the peak response of the hard target is at R_0
                x_list = x_list - p.tau_h * c / 2

            argmax = np.argmax(y_list)

            fog_distance = x_list[argmax]
            fog_response = y_list[argmax]

            fog_integral = fog_response / (p.c_a * p.p_0 * p.beta)

            integral[p.r_0] = (fog_distance, fog_integral)

            r_0 += granularity

        with open(filepath, 'wb') as f:
            pickle.dump(integral, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    args = parse_arguments()

    generate_integral_lookup_tables(arguments=args)
