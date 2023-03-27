__author__ = "Mario Bijelic"
__contact__ = "mario.bijelic@t-online.de"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import os
import csv

import numpy as np
import matplotlib.pyplot as plt


def frenel_equations(ain, nair=1.0003, nw=1.33):
    """
    :param ain:     incident angle
    :param nair:    refractive index air
    :param nw:      refractive index water
    :return:        rs, ts, rp, tp
    """

    # Using Snell's law to calculate the out angle
    a = np.clip(np.sin(ain) * nair / nw, -1, 1)
    # angles affected by total reflection due to brewster

    aout = np.arcsin(a)

    rs = (nair * np.cos(ain) - nw * np.cos(aout)) / (nair * np.cos(ain) + nw * np.cos(aout))
    ts = 2 * nair * np.cos(ain) / (nair * np.cos(ain) + nw * np.cos(aout))

    rp = (nw * np.cos(ain) - nair * np.cos(aout)) / (nw * np.cos(ain) + nair * np.cos(aout))
    tp = 2 * nair * np.cos(ain) / (nw * np.cos(ain) + nair * np.cos(aout))

    return rs, ts, rp, tp, aout


def frenel_equations_power(ain, nair=1.0003, nw=1.33):
    """
    # calculate the power ratio of transmitted laser power
    # Typically the Fresnel Equations are applied for the Light Wave Amplitude. For the power the transmission
       the values have to be transformed. Please see:  https://en.wikipedia.org/wiki/Fresnel_equations#Complex_amplitude_reflection_and_transmission_coefficients
      and the section about power transmission coefficients.
    :param ain:     incident angle
    :param nair:    refractive index air
    :param nw:      refractive index water
    :return:        rs, ts, rp, tp
    """

    # Using Snell's law to calculate the out angle
    a = np.clip(np.sin(ain) * nair / nw, -1, 1)
    # angles affected by total reflection due to brewster

    aout = np.arcsin(a)

    power_fraction_transmittance = np.cos(ain) * nair / nw / np.cos(aout)

    rs = (nair * np.cos(ain) - nw * np.cos(aout)) / (nair * np.cos(ain) + nw * np.cos(aout))
    ts = 2 * nair * np.cos(ain) / (nair * np.cos(ain) + nw * np.cos(aout))

    rp = (nw * np.cos(ain) - nair * np.cos(aout)) / (nw * np.cos(ain) + nair * np.cos(aout))
    tp = 2 * nair * np.cos(ain) / (nw * np.cos(ain) + nair * np.cos(aout))

    rs = rs ** 2
    ts = ts ** 2 / power_fraction_transmittance

    rp = rp ** 2
    tp = tp ** 2 / power_fraction_transmittance

    return rs, ts, rp, tp, aout


def total_reflection_from_ground(ain, nair=1.0003, nw=1.33, rho=0.9):
    """
    Defines a optical transition from air to water.
    :param ain:     incident angle
    :param nair:    refractive index air
    :param nw:      refractive index water
    :param rho:     ground reflectivity
    :return:        rs, ts, rp, tp
    """

    # First reflection parameters from air to water transition
    ras, tas, rap, tap, aaout = frenel_equations_power(ain, nair=nair, nw=nw)
    # Second reflection parameters from water to air transition
    rws, tws, rwp, twp, awout = frenel_equations_power(aaout, nair=nw, nw=nair)

    rs = ras
    ts = tas * rho * tws / (1 - rho * rws)

    rp = rap
    tp = tap * rho * twp / (1 - rho * rwp)

    return rs, ts, rp, tp, aaout


def total_transmittance_from_ground(ain, nair=1.0003, nw=1.33, rho=0.9, water_absorbtion=0.075, water_thickness=0.0025):
    """
    Defines the power transmission to lower the Laser power for a trsition from air to water and backprojection.
    :param ain:                 incident angle
    :param nair:                refractive index air
    :param nw:                  refractive index water
    :param rho:                 ground reflectivity
    :param water_absorbtion:    Komplex absorption within the water slab is neglected in the current implementation
    :param water_thickness:     The absorption along the path lengths in water is currently neglected. As due to the low
                                absorption factor this can be neglected.
    :return:                    Rs, Ts, Rp, Tp
    """
    rs, ts, rp, tp, aaout = total_reflection_from_ground(ain, nair=nair, nw=nw, rho=rho)

    return rs, ts, rp, tp, aaout


def baryometric_sattering(ain, depth=0.002, nair=1.0003, nw=1.33, beta=0.001, height=1.55):
    """
    :param ain:     incident angle
    :param depth:   depth of the water layer
    :param nair:    refractive index air
    :param nw:      refractive index water
    :param beta:    Absorption
    :param height:  Lidar height
    :return: r
    """

    aout = np.arcsin(np.sin(ain) * nair / nw)

    r = np.exp(-2 * beta * depth / np.cos(aout)) * np.cos(ain) ** 2

    return r


def plot_frenel_equations(angles, x_axis, nair=1.0003, nw=1.33, equation=frenel_equations):
    rs, ts, rp, tp, out_angle = equation(angles, nair=nair, nw=nw)

    plt.plot(x_axis, rs, 'r', label='rs')
    plt.plot(x_axis, ts, 'b', label='ts')
    plt.plot(x_axis, rp, 'r', linestyle='dashed', label='rp')
    plt.plot(x_axis, tp, 'b', linestyle='dashed', label='tp')
    plt.title('Single reflection on water surface')
    plt.legend()
    plt.show()


def plot_total_equations(angles, x_axis, nair=1.0003, nw=1.33, rho=0.9):
    rs, ts, rp, tp, out_angle = total_reflection_from_ground(angles, nair=nair, nw=nw, rho=rho)

    power_fraction_transmittance = np.cos(angles) * nair / nw / np.cos(out_angle)

    plt.plot(x_axis, rs, 'r', label='rs')
    plt.plot(x_axis, ts, 'b', label='ts')
    plt.plot(x_axis, rp, 'r',linestyle='dashed', label='rp')
    plt.plot(x_axis, tp, 'b',linestyle='dashed', label='tp')
    plt.title('Power Intensities after laser relection on wet ground')
    plt.plot(x_axis, rp ** 2, 'r', linestyle='dotted', label='lost power due to relfection p Pol.')
    plt.plot(x_axis, rs ** 2, 'r', linestyle='dashed', label='lost power due to relfection s Pol.')
    plt.plot(x_axis, power_fraction_transmittance * tp ** 2, 'g', linestyle='dotted', label='recieved power p Pol.')
    plt.plot(x_axis, power_fraction_transmittance * ts ** 2, 'g', linestyle='dashed', label='recieved power s Pol.')
    plt.plot(x_axis, power_fraction_transmittance * tp ** 2 + rp ** 2, 'b', linestyle='dashed',
             label='total power p Pol.')
    plt.plot(x_axis, power_fraction_transmittance * ts ** 2 + rs ** 2, 'b', linestyle='dashed',
             label='total power s Pol.')
    plt.xlabel('Ground Distance')
    plt.ylabel('Power Intensity')

    plt.legend()
    plt.show()


def plot_total_equations2(angles, x_axis, nair=1.0003, nw=1.33, rho=0.9, save_data=True):
    Rs, Ts, Rp, Tp, out_angle = total_transmittance_from_ground(angles, nair=nair, nw=nw, rho=rho)

    plt.title('Power Intensities after laser relection on wet ground')
    plt.plot(x_axis, Rp, 'r', linestyle='dotted', label='lost power due to relfection p Pol.')
    plt.plot(x_axis, Rs, 'r', linestyle='dashed', label='lost power due to relfection s Pol.')
    plt.plot(x_axis, Tp, 'g', linestyle='dotted', label='recieved power p Pol.')
    plt.plot(x_axis, Ts, 'g', linestyle='dashed', label='recieved power s Pol.')
    plt.plot(x_axis, Tp + Rp, 'b', linestyle='dashed', label='total power p Pol.')
    plt.plot(x_axis, Ts + Rs, 'b', linestyle='dashed', label='total power s Pol.')

    plt.xlabel('Ground Distance')
    plt.ylabel('Power Intensity')

    plt.legend()
    plt.show()
    if save_data:
        write_csv('fresnel', x_axis, Rs.tolist(), 'relectivity_s')
        write_csv('fresnel', x_axis, Ts.tolist(), 'transmission_s')
        write_csv('fresnel', x_axis, Rp.tolist(), 'relectivity_p')
        write_csv('fresnel', x_axis, Tp.tolist(), 'transmission_p')


def write_csv(name, x_data, y_data, label_file='undefined', folder='statistics_output'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open('%s/%s_%s.csv' % (folder, label_file, name), 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        x_data_append = ['x_data']
        x_data_append.extend(x_data)
        y_data_append = ['y_data'] + y_data
        for x, y in zip(x_data_append, y_data_append):
            spamwriter.writerow([x, y])


def plot_total_reflection_from_ground_angles(scope=(0, np.pi / 2), num=1000, nair=1.0003, nw=1.33, rho=0.05):
    angles = np.linspace(scope[0], scope[1], num=num)
    plot_total_equations2(angles, angles, nair=nair, nw=nw, rho=rho)


def plot_total_reflection_from_ground_distance(scope=(0, 80), num=100, height=1.55, nair=1.0003, nw=1.33, rho=0.05):
    distances = np.linspace(scope[0], scope[1], num=num)
    angles = np.arctan(distances / height)
    plot_total_equations2(angles, distances, nair=nair, nw=nw, rho=rho)


def plot_transmission_reflectionfactors_equations(angles, x_axis, nair=1.0003, nw=1.33):
    rs, ts, rp, tp, out_angle = frenel_equations(angles, nair=nair, nw=nw)

    plt.plot(x_axis, rs ** 2, 'r', label='rs')
    plt.plot(x_axis, np.tan(angles) * ts ** 2 / np.tan(out_angle), 'b', label='ts')
    plt.plot(x_axis, rp ** 2, 'r', linestyle='dashed', label='rp')
    plt.plot(x_axis, np.tan(angles) * tp ** 2 / np.tan(out_angle), 'b', linestyle='dashed', label='tp')
    plt.plot(x_axis, np.tan(angles) * tp ** 2 / np.tan(out_angle) + rp ** 2, 'c', linestyle='dotted', label='sump')
    plt.legend()
    plt.show()


def plot_frenel_angles(scope=(0, np.pi / 2), num=100, nair=1.0003, nw=1.33):
    angles = np.linspace(scope[0], scope[1], num=num)
    plot_frenel_equations(angles, angles, nair=nair, nw=nw)


def plot_frenel_power_angles(scope=(0, np.pi / 2), num=100, nair=1.0003, nw=1.33):
    angles = np.linspace(scope[0], scope[1], num=num)
    plot_frenel_equations(angles, angles, nair=nair, nw=nw, equation=frenel_equations_power)


def plot_frenel_distance(scope=(0, 80), num=100, height=1.55, nair=1.0003, nw=1.33):
    distances = np.linspace(scope[0], scope[1], num=num)
    angles = np.arctan(distances / height)
    plot_transmission_reflectionfactors_equations(angles, distances, nair=nair, nw=nw)


def plot_barymetric_lidar_equation(scope=(0, 80), num=100, height=1.55):
    distances = np.linspace(scope[0], scope[1], num=num)
    angles = np.arctan(distances / height)
    r = baryometric_sattering(angles)

    plt.plot(distances, r, 'g', label='gs')
    plt.show()


def lidar_measurement(distances, height=1.55, Iout=1, rho=0.12, natm=1, C=1, debug=True):
    """
    Formula taken from: https://doi.org/10.1155/2019/8973248
    Simulate recieved intensities based on a flat earth assumption, ground reflectivity and scanner height.
    :param distance: Distance to object
    :param Iout: Sended Laser Intensity
    :param height: Scanner height
    :param rho: reflectivity
    :param natm: athmospheric absorbtion
    :param C: Sensor Paramenets
    :return: Irec: Measured Intensity by scanner
    """
    # asume flat earth
    angles = np.arctan(distances / height)
    Irec = Iout * np.cos(angles) * rho * natm * C
    if debug:
        plt.plot(distances, Irec)
        plt.show()

    return Irec


if __name__ == '__main__':
    # plot_frenel_angles(nair=1.0003, nw=1.5)
    plot_frenel_power_angles()
    # plot_frenel_distance(nair=1.0003, nw=1.33)
    # for rho in range(1,5,1):
    #    plot_total_reflection_from_ground_angles(rho=rho/10, nw=1.003, nair=1.333)
    plot_total_reflection_from_ground_distance(rho=1, nair=1.003, nw=1.333)
    # plot_total_reflection_from_ground_distance(rho=1, nair=1.003, nair=1.333)
    # lidar_measurement(np.linspace(1,100,1000))
    # plot_barymetric_lidar_equation()
