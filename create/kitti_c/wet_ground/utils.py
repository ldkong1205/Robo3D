__author__ = "Mario Bijelic"
__contact__ = "mario.bijelic@t-online.de"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import json

import numpy as np
import matplotlib.pyplot as plt



def plot_2d_hist(var1, var2, binsX=100, binsY=100):
    """
    Taken from https://stackoverflow.com/questions/63415624/normalising-a-2d-histogram
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))

    norm = 10
    ax1.hist2d(var1, var2, bins=(binsX, binsY), cmap='BuPu')
    ax1.set_title('regular 2d histogram')

    hist, xedges, yedges = np.histogram2d(var1, var2, bins=(binsX, binsY))
    hist = hist.T
    with np.errstate(divide='ignore', invalid='ignore'):  # suppress division by zero warnings
        hist *= norm / hist.sum(axis=0, keepdims=True)
    ax2.pcolormesh(xedges, yedges, hist, cmap='BuPu')
    ax2.set_title('normalized columns')
    plt.show()

def load_velodyne_scan(file):
    """Load and parse velodyne binary file"""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 5))  # [:, :4]

def read_meta_label(path):
    with open(path) as k:
        data = json.load(k)
    return data

def filter_weather(meta, weather_type):
    if meta['weather'][weather_type] == True:
        return True
    else:
        return False

def filter_env(meta, env_type):
    if meta['meta']['environment'][env_type] == True:
        return True
    else:
        return False

def filter_daytime(meta, day_type):
    if meta['daytime'][day_type] == True:
        return True
    else:
        return False

def filter_infra(meta, infra_type):
    if meta['meta']['infrastructure'][infra_type] == True:
        return True
    else:
        return False

def read_road_wetness(path):
    try:
        with open(path) as f:
            data = json.load(f)
        # print(data)
        return (float(data['water_thickness']), data['surface_state_result'])
    except:
        pass

def draw_image(pointcloud, map_size=(80, 15), resolution=100, color='depth'):
    # Slow elementwise circle drawing through opencv
    # For quick illustration purposes of example pointclouds.

    image = np.zeros((2*resolution*map_size[1],2*resolution*map_size[0],3)).astype(np.uint8)
    import matplotlib as mpl
    import matplotlib.cm as cm
    import cv2
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.jet
    m = cm.ScalarMappable(norm, cmap)
    if color=='depth':
        values = np.sqrt(np.sum(np.square(pointcloud[:,:3]), axis=1))/80
    elif color=='intensity':
        values = pointcloud[:,3]/255



    depth_map_color = m.to_rgba(values)
    depth_map_color = (255 * depth_map_color).astype(dtype=np.uint8)
    for idx in range(len(values)):
        x, y = resolution*(pointcloud[idx,:2] + [map_size[0], map_size[1]])

        if 0<x<2*resolution*map_size[0] or 0<y<2*resolution*map_size[1]:
            value = depth_map_color[idx]
            tupel_value = (int(value[0]), int(value[1]), int(value[2]))
            #tupel_value = (int(255), int(0), int(0))
            cv2.circle(image, (x.astype(int), y.astype(int)), 20, tupel_value, -1)

    return image