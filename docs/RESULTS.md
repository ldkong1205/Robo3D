<img src="https://github.com/ldkong1205/Robo3D/blob/main/docs/figs/logo2.png" align="right" width="30%">

# Robo3D Benchmark

### Outline
- [Metrics](#metrics)
- [Benchmark](#benchmark)
  - [SemanticKITTI-C](#red_car-semantickitti-c)
  - [nuScenes-C (Seg)]()
  - [WaymoOpen-C (Seg)]()
  - [KITTI-C]()
  - [nuScenes-C (Det)]()
  - [WaymoOpen-C (Det)]()
- [Visualization](#visualization)


## Metrics

### LiDAR Semantic Segmentation

The *mean Intersection-over-Union (mIoU)* is consistently used as the main indicator for evaluating model performance in our  LiDAR semantic segmentation benchmark. The following two metrics are adopted to compare between models' robustness:
- **mCE (the lower the better):** The *average corruption error* (in percentage) of a candidate model compared to the baseline model, which is calculated among all corruption types across three severity levels.
- **mRR (the higher the better):** The *average resilience rate* (in percentage) of a candidate model compared to its "clean" performance, which is calculated among all corruption types across three severity levels.

### 3D Object Detection

The *mean average precision (mAP)* and *nuScenes detection score (NDS)* are consistently used as the main indicator for evaluating model performance in our  LiDAR semantic segmentation benchmark. The following two metrics are adopted to compare between models' robustness:
- **mCE (the lower the better):** The *average corruption error* (in percentage) of a candidate model compared to the baseline model, which is calculated among all corruption types across three severity levels.
- **mRR (the higher the better):** The *average resilience rate* (in percentage) of a candidate model compared to its "clean" performance, which is calculated among all corruption types across three severity levels.



## Benchmark

### :red_car:&nbsp; SemanticKITTI-C 

#### Benchmark: IoU (%)

| Model | mCE (%) | mRR (%) | Clean | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| -: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [SqueezeSeg](docs/results/SqueezeSeg.md) | 164.87 | 66.81 | 31.61 | 18.85 | 27.30 | 22.70 | 17.93 | 25.01 | 21.65 | 27.66 | 7.85 |
| [SqueezeSegV2](docs/results/SqueezeSegV2.md) | 152.45 | 65.29 | 41.28 | 25.64 | 35.02 | 27.75 | 22.75 | 32.19 | 26.68 | 33.80 | 11.78 |
| [RangeNet<sub>21</sub>](docs/results/RangeNet-dark21.md) | 136.33 | 73.42 | 47.15 | 31.04 | 40.88 | 37.43 | 31.16 | 38.16 | 37.98 | 41.54 | 18.76 |
| [RangeNet<sub>53</sub>](docs/results/RangeNet-dark21.md) | 130.66 | 73.59 | 50.29 | 36.33 | 43.07 | 40.02 | 30.10 | 40.80 | 46.08 | 42.67 | 16.98 |
| [SalsaNext](docs/results/SalsaNext.md) | 116.14 | 80.51 | 55.80 | 34.89 | 48.44 | 45.55 | 47.93 | 49.63 | 40.21 | 48.03 | 44.72 |
| [FIDNet<sub>34</sub>](docs/results/FIDNet.md) | 113.81 | 76.99 | 58.80 | 43.66 | 51.63 | 49.68 | 40.38 | 49.32 | 49.46 | 48.17 | 29.85 |
| [CENet<sub>34</sub>](docs/results/CENet.md) | 103.41 | 81.29 | 62.55 | 42.70 | 57.34 | 53.64 | 52.71 | 55.78 | 45.37 | 53.40 | 45.84 |
| |
| [KPConv](docs/results/KPConv.md) | 99.54 | 82.90 | 62.17 | 54.46 | 57.70 | 54.15 | 25.70 | 57.35 | 53.38 | 55.64 | 53.91 |
| [PIDS<sub>NAS1.25x</sub>]() | 104.13 | 77.94 | 63.25 | 47.90 | 54.48 | 48.86 | 22.97 | 54.93 | 56.70 | 55.81 | 52.72 |
| [PIDS<sub>NAS2.0x</sub>]()  | 101.20 | 78.42 | 64.55 | 51.19 | 55.97 | 51.11 | 22.49 | 56.95 | 57.41 | 55.55 | 54.27 |
| [WaffleIron](docs/results/WaffleIron.md) | 109.54 | 72.18 | 66.04 | 45.52 | 58.55 | 49.30 | 33.02 | 59.28 | 22.48 | 58.55 | 54.62 |
| |
| [PolarNet](docs/results/PolarNet.md) | 118.56 | 74.98 | 58.17 | 38.74 | 50.73 | 49.42 | 41.77 | 54.10 | 25.79 | 48.96 | 39.44 |
| |
| <sup>:star:</sup>[MinkUNet<sub>18</sub>](docs/results/MinkUNet-18_cr1.0.md) | 100.00 | 81.90 | 62.76 | 55.87 | 53.99 | 53.28 | 32.92 | 56.32 | 58.34 | 54.43 | 46.05 |
| [MinkUNet<sub>34</sub>](docs/results/MinkUNet-34_cr1.6.md) | 100.61 | 80.22 | 63.78 | 53.54 | 54.27 | 50.17 | 33.80 | 57.35 | 58.38 | 54.88 | 46.95 |
| [Cylinder3D<sub>SPC</sub>](docs/results/Cylinder3D.md) | 103.25 | 80.08 | 63.42 | 37.10 | 57.45 | 46.94  | 52.45 | 57.64 | 55.98 | 52.51 | 46.22 |
| [Cylinder3D<sub>TSC</sub>](docs/results/Cylinder3D-TS.md) | 103.13 | 83.90 | 61.00 | 37.11 | 53.40 | 45.39 | 58.64 | 56.81 | 53.59 | 54.88 | 49.62 |
| |
| [SPVCNN<sub>18</sub>](docs/results/SPVCNN-18_cr1.0.md) | 100.30 | 82.15 | 62.47 | 55.32 | 53.98 | 51.42 | 34.53 | 56.67 | 58.10 | 54.60 | 45.95 |
| [SPVCNN<sub>34</sub>](docs/results/SPVCNN-34_cr1.6.md) | 99.16 | 82.01 | 63.22 | 56.53 | 53.68 | 52.35 | 34.39 | 56.76 | 59.00 | 54.97 | 47.07 |
| [RPVNet](docs/results/RPVNet.md) | 111.74 | 73.86 | 63.75 | 47.64 | 53.54 | 51.13 | 47.29 | 53.51 | 22.64 | 54.79 | 46.17 |
| [CPGNet]() | 107.34 | 81.05 | 61.50 | 37.79 | 57.39 | 51.26 | 59.05 | 60.29 | 18.50 | 56.72 | 57.79 |
| [2DPASS](docs/results/DPASS.md) | 106.14 | 77.50 | 64.61 | 40.46 | 60.68 | 48.53 | 57.80 | 58.78 | 28.46 | 55.84 | 50.01 |
| [GFNet](docs/results/GFNet.md) | 108.68 | 77.92 | 63.00 | 42.04 | 56.57 | 56.71 | 58.59 | 56.95 | 17.14 | 55.23 | 49.48 |


#### Benchmark: CE (%)

| Model | mCE (%) | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| -: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [SqueezeSeg](docs/results/SqueezeSeg.md) | 164.87 | 
| [SqueezeSegV2](docs/results/SqueezeSegV2.md) | 152.45 | 
| [RangeNet<sub>21</sub>](docs/results/RangeNet-dark21.md) | 136.33 |
| [RangeNet<sub>53</sub>](docs/results/RangeNet-dark21.md) | 130.66	|
| [SalsaNext](docs/results/SalsaNext.md) | 116.14 | 
| [FIDNet<sub>34</sub>](docs/results/FIDNet.md) | 113.81 | 
| [CENet<sub>34</sub>](docs/results/CENet.md) | 103.41 | 
| |
| [KPConv](docs/results/KPConv.md) | 99.54 | 
| [PIDS<sub>NAS1.25x</sub>]() | 104.13 | 
| [PIDS<sub>NAS2.0x</sub>]()  | 101.20 | 
| [WaffleIron](docs/results/WaffleIron.md) | 109.54 | 
| |
| [PolarNet](docs/results/PolarNet.md) | 118.56 | 
| |
| [MinkUNet<sub>18</sub>](docs/results/MinkUNet-18_cr1.0.md) | 100.00 | 
| [MinkUNet<sub>34</sub>](docs/results/MinkUNet-34_cr1.6.md) | 100.61 | 
| [Cylinder3D<sub>SPC</sub>](docs/results/Cylinder3D.md) | 103.25 | 
| [Cylinder3D<sub>TSC</sub>](docs/results/Cylinder3D-TS.md) | 103.13 | 
| |
| [SPVCNN<sub>18</sub>](docs/results/SPVCNN-18_cr1.0.md) | 100.30 | 
| [SPVCNN<sub>34</sub>](docs/results/SPVCNN-34_cr1.6.md) | 99.16 | 
| [RPVNet](docs/results/RPVNet.md) | 111.74 | 
| [CPGNet]() | 107.34 | 
| [2DPASS](docs/results/DPASS.md) | 106.14 | 
| [GFNet](docs/results/GFNet.md) | 108.68 | 


#### Benchmark: RR (%)

| Model | mRR (%) | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| -: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [SqueezeSeg](docs/results/SqueezeSeg.md) | 66.81 | 
| [SqueezeSegV2](docs/results/SqueezeSegV2.md) | 65.29 | 
| [RangeNet<sub>21</sub>](docs/results/RangeNet-dark21.md) | 73.42 |
| [RangeNet<sub>53</sub>](docs/results/RangeNet-dark21.md) | 73.59 |
| [SalsaNext](docs/results/SalsaNext.md) | 80.51 | 
| [FIDNet<sub>34</sub>](docs/results/FIDNet.md) | 76.99 | 
| [CENet<sub>34</sub>](docs/results/CENet.md) | 81.29 | 
| |
| [KPConv](docs/results/KPConv.md) | 82.90 | 
| [PIDS<sub>NAS1.25x</sub>]() | 77.94 | 
| [PIDS<sub>NAS2.0x</sub>]()  | 78.42 | 
| [WaffleIron](docs/results/WaffleIron.md) | 
| |
| [PolarNet](docs/results/PolarNet.md) | 
| |
| [MinkUNet<sub>18</sub>](docs/results/MinkUNet-18_cr1.0.md) | 
| [MinkUNet<sub>34</sub>](docs/results/MinkUNet-34_cr1.6.md) | 
| [Cylinder3D<sub>SPC</sub>](docs/results/Cylinder3D.md) | 
| [Cylinder3D<sub>TSC</sub>](docs/results/Cylinder3D-TS.md) | 
| |
| [SPVCNN<sub>18</sub>](docs/results/SPVCNN-18_cr1.0.md) | 
| [SPVCNN<sub>34</sub>](docs/results/SPVCNN-34_cr1.6.md) | 
| [RPVNet](docs/results/RPVNet.md) | 
| [CPGNet]() | 
| [2DPASS](docs/results/DPASS.md) | 
| [GFNet](docs/results/GFNet.md) | 



### :blue_car:&nbsp; nuScenes-C (Seg) 

To be updated.


