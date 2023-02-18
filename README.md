<p align="right">English | <a href="">简体中文</a></p>

<p align="center">
  <img src="docs/figs/logo.png" align="center" width="21.5%">
  
  <h3 align="center"><strong>Robo3D: Towards Robust and Reliable 3D Perception against Common Corruptions</strong></h3>

  <p align="center">
  </p>

</p>

<p align="center">
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-slategray">
  </a>
  
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-lightblue">
  </a>
  
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/Demo-%F0%9F%8E%AC-pink">
  </a>
  
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/%E4%B8%AD%E8%AF%91%E7%89%88-%F0%9F%90%BC-firebrick">
  </a>
</p>



## About
`Robo3D` is an evaluation benchmark heading toward robust and reliable 3D perception in autonomous driving. With it, we probe the robustness of 3D detectors and segmentors under out-of-distribution (OoD) scenarios against corruptions that tend to occur in the real-world environment. Specifically, we consider natural corruptions happen in the following cases:
1. **Adversarial weather conditions**, such as `fog`, `wet ground`, and `snow`;
2. **External disturbances** that are caused by `motion blur` or result in LiDAR `beam missing`;
3. **Internal sensor failure**, including `crosstalk`, possible `incomplete echo`, and `cross-sensor` scenarios.

| | | | |
| :---: | :---: | :---: | :---: |
| <img src="docs/figs/clean.png" width="180" height="80"> | <img src="docs/figs/fog.png" width="180" height="80"> | <img src="docs/figs/wet_ground.png" width="180" height="80"> | <img src="docs/figs/snow.png" width="180" height="80"> |
| **Fog** | **Wet Ground** | **Snow** | **Motion Blur** |
| <img src="docs/figs/motion_blur.png" width="180" height="80"> | <img src="docs/figs/beam_missing.png" width="180" height="80"> | <img src="docs/figs/crosstalk.png" width="180" height="80"> | <img src="docs/figs/incomplete_echo.png" width="180" height="80"> |
| **Beam Missing** | **Crosstalk** | **Incomplete Echo** | **Cross-Sensor** |
| | | | |

Visit our [project page]() to explore more examples. :oncoming_automobile:



## Updates
- [2023.02] - The `KITTI-C`, `SemanticKITTI-C`, `nuScenes-C`, and `WaymoOpen-C` datasets are ready for downloading! See [here](docs/DATA_PREPARE.md) for more details.


## Outline
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Getting Started](#getting-started)
- [Taxonomy](#taxonomy)
- [Model Zoo](#model-zoo)
- [Benchmark](#benchmark)
- [Create Corruption Set](#create-corruption-set)
- [TODO List](#todo-list)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Installation
For details related to installation, kindly refer to [INSTALL.md](docs/INSTALL.md).


## Data Preparation

Kindly refer to [DATA_PREPARE.md](docs/DATA_PREPARE.md) for the details to prepare the <sup>1</sup>`KITTI`, <sup>2</sup>`KITTI-C`, <sup>3</sup>`SemanticKITTI`, <sup>4</sup>`SemanticKITTI-C`, <sup>5</sup>`nuScenes`, <sup>6</sup>`nuScenes-C`, <sup>7</sup>`WaymoOpen`, and <sup>8</sup>`WaymoOpen-C` datasets.


## Getting Started

To learn more usage about this codebase, kindly refer to [GET_STARTED.md](docs/GET_STARTED.md).


## Taxonomy
To be updated.


## Model Zoo

<details open>
<summary>&nbsp<b>LiDAR Semantic Segmentation</b></summary>

> - [x] **[SqueezeSeg](https://arxiv.org/abs/1710.07368), ICRA 2018.** <sup>[**`[Code]`**](https://github.com/BichenWuUCB/SqueezeSeg)</sup>
> - [x] **[SqueezeSegV2](https://arxiv.org/abs/1809.08495), ICRA 2019.** <sup>[**`[Code]`**](https://github.com/xuanyuzhou98/SqueezeSegV2)</sup>
> - [x] **[MinkowskiNet](https://arxiv.org/abs/1904.08755), CVPR 2019.** <sup>[**`[Code]`**](https://github.com/NVIDIA/MinkowskiEngine)</sup>
> - [x] **[RangeNet++](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf), IROS 2019.** <sup>[**`[Code]`**](https://github.com/PRBonn/lidar-bonnetal)</sup>
> - [x] **[KPConv](https://arxiv.org/abs/1904.08889), ICCV 2019.** <sup>[**`[Code]`**](https://github.com/HuguesTHOMAS/KPConv)</sup>
> - [x] **[SalsaNext](https://arxiv.org/abs/2003.03653), ISVC 2020.** <sup>[**`[Code]`**](https://github.com/TiagoCortinhal/SalsaNext)</sup>
> - [ ] **[RandLA-Net](https://arxiv.org/abs/1911.11236), CVPR 2020.** <sup>[**`[Code]`**](https://github.com/QingyongHu/RandLA-Net)</sup>
> - [x] **[PolarNet](https://arxiv.org/abs/2003.14032), CVPR 2020.** <sup>[**`[Code]`**](https://github.com/edwardzhou130/PolarSeg)</sup>
> - [ ] **[3D-MiniNet](https://arxiv.org/abs/2002.10893), IROS 2020.** <sup>[**`[Code]`**](https://github.com/Shathe/3D-MiniNet)</sup>
> - [x] **[SPVCNN](https://arxiv.org/abs/2007.16100), ECCV 2020.** <sup>[**`[Code]`**](https://github.com/mit-han-lab/spvnas)</sup>
> - [x] **[Cylinder3D](https://arxiv.org/abs/2011.10033), CVPR 2021.** <sup>[**`[Code]`**](https://github.com/xinge008/Cylinder3D)</sup>
> - [x] **[FIDNet](https://arxiv.org/abs/2109.03787), IROS 2021.** <sup>[**`[Code]`**](https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI)</sup>
> - [x] **[RPVNet](https://arxiv.org/abs/2103.12978), ICCV 2021.**
> - [x] **[CENet](https://arxiv.org/abs/2207.12691), ICME 2022.** <sup>[**`[Code]`**](https://github.com/huixiancheng/CENet)</sup>
> - [ ] **[CPGNet](https://arxiv.org/abs/2204.09914), ICRA 2022.** <sup>[**`[Code]`**](https://github.com/GangZhang842/CPGNet)</sup>
> - [x] **[2DPASS](https://arxiv.org/abs/2207.04397), ECCV 2022.** <sup>[**`[Code]`**](https://github.com/yanx27/2DPASS)</sup>
> - [x] **[GFNet](https://arxiv.org/abs/2207.02605), TMLR 2022.** <sup>[**`[Code]`**](https://github.com/haibo-qiu/GFNet)</sup>
> - [ ] **[PCB-RandNet](https://arxiv.org/abs/2209.13797), arXiv 2022.** <sup>[**`[Code]`**](https://github.com/huixiancheng/PCB-RandNet)</sup>
> - [ ] **[PIDS](https://arxiv.org/abs/2211.15759), WACV 2023.** <sup>[**`[Code]`**](https://github.com/lordzth666/WACV23_PIDS-Joint-Point-Interaction-Dimension-Search-for-3D-Point-Cloud)</sup>
> - [x] **[WaffleIron](http://arxiv.org/abs/2301.10100), arXiv 2023.** <sup>[**`[Code]`**](https://github.com/valeoai/WaffleIron)</sup>

</details>


<details open>
<summary>&nbsp<b>3D Object Detection</b></summary>

> - [x] **[SECOND](https://www.mdpi.com/1424-8220/18/10/3337), Sensors 2018.** <sup>[**`[Code]`**](https://github.com/traveller59/second.pytorch)</sup>
> - [x] **[PointPillars](https://arxiv.org/abs/1812.05784), CVPR 2019.** <sup>[**`[Code]`**](https://github.com/nutonomy/second.pytorch)</sup>
> - [x] **[PointRCNN](https://arxiv.org/abs/1812.04244), CVPR 2019.** <sup>[**`[Code]`**](https://github.com/sshaoshuai/PointRCNN)</sup>
> - [x] **[Part-A2](https://arxiv.org/abs/1907.03670), TPAMI 2020.**
> - [x] **[PV-RCNN](https://arxiv.org/abs/1912.13192), CVPR 2020.** <sup>[**`[Code]`**](https://github.com/sshaoshuai/PV-RCNN)</sup>
> - [x] **[CenterPoint](https://arxiv.org/abs/2006.11275), CVPR 2021.** <sup>[**`[Code]`**](https://github.com/tianweiy/CenterPoint)</sup>

</details>


## Benchmark

### LiDAR Semantic Segmentation

The *mean Intersection-over-Union (mIoU)* is consistently used as the main indicator for evaluating model performance in our  LiDAR semantic segmentation benchmark. The following two metrics are adopted to compare between models' robustness:
- **mCE (the lower the better):** The *average corruption error* (in percentage) of a candidate model compared to the baseline model, which is calculated among all corruption types across three severity levels.
- **mRR (the higher the better):** The *average resilience rate* (in percentage) of a candidate model compared to its "clean" performance, which is calculated among all corruption types across three severity levels.


### :red_car:&nbsp; SemanticKITTI-C

| Model | mCE (%) | mRR (%) | Clean | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [SqueezeSeg](docs/results/SqueezeSeg.md) | 164.87 | 66.81 | 31.61 | 18.85 | 27.30 | 22.70 | 17.93 | 25.01 | 21.65 | 27.66 | 7.85 |
| [SqueezeSegV2](docs/results/SqueezeSegV2.md) | 152.45 | 65.29 | 41.28 | 25.64 | 35.02 | 27.75 | 22.75 | 32.19 | 26.68 | 33.80 | 11.78 |
| [RangeNet-21](docs/results/RangeNet-dark21.md) | 136.33 | 73.42 | 47.15 | 31.04 | 40.88 | 37.43 | 31.16 | 38.16 | 37.98 | 41.54 | 18.76 |
| [RangeNet-53](docs/results/RangeNet-dark21.md) | 130.66 | 73.59 | 50.29 | 36.33 | 43.07 | 40.02 | 30.10 | 40.80 | 46.08 | 42.67 | 16.98 |
| [SalsaNext](docs/results/SalsaNext.md) | 116.14 | 80.51 | 55.80 | 34.89 | 48.44 | 45.55 | 47.93 | 49.63 | 40.21 | 48.03 | 44.72 |
| [FIDNet](docs/results/FIDNet.md) |      |      |         |         |         |         |         |         |         |         |
| [CENet](docs/results/CENet.md) | 103.41 | 81.29 | 62.55 | 42.70 | 57.34 | 53.64 | 52.71 | 55.78 | 45.37 | 53.40 | 45.84 |
| |
| [KPConv](docs/results/KPConv.md) | 99.54 | 82.90 | 62.17 | 54.46 | 57.70 | 54.15 | 25.70 | 57.35 | 53.38 | 55.64 | 53.91 |
| [WaffleIron](docs/results/WaffleIron.md) | 109.54 | 72.18 | 66.04 | 45.52 | 58.55 | 49.30 | 33.02 | 59.28 | 22.48 | 58.55 | 54.62 |
| |
| [PolarNet](docs/results/PolarNet.md) | 118.56 | 74.98 | 58.17 | 38.74 | 50.73 | 49.42 | 41.77 | 54.10 | 25.79 | 48.96 | 39.44 |
| |
| [MinkUNet-18_cr1.0](docs/results/MinkUNet-18_cr1.0.md)<sup>:star:</sup> | 100.00 | 81.90 | 62.76 | 55.87 | 53.99 | 53.28 | 32.92 | 56.32 | 58.34 | 54.43 | 46.05 |
| [MinkUNet-34_cr1.6](docs/results/MinkUNet-34_cr1.6.md) | 100.61 | 80.22 | 63.78 | 53.54 | 54.27 | 50.17 | 33.80 | 57.35 | 58.38 | 54.88 | 46.95 |
| [Cylinder3D](docs/results/Cylinder3D.md) |  |  |         |         |         |         |         |         |         |         |
| [Cylinder3D-TS](docs/results/Cylinder3D-TS.md) | 103.13 | 83.90 | 61.00 | 37.11 | 53.40 | 45.39 | 58.64 | 56.81 | 53.59 | 54.88 | 49.62 |
| |
| [SPVCNN-18_cr1.0](docs/results/SPVCNN-18_cr1.0.md) | 100.30 | 82.15 | 62.47 | 55.32 | 53.98 | 51.42 | 34.53 | 56.67 | 58.10 | 54.60 | 45.95 |
| [SPVCNN-34_cr1.6](docs/results/SPVCNN-34_cr1.6.md) | 99.16 | 82.01 | 63.22 | 56.53 | 53.68 | 52.35 | 34.39 | 56.76 | 59.00 | 54.97 | 47.07 |
| [RPVNet](docs/results/RPVNet.md) | 111.74 | 73.86 | 63.75 | 47.64 | 53.54 | 51.13 | 47.29 | 53.51 | 22.64 | 54.79 | 46.17 |
| [2DPASS](docs/results/DPASS.md) | 106.14 | 77.50 | 64.61 | 40.46 | 60.68 | 48.53 | 57.80 | 58.78 | 28.46 | 55.84 | 50.01 |
| [GFNet](docs/results/GFNet.md) | 108.68 | 77.92 | 63.00 | 42.04 | 56.57 | 56.71 | 58.59 | 56.95 | 17.14 | 55.23 | 49.48 |

**Note:** Symbol <sup>:star:</sup> denotes the baseline model adopted in *mCE* calculation.


### :blue_car:&nbsp; nuScenes-C

| Model | mCE (%) | mRR (%) | Clean  | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [FIDNet](docs/results/FIDNet.md) |  |
| [CENet](docs/results/CENet.md) |  |
| |
| [WaffleIron](docs/results/WaffleIron.md) | 
| |
| [PolarNet](docs/results/PolarNet.md) |
| |
| [MinkUNet-18_cr1.0](docs/results/MinkUNet-18_cr1.0.md) |
| [MinkUNet-34_cr1.6](docs/results/MinkUNet-34_cr1.6.md) |
| [Cylinder3D](docs/results/Cylinder3D.md) |
| [Cylinder3D-TS](docs/results/Cylinder3D-TS.md) |
| |
| [SPVCNN-18_cr1.0](docs/results/SPVCNN-18_cr1.0.md) |
| [SPVCNN-34_cr1.6](docs/results/SPVCNN-34_cr1.6.md) |
| [2DPASS](docs/results/DPASS.md) | 
| [GFNet](docs/results/GFNet.md) |

**Note:** Symbol <sup>:star:</sup> denotes the baseline model adopted in *mCE* calculation.


### :taxi:&nbsp; WaymoOpen-C

| Model | mCE (%) | mRR (%) | Clean  | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|                   |         |         |  |    |      |       |      |  |     | |

**Note:** Symbol <sup>:star:</sup> denotes the baseline model adopted in *mCE* calculation.


### 3D Object Detection

The *mean average precision (mAP)* and *nuScenes detection score (NDS)* are consistently used as the main indicator for evaluating model performance in our  LiDAR semantic segmentation benchmark. The following two metrics are adopted to compare between models' robustness:
- **mCE (the lower the better):** The *average corruption error* (in percentage) of a candidate model compared to the baseline model, which is calculated among all corruption types across three severity levels.
- **mRR (the higher the better):** The *average resilience rate* (in percentage) of a candidate model compared to its "clean" performance, which is calculated among all corruption types across three severity levels.


### :red_car:&nbsp; KITTI-C

| Model | mCE (%) | mRR (%) | Clean  | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PointPillars | | | 66.70 | 45.70 |  | | | 52.24 | 
| SECOND<sup>:star:</sup> | | | 68.49 | 53.24 |  |  |  | 54.14 |  |  |
| PointRCNN | | | 70.26 | 56.31 | | | | 56.84 |
| PartA2-Free | | | 76.28 | 58.06 | | | | 59.46 |
| PartA2-Anchor | | | 73.98 | 56.59 | | | | 56.38 | 
| PVRCNN | | | 72.36 | 55.36 | | | | 56.88 | 

**Note:** Symbol <sup>:star:</sup> denotes the baseline model adopted in *mCE* calculation.


### :blue_car:&nbsp; nuScenes-C
| Model | mCE (%) | mRR (%) | Clean  | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PointPillars-MH | 
| SECOND-MH<sup>:star:</sup> | 
| CenterPoint-PP | 
| CenterPoint-LR |
| CenterPoint-HR | 

**Note:** Symbol <sup>:star:</sup> denotes the baseline model adopted in *mCE* calculation.


### :taxi:&nbsp; WaymoOpen-C
| Model | mCE (%) | mRR (%) | Clean  | Fog | Wet Ground | Snow | Motion Blur | Beam Missing | Cross-Talk | Incomplete Echo | Cross-Sensor |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|                   |         |         |  |    |      |       |      |  |     | |

**Note:** Symbol <sup>:star:</sup> denotes the baseline model adopted in *mCE* calculation.


## Create Corruption Set
You can manage to create your own "RoboDet" corrpution sets! Follow the instructions listed in [CREATE.md](docs/CREATE.md).


## TODO List
- [x] Initial release. 🚀
- [x] Add scripts for creating common corruptions.
- [ ] Add download links for corruption sets.
- [ ] Add evaluation scripts on corruption sets.
- [ ] ...


## Citation
If you find this work helpful, please kindly consider citing our paper:

```bibtex
@ARTICLE{robo3d,
  title={Robo3D: Towards Robust and Reliable 3D Perception against Common Corruptions},
  author={xxx},
  journal={arXiv preprint arXiv:23xx.xxxxx}, 
  year={2023},
}
```


## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>, while some specific operations in this codebase might be with other licenses. Please refer to [LICENSE.md](docs/LICENSE.md) for a more careful check, if you are using our code for commercial matters.


## Acknowledgements
We thank the exceptional support from [Shanghai AI Laboratory](https://www.shlab.org.cn/)! Kindly refer to [ACKNOWLEDGE.md](docs/ACKNOWLEDGE.md) for more detailed acknowledgments of this codebase.

<img src="docs/figs/shlab.png" align="center" width="96%">



