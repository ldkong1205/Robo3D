<img src="./figs/logo2.png" align="right" width="30%">

# Data Preparation

### Overall Structure

```
└── data 
    └── sets
         │── Robo3D      
         │   │── KITTI-C                
         │   │    │── beam_missing       
         │   │    │── crosstalk
         │   │    │── ...  
         │   │    └── wet_ground
         │   │── SemanticKITTI-C    
         │   │── nuScenes-C
         │   └── WOD-C                                                                                   
         │── KITTI          
         │── SemanticKITTI       
         │── nuScenes                
         └── WOD            
```


## KITTI
To install the [KITTI](https://www.cvlibs.net/datasets/kitti/index.php) dataset, download the data, annotations, and other files from https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d. Unpack the compressed file(s) into `/data/sets/kitti` and re-organize the data structure. Your folder structure should end up looking like this:

```
└── kitti  
    ├── ImageSets
    │       ├── train.txt
    │       └── val.txt
    ├── training
    │       └── calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
    └── testing
            └── calib & velodyne & image_2
```

## KITTI-C

Download the `KITTI-C` dataset at the [OpenDataLab](https://opendatalab.com/) platform and unpack it. Alternatively, you can follow the `create` folder for generation.  Your folder structure should end up looking like this:

```  
└── KITTI-C  
    ├── fog
    │    ├── light
    │    │     └── velodyne         
    │    │         
    │    ├── moderate
    │    └── heavy
    ├── wet_ground
    ├── snow
    ├── motion_blur
    ├── beam_missing
    ├── crosstalk
    ├── incomplete_echo
    └── cross_sensor
```



## SemanticKITTI

To install the [SemanticKITTI](http://semantic-kitti.org/index) dataset, download the data, annotations, and other files from http://semantic-kitti.org/dataset. Unpack the compressed file(s) into `/data/sets/semantickitti` and re-organize the data structure. Your folder structure should end up looking like this:

```
└── semantickitti  
    └── sequences
        ├── velodyne <- contains the .bin files; a .bin file contains the points in a point cloud
        │    └── 00
        │    └── ···
        │    └── 21
        ├── labels   <- contains the .label files; a .label file contains the labels of the points in a point cloud
        │    └── 00
        │    └── ···
        │    └── 10
        ├── calib
        │    └── 00
        │    └── ···
        │    └── 21
        └── semantic-kitti.yaml
```


## SemanticKITTI-C

Download the `SemanticKITTI-C` dataset at the [OpenDataLab](https://opendatalab.com/) platform and unpack it. Alternatively, you can follow the `create` folder for generation.  Your folder structure should end up looking like this:
```  
└── SemanticKITTI-C  
    ├── fog
    │    ├── light
    │    │     ├── velodyne           
    │    │     └── labels    
    │    ├── moderate
    │    └── heavy
    ├── wet_ground
    ├── snow
    ├── motion_blur
    ├── beam_missing
    ├── crosstalk
    ├── incomplete_echo
    └── cross_sensor
```


## nuScenes

To install the [nuScenes](https://www.nuscenes.org/nuscenes) dataset, download the data, annotations, and other files from https://www.nuscenes.org/download. Unpack the compressed file(s) into `/data/sets/nuscenes` and your folder structure should end up looking like this:

```
└── nuscenes  
    ├── Usual nuscenes folders (i.e. samples, sweep)
    │
    ├── lidarseg
    │   └── v1.0-{mini, test, trainval} <- contains the .bin files; a .bin file 
    │                                      contains the labels of the points in a 
    │                                      point cloud (note that v1.0-test does not 
    │                                      have any .bin files associated with it)
    │
    └── v1.0-{mini, test, trainval}
    |   ├── Usual files (e.g. attribute.json, calibrated_sensor.json etc.)
    |   ├── lidarseg.json  <- contains the mapping of each .bin file to the token   
    |   └── category.json  <- contains the categories of the labels (note that the 
    |                         category.json from nuScenes v1.0 is overwritten)
    |
    └──  nuscenes_infos_val.pkl  
```
Notably that nuscenes_infos_val.pkl we follow the data pre-processing in [Cylinder3D](https://github.com/xinge008/Cylinder3D/blob/master/NUSCENES-GUIDE.md) to prepare the nuscenes_info file for nuScenes validation set. You can find nuscenes_infos_val.pkl in [Link](https://drive.google.com/drive/folders/1zSZ9xE4UkKBMCMH0le7KdSxvbyjuuUp8). 

## nuScenes-C

Download the `nuScenes-C` dataset at the [OpenDataLab](https://opendatalab.com/) platform and unpack it. Alternatively, you can follow the `create` folder for generation and you need to download the precomputed snowflake patterns in [Link](https://drive.google.com/drive/folders/1Rx_OBWXBl6OxsHVVtn0YPloUlPlH_TUk?usp=sharing) and put it into `./Robo3D/create/nuscenes_c/snow` folder.  Your folder structure should end up looking like this:

```  
└── nuScenes-C  
    ├── fog
    │    ├── light
    │    │     ├── samples/LIDAR_TOP        
    │    │     └── lidarseg/v1.0-trainval   
    │    ├── moderate
    │    └── heavy
    ├── wet_ground
    ├── snow
    ├── motion_blur
    ├── beam_missing
    ├── crosstalk
    ├── incomplete_echo
    └── cross_sensor
```

## WaymoOpen

Coming soon.


## WaymoOpen-C

Coming soon.



## References

Please note that you should cite the corresponding paper(s) once you use these datasets.


### KITTI

```bibtex
@inproceedings{geiger2012kitti,
    author = {A. Geiger and P. Lenz and R. Urtasun},
    title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {3354--3361},
    year = {2012}
}
```

### SemanticKITTI

```bibtex
@inproceedings{behley2019semantickitti,
    author = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
    title = {SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages = {9297--9307},
    year = {2019}
}
```

### nuScenes
```bibtex
@article{fong2022panopticnuscenes,
    author = {W. K. Fong and R. Mohan and J. V. Hurtado and L. Zhou and H. Caesar and O. Beijbom and A. Valada},
    title = {Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking},
    journal = {IEEE Robotics and Automation Letters},
    volume = {7},
    number = {2},
    pages = {3795--3802},
    year = {2022}
}
```
```bibtex
@inproceedings{caesar2020nuscenes,
    author = {H. Caesar and V. Bankiti and A. H. Lang and S. Vora and V. E. Liong and Q. Xu and A. Krishnan and Y. Pan and G. Baldan and O. Beijbom},
    title = {nuScenes: A Multimodal Dataset for Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {11621--11631},
    year = {2020}
}
```


### Waymo Open

```bibtex
@inproceedings{sun2020waymoopen,
    author = {P. Sun and H. Kretzschmar and X. Dotiwalla and A. Chouard and V. Patnaik and P. Tsui and J. Guo and Y. Zhou and Y. Chai and B. Caine and V. Vasudevan and W. Han and J. Ngiam and H. Zhao and A. Timofeev and S. Ettinger and M. Krivokon and A. Gao and A. Joshi and Y. Zhang and J. Shlens and Z. Chen and D. Anguelov},
    title = {Scalability in Perception for Autonomous Driving: Waymo Open Dataset},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {2446--2454},
    year = {2020}
}
```
