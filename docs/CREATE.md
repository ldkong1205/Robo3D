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
         │   └── WaymoOpen-C                                                                                   
         │── KITTI          
         │── SemanticKITTI       
         │── nuScenes                
         └── WaymoOpen            
```


### nuScenes

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
        ├── Usual files (e.g. attribute.json, calibrated_sensor.json etc.)
        ├── lidarseg.json  <- contains the mapping of each .bin file to the token   
        └── category.json  <- contains the categories of the labels (note that the 
                              category.json from nuScenes v1.0 is overwritten)
```


### nuScenes-C

To be updated.

