<img src="../figs/logo2.png" align="right" width="30%">

# Robo3D Benchmark

The following metrics are consistently used in our benchmark:

- **Mean Corruption Error (mCE):**
  - The *Corruption Error (CE)* for model $A$ under corruption type $i$ across 3 severity levels is:
  $\text{CE}_i^{\text{Model}A} = \frac{\sum((1 - \text{mIoU})^{\text{Model}A})}{\sum((1 - \text{mIoU})^{\text{Baseline}})}$.
  - The average CE for model $A$ on all $N$ corruption types, i.e., *mCE*, is calculated as: $\text{mCE} = \frac{1}{N}\sum\text{CE}_i$.
  
- **Mean Resilience Rate (mRR):**
  - The *Resilience Rate (RR)* for model $A$ under corruption type $i$ across 3 severity levels is:
  $\text{RR}_i^{\text{Model}A} = \frac{\sum(\text{mIoU}^{\text{Model}A})}{3\times (\text{clean-mIoU}^{\text{Model}A})} .$
  - The average RR for model $A$ on all $N$ corruption types, i.e., *mRR*, is calculated as: $\text{mRR} = \frac{1}{N}\sum\text{RR}_i$.


## SqueezeSeg

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 20.21 | 18.62 | 17.72
| Wet Ground      |
| Snow            |
| Motion Blur     |
| Beam Missing    |
| Crosstalk       |
| Incomplete Echo |

- **Summary:** $\text{mIoU}_{\text{clean}} = 31.61$, $\text{mCE} = $, $\text{mRR} = $.


### nuScenes-C
To be updated.


### WaymoOpen-C
To be updated.


## References

```bib
@inproceedings{wu2017squeezeseg,
  title = {Squeezeseg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud},
  author = {Wu, Bichen and Wan, Alvin and Yue, Xiangyu and Keutzer, Kurt},
  booktitle = {IEEE International Conference on Robotics and Automation},
  year = {2018},
}
```
```bib
@inproceedings{milioto2019rangenet,
  title = {RangeNet++: Fast and Accurate LiDAR Semantic Segmentation},
  author = {A. Milioto and I. Vizzo and J. Behley and C. Stachniss},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year = {2019},
}
```
