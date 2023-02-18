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
| Fog             | 20.21 | 18.62 | 17.72 | 18.85 | 183.89 | 59.63 |
| Wet Ground      | 29.07 | 26.51 | 26.33 | 27.30 | 158.01 | 86.37 |
| Snow            | 21.79 | 22.74 | 23.57 | 22.70 | 165.45 | 71.81 |
| Motion Blur     | 19.52 | 17.86 | 16.42 | 17.93 | 122.35 | 56.72 |
| Beam Missing    | 28.67 | 24.92 | 21.44 | 25.01 | 171.68 | 79.12 |
| Crosstalk       | 23.08 | 21.58 | 20.30 | 21.65 | 188.07 | 68.49 |
| Incomplete Echo | 28.22 | 27.85 | 26.90 | 27.66 | 158.74 | 87.50 |
| Cross-Sensor    | 10.45 | 8.07  | 5.04  | 7.85  | 170.81 | 24.83 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 31.61%, $\text{mCE} =$ 164.87%, $\text{mRR} =$ 66.81%.


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
