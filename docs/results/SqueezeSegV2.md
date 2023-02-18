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


## SqueezeSegV2

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 27.12 | 25.61 | 24.18 | 25.64 | 168.50 | 62.11 |
| Wet Ground      | 38.65 | 33.51 | 32.89 | 35.02 | 141.23 | 84.84 |
| Snow            | 26.44 | 27.86 | 28.95 | 27.75 | 154.64 | 67.22 |
| Motion Blur     | 24.86 | 22.63 | 20.77 | 22.75 | 115.16 | 55.11 |
| Beam Missing    | 37.78 | 32.21 | 26.58 | 32.19 | 155.24 | 77.98 |
| Crosstalk       | 29.06 | 26.56 | 24.42 | 26.68 | 176.00 | 64.63 |
| Incomplete Echo | 35.05 | 34.06 | 32.28 | 33.80 | 145.27 | 81.88 |
| Cross-Sensor    | 15.99 | 11.08 | 8.27  | 11.78 | 163.52 | 28.54 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 41.28%, $\text{mCE} =$ 152.45%, $\text{mRR} =$ 65.29%.


## References

```bib
@inproceedings{wu2017squeezeseg,
  title = {SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud},
  author = {Wu, Bichen and Zhou, Xuanyu and Zhao, Sicheng and Yue, Xiangyu and Keutzer, Kurt},
  booktitle = {IEEE International Conference on Robotics and Automation},
  year = {2019},
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
