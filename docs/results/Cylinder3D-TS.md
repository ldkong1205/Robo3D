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


## Cylinder3D (TorchSparse)

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 42.63 | 39.82 | 28.87 | 37.11 | 142.51 | 60.84 |
| Wet Ground      | 58.22 | 52.18 | 49.79 | 53.40 | 101.28 | 87.54 |
| Snow            | 47.13 | 45.53 | 43.52 | 45.39 | 116.89 | 74.41 |
| Motion Blur     | 59.72 | 58.77 | 57.44 | 58.64 | 61.66  | 96.13 |
| Beam Missing    | 59.65 | 57.03 | 53.74 | 56.81 | 98.88  | 93.13 |
| Crosstalk       | 56.13 | 53.70 | 50.96 | 53.59 | 111.4  | 87.85 |
| Incomplete Echo | 57.95 | 55.92 | 50.78 | 54.88 | 99.01  | 89.97 |
| Cross-Sensor    | 56.37 | 51.96 | 40.53 | 49.62 | 93.38  | 81.34 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 61.00%, $\text{mCE} =$ 103.13%, $\text{mRR} =$ 83.90%.


### nuScenes-C



## References

```bib
@inproceedings{zhu2021cylinder3d,
  title = {Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation},
  author = {Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Ma, Yuexin and Li, Wei and Li, Hongsheng and Lin, Dahua},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition}
  year = {2021}
}
```
