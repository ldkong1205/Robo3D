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


## MinkowskiNet (34_cr_1.6)

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 61.84 | 56.66 | 42.12 | 53.54 | 105.28 | 83.94 |
| Wet Ground      | 59.76 | 52.42 | 50.64 | 54.27 | 99.39  | 85.09 |
| Snow            | 53.32 | 50.29 | 46.91 | 50.17 | 106.66 | 78.66 |
| Motion Blur     | 45.62 | 31.44 | 24.33 | 33.80 | 98.69  | 52.99 |
| Beam Missing    | 61.42 | 57.96 | 52.66 | 57.35 | 97.64  | 89.92 |
| Crosstalk       | 60.45 | 58.53 | 56.16 | 58.38 | 99.90  | 91.53 |
| Incomplete Echo | 57.92 | 54.82 | 51.89 | 54.88 | 99.01  | 86.05 |
| Cross-Sensor    | 58.07 | 52.63 | 30.16 | 46.95 | 98.33  | 73.61 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 63.78%, $\text{mCE} =$ 100.61%, $\text{mRR} =$ 80.22%.


### nuScenes-C



## References

```bib
@inproceedings{tang2020searching,
  title = {Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution},
  author = {Tang, Haotian and Liu, Zhijian and Zhao, Shengyu and Lin, Yujun and Lin, Ji and Wang, Hanrui and Han, Song},
  booktitle = {European Conference on Computer Vision}
  year = {2020}
}
```
