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


## SPVCNN (18_cr_1.0)

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 61.62 | 58.36 | 45.99 | 55.32 | 101.25 | 88.55 |
| Wet Ground      | 58.40 | 52.60 | 50.95 | 53.98 | 100.02 | 86.41 |
| Snow            | 54.32 | 51.64 | 48.29 | 51.42 | 103.98 | 82.31 |
| Motion Blur     | 44.06 | 33.45 | 26.08 | 34.53 | 97.60  | 55.27 |
| Beam Missing    | 60.73 | 57.17 | 52.11 | 56.67 | 99.20  | 90.72 |
| Crosstalk       | 59.52 | 58.26 | 56.53 | 58.10 | 100.58 | 93.00 |
| Incomplete Echo | 58.08 | 54.93 | 50.80 | 54.60 | 99.63  | 87.40 |
| Cross-Sensor    | 57.59 | 51.37 | 28.89 | 45.95 | 100.19 | 73.56 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 62.47%, $\text{mCE} =$ 100.30%, $\text{mRR} =$ 82.15%.


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
