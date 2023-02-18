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


## MinkowskiNet (18_cr_1.0)

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 61.71 | 58.32 | 47.59 | 55.87 | 100.00 | 89.02 |
| Wet Ground      | 58.38 | 52.38 | 51.21 | 53.99 | 100.00 | 86.03 |
| Snow            | 55.50 | 53.70 | 50.65 | 53.28 | 100.00 | 84.89 |
| Motion Blur     | 42.78 | 31.62 | 24.35 | 32.92 | 100.00 | 52.45 |
| Beam Missing    | 60.70 | 56.78 | 51.47 | 56.32 | 100.00 | 89.74 |
| Crosstalk       | 59.87 | 58.47 | 56.68 | 58.34 | 100.00 | 92.96 |
| Incomplete Echo | 57.89 | 54.68 | 50.72 | 54.43 | 100.00 | 86.73 |
| Cross-Sensor    | 57.53 | 52.08 | 28.53 | 46.05 | 100.00 | 73.37 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 62.76%, $\text{mCE} =$ 100.00%, $\text{mRR} =$ 81.90%.


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
