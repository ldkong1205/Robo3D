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


## SPVCNN (34_cr_1.6)

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 62.15 | 59.05 | 48.39 | 56.53 | 98.50  | 89.42 |
| Wet Ground      | 59.43 | 51.83 | 49.79 | 53.68 | 100.67 | 84.91 |
| Snow            | 54.32 | 52.22 | 50.52 | 52.35 | 101.99 | 82.81 |
| Motion Blur     | 46.67 | 32.10 | 24.40 | 34.39 | 97.81  | 54.40 |
| Beam Missing    | 60.66 | 57.36 | 52.26 | 56.76 | 98.99  | 89.78 |
| Crosstalk       | 60.56 | 59.22 | 57.23 | 59.00 | 98.42  | 93.32 |
| Incomplete Echo | 58.41 | 54.90 | 51.61 | 54.97 | 98.82  | 86.95 |
| Cross-Sensor    | 57.72 | 52.72 | 30.76 | 47.07 | 98.11  | 74.45 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 63.22%, $\text{mCE} =$ 99.16%, $\text{mRR} =$ 82.01%.


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
