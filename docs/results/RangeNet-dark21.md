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


## RangeNet (DarkNet21)

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 33.35 | 30.88 | 28.88 | 31.04 | 156.27 | 65.83 |
| Wet Ground      | 45.06 | 39.38 | 38.19 | 40.88 | 128.49 | 86.70 |
| Snow            | 36.15 | 37.44 | 38.69 | 37.43 | 133.93 | 79.38 |
| Motion Blur     | 34.27 | 31.04 | 28.17 | 31.16 | 102.62 | 66.09 |
| Beam Missing    | 44.10 | 38.51 | 31.87 | 38.16 | 141.58 | 80.93 |
| Crosstalk       | 39.66 | 37.99 | 36.28 | 37.98 | 148.87 | 80.55 |
| Incomplete Echo | 42.85 | 41.81 | 39.97 | 41.54 | 128.29 | 88.10 |
| Cross-Sensor    | 27.16 | 21.74 | 7.39  | 18.76 | 150.58 | 39.79 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 47.15%, $\text{mCE} =$ 136.33%, $\text{mRR} =$ 73.42%.


## References

```bib
@inproceedings{milioto2019rangenet,
  title = {RangeNet++: Fast and Accurate LiDAR Semantic Segmentation},
  author = {A. Milioto and I. Vizzo and J. Behley and C. Stachniss},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year = {2019},
}
```
