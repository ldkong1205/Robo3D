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


## RangeNet (DarkNet53)

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 38.24 | 36.81 | 33.95 | 36.33 | 144.28 | 72.24 |
| Wet Ground      | 46.99 | 41.74 | 40.49 | 43.07 | 123.73 | 85.64 |
| Snow            | 38.75 | 39.98 | 41.34 | 40.02 | 128.38 | 79.58 |
| Motion Blur     | 33.80 | 29.96 | 26.53 | 30.10 | 104.20 | 59.85 |
| Beam Missing    | 47.23 | 41.25 | 33.92 | 40.80 | 135.53 | 81.13 |
| Crosstalk       | 46.85 | 46.09 | 45.29 | 46.08 | 129.43 | 91.63 |
| Incomplete Echo | 43.98 | 42.95 | 41.09 | 42.67 | 125.81 | 84.85 |
| Cross-Sensor    | 23.61 | 19.64 | 7.69  | 16.98 | 153.88 | 33.76 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 50.29%, $\text{mCE} =$ 130.66%, $\text{mRR} =$ 73.59%.


## References

```bib
@inproceedings{milioto2019rangenet,
  title = {RangeNet++: Fast and Accurate LiDAR Semantic Segmentation},
  author = {A. Milioto and I. Vizzo and J. Behley and C. Stachniss},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year = {2019},
}
```
