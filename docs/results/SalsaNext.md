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


## SalsaNext

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 36.76 | 36.62 | 31.29 | 34.89 | 147.54 | 62.53 |
| Wet Ground      | 52.89 | 46.86 | 45.58 | 48.44 | 112.06 | 86.81 |
| Snow            | 43.80 | 45.77 | 47.08 | 45.55 | 116.55 | 81.63 |
| Motion Blur     | 50.97 | 48.00 | 44.81 | 47.93 | 77.62 | 85.90 |
| Beam Missing    | 54.45 | 50.18 | 44.27 | 49.63 | 115.32 | 88.94 |
| Crosstalk       | 43.34 | 40.17 | 37.12 | 40.21 | 143.52 | 72.06 |
| Incomplete Echo | 52.09 | 48.29 | 43.70 | 48.03 | 114.04 | 86.08 |
| Cross-Sensor    | 53.43 | 50.10 | 30.64 | 44.72 | 102.47 | 80.14 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 55.80%, $\text{mCE} =$ 116.14%, $\text{mRR} =$ 80.51%.


## References

```bib
@inproceedings{cortinhal2020salsanext,
  title = {SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving},
  author = {Tiago Cortinhal and George Tzelepis and Eren Erdal Aksoy},
  booktitle = {Advances in Visual Computing: 15th International Symposium},
  year = {2020},
}
```
