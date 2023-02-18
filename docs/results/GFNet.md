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


## GFNet

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 46.57 | 46.33 | 33.23 | 42.04 | 131.34 | 66.73 |
| Wet Ground      | 59.30 | 55.89 | 54.53 | 56.57 | 94.39  | 89.79 |
| Snow            | 55.76 | 56.81 | 57.57 | 56.71 | 92.66  | 90.02 |
| Motion Blur     | 60.25 | 58.54 | 56.97 | 58.59 | 61.73  | 93.00 |
| Beam Missing    | 60.91 | 57.01 | 52.93 | 56.95 | 98.56  | 90.40 |
| Crosstalk       | 22.59 | 16.23 | 12.61 | 17.14 | 198.90 | 27.21 |
| Incomplete Echo | 58.70 | 55.55 | 51.43 | 55.23 | 98.24  | 87.67 |
| Cross-Sensor    | 58.40 | 52.70 | 37.35 | 49.48 | 93.64  | 78.54 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 63.00%, $\text{mCE} =$ 108.68%, $\text{mRR} =$ 77.92%.


## References

```bib
@inproceedings{qiu2022gfnet,
  title = {GFNet: Geometric Flow Network for 3D Point Cloud Semantic Segmentation},
  author = {Haibo Qiu and Baosheng Yu and Dacheng Tao},
  booktitle = {Transactions on Machine Learning Research},
  year = {2022},
}
```
