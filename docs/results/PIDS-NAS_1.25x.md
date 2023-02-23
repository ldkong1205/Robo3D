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


## PIDS (NAS 1.25x)

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 56.80 | 48.72 | 38.17 | 47.90 | 118.06 | 75.73 |
| Wet Ground      | 56.23 | 55.38 | 51.84 | 54.48 | 98.94  | 86.13 |
| Snow            | 52.89 | 47.95 | 45.75 | 48.86 | 109.46 | 77.25 |
| Motion Blur     | 32.79 | 21.59 | 14.54 | 22.97 | 114.83 | 36.32 |
| Beam Missing    | 58.34 | 56.27 | 50.18 | 54.93 | 103.18 | 86.85 |
| Crosstalk       | 58.43 | 56.61 | 55.05 | 56.70 | 103.94 | 89.64 |
| Incomplete Echo | 58.57 | 57.20 | 51.66 | 55.81 | 96.97  | 88.24 |
| Cross-Sensor    | 58.95 | 55.22 | 43.99 | 52.72 | 87.64  | 83.35 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 63.25%, $\text{mCE} =$ 104.13%, $\text{mRR} =$ 77.94%.


## References

```bib
@inproceedings{zhang2023pids,
  title = {PIDS: Joint Point Interaction-Dimension Search for 3D Point Cloud},
  author = {Zhang, Tunhou and Ma, Mingyuan and Yan, Feng and Li, Hai and Chen, Yiran},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision},
  year = {2023},
}
```
