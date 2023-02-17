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


## Model Name

### SemanticKITTI-C

| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             |
| Wet Ground      |
| Snow            |
| Motion Blur     |
| Beam Missing    |
| Crosstalk       |
| Incomplete Echo |
| Cross-Sensor    |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ %, $\text{mCE} =$ %, $\text{mRR} =$ %.


### nuScenes-C

| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             |
| Wet Ground      |
| Snow            |
| Motion Blur     |
| Beam Missing    |
| Crosstalk       |
| Incomplete Echo |
| Cross-Sensor    |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ %, $\text{mCE} =$ %, $\text{mRR} =$ %.


### WaymoOpen-C

| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             |
| Wet Ground      |
| Snow            |
| Motion Blur     |
| Beam Missing    |
| Crosstalk       |
| Incomplete Echo |
| Cross-Sensor    |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ %, $\text{mCE} =$ %, $\text{mRR} =$ %.


## References

```bib
@article{author20xxmethod,
  title = {Title},
  author = {Author 1, Author 2, Author 3},
  journal = {arXiv preprint arXiv:2xxx.xxxxx},
  year = {20xx},
}
```
