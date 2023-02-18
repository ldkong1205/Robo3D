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


## WaffleIron

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 54.57 | 49.31 | 32.67 | 45.52 | 123.45 | 68.93 |
| Wet Ground      | 62.55 | 59.56 | 53.54 | 58.55 | 90.09  | 88.66 |
| Snow            | 49.60 | 49.41 | 48.88 | 49.30 | 108.52 | 74.65 |
| Motion Blur     | 37.61 | 32.75 | 28.69 | 33.02 | 99.85  | 50.00 |
| Beam Missing    | 63.76 | 59.46 | 54.63 | 59.28 | 93.22  | 89.76 |
| Crosstalk       | 26.66 | 22.02 | 18.76 | 22.48 | 186.08 | 34.04 |
| Incomplete Echo | 62.55 | 59.56 | 53.54 | 58.55 | 90.96  | 88.66 |
| Cross-Sensor    | 63.34 | 59.77 | 40.74 | 54.62 | 84.11  | 82.71 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 66.04%, $\text{mCE} =$ 109.54%, $\text{mRR} =$ 72.18%.


### nuScenes-C



## References

```bib
@article{puy2023waffleiron,
  title = {Using a Waffle Iron for Automotive Point Cloud Semantic Segmentation},
  author = {Puy, Gilles and Boulch, Alexandre and Marlet, Renaud},
  journal = {arXiv preprint arxiv:2301.10100}
  year = {2023}
}
```
