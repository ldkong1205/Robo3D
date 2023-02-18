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


## 2DPASS

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 48.89 | 46.70 | 25.78 | 40.46 | | 62.62 |
| Wet Ground      | 63.08 | 59.66 | 59.32 | 60.68 | | 93.92 |
| Snow            | 47.99 | 48.69 | 48.91 | 48.53 | | 75.11 |
| Motion Blur     | 61.49 | 58.14 | 53.79 | 57.80 | | 89.46 |
| Beam Missing    | 63.02 | 59.55 | 53.76 | 58.78 | | 90.98 |
| Crosstalk       | 32.53 | 28.08 | 24.77 | 28.46 | | 44.05 |
| Incomplete Echo | 59.71 | 56.69 | 51.12 | 55.84 | | 86.43 |
| Cross-Sensor    | 60.28 | 54.74 | 35.00 | 50.01 | | 77.40 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 64.61%, $\text{mCE} =$ %, $\text{mRR} =$ 77.50%.


### nuScenes-C



## References

```bib
@inproceedings{yan2022dpass,
  title = {2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds},
  author = {Yan, Xu and Gao, Jiantao and Zheng, Chaoda and Zheng, Chao and Zhang, Ruimao and Cui, Shuguang and Li, Zhen},
  booktitle = {European Conference on Computer Vision},
  year = {2022},
}
```
