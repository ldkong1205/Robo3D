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


## PolarNet

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 42.20 | 41.45 | 32.56 | 38.74 | 138.82 | 66.60 |
| Wet Ground      | 54.74 | 49.09 | 48.37 | 50.73 | 107.09 | 87.21 |
| Snow            | 50.89 | 49.48 | 47.88 | 49.42 | 108.26 | 84.96 |
| Motion Blur     | 48.60 | 41.44 | 35.28 | 41.77 | 86.81  | 71.81 |
| Beam Missing    | 56.92 | 54.30 | 51.09 | 54.10 | 105.08 | 93.00 |
| Crosstalk       | 29.36 | 25.39 | 22.64 | 25.79 | 178.13 | 44.34 |
| Incomplete Echo | 51.42 | 49.04 | 46.41 | 48.96 | 112.00 | 84.17 |
| Cross-Sensor    | 47.01 | 40.77 | 30.54 | 39.44 | 112.25 | 67.80 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 58.17%, $\text{mCE} =$ 118.56%, $\text{mRR} =$ 74.98%.


### nuScenes-C



## References

```bib
@inproceedings{zhang2020polarnet,
  title = {PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation},
  author = {Zhang, Yang and Zhou, Zixiang and David, Philip and Yue, Xiangyu and Xi, Zerong and Gong, Boqing and Foroosh, Hassan},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition}
  year = {2020}
}
```
