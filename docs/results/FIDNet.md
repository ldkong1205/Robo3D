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


## FIDNet

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 45.49 | 44.98 | 40.51 | 43.66 | 127.67 | 74.25 |
| Wet Ground      | 55.67 | 50.22 | 48.99 | 51.63 | 105.13 | 87.81 |
| Snow            | 48.10 | 49.82 | 51.11 | 49.68 | 107.71 | 84.49 |
| Motion Blur     | 45.18 | 40.37 | 35.59 | 40.38 | 88.88  | 68.67 |
| Beam Missing    | 55.65 | 49.31 | 43.00 | 49.32 | 116.03 | 83.88 |
| Crosstalk       | 51.77 | 49.43 | 47.18 | 49.46 | 121.32 | 84.12 |
| Incomplete Echo | 49.46 | 48.29 | 46.77 | 48.17 | 113.74 | 81.92 |
| Cross-Sensor    | 40.85 | 30.73 | 17.97 | 29.85 | 130.03 | 50.77 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 58.80%, $\text{mCE} =$ 113.81%, $\text{mRR} =$ 76.99%.


### nuScenes-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 66.31 | 65.56 | 62.52 | 64.80
| Wet Ground      | 69.65 | 68.44 | 65.98 | 68.02
| Snow            | 
| Motion Blur     | 58.53 | 48.80 | 39.38 | 48.90
| Beam Missing    | 57.44 | 47.42 | 39.56 | 48.14
| Crosstalk       | 
| Incomplete Echo | 52.08 | 48.47 | 45.73 | 48.76
| Cross-Sensor    | 29.91 | 20.83 | 

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 71.38%, $\text{mCE} =$ %, $\text{mRR} =$ %.


## References

```bib
@inproceedings{zhao2021fidnet,
  title = {FIDNet: LiDAR Point Cloud Semantic Segmentation with Fully Interpolation Decoding},
  author = {Zhao, Yiming and Bai, Lin and Huang, Xinming},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year = {2021},
}
```
