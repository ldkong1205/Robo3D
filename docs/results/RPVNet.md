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


## RPVNet

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 53.12 | 50.60 | 39.19 | 47.64 | 118.65 | 74.73 |
| Wet Ground      | 55.48 | 52.64 | 52.49 | 53.54 | 100.98 | 83.98 |
| Snow            | 51.70 | 51.28 | 50.42 | 51.13 | 104.60 | 80.20 |
| Motion Blur     | 54.99 | 47.17 | 39.72 | 47.29 | 78.58  | 74.18 |
| Beam Missing    | 59.86 | 54.11 | 46.55 | 53.51 | 106.43 | 83.94 |
| Crosstalk       | 26.39 | 22.19 | 19.35 | 22.64 | 185.69 | 35.51 |
| Incomplete Echo | 58.72 | 54.90 | 50.76 | 54.79 | 99.21  | 85.95 |
| Cross-Sensor    | 57.13 | 50.87 | 30.52 | 46.17 | 99.78  | 72.42 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 63.75%, $\text{mCE} =$ 111.74%, $\text{mRR} =$ 73.86%.


## References

```bib
@inproceedings{xu2021rpvnet,
  title = {RPVNet: A Deep and Efficient Range-Point-Voxel Fusion Network for LiDAR Point Cloud Segmentation},
  author = {Xu, Jianyun, Ruixiang Zhang, Jian Dou, Yushi Zhu, Jie Sun, and Shiliang Pu},
  booktitle = {IEEE/CVF International Conference on Computer Vision},
  year = {2021},
}
```
