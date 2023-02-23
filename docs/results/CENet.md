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


## CENet

### SemanticKITTI-C

| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 45.80 | 44.84 | 37.47 | 42.70 | 129.84 | 68.27 |
| Wet Ground      | 60.67 | 56.35 | 54.99 | 57.34 | 92.72  | 91.67 |
| Snow            | 55.53 | 53.85 | 51.55 | 53.64 | 99.23  | 85.76 |
| Motion Blur     | 56.92 | 52.87 | 48.35 | 52.71 | 70.50  | 84.27 |
| Beam Missing    | 61.40 | 56.67 | 49.28 | 55.78 | 101.24 | 89.18 |
| Crosstalk       | 48.81 | 45.43 | 41.87 | 45.37 | 131.13 | 72.53 |
| Incomplete Echo | 57.77 | 54.25 | 48.19 | 53.40 | 102.26 | 85.37 |
| Cross-Sensor    | 58.16 | 51.34 | 28.01 | 45.84 | 100.39 | 73.29 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 62.55%, $\text{mCE} =$ 103.41%, $\text{mRR} =$ 81.29%.


### nuScenes-C

| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 68.49 | 67.98 | 64.56 | 67.01
| Wet Ground      | 71.51 | 70.23 | 67.86 | 69.87
| Snow            | 
| Motion Blur     | 63.99 | 58.75 | 52.18 | 58.31
| Beam Missing    | 58.57 | 49.11 | 42.23 | 49.97
| Crosstalk       | 
| Incomplete Echo | 56.53 | 52.99 | 50.40 | 53.31
| Cross-Sensor    | 

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 73.28%, $\text{mCE} =$ %, $\text{mRR} =$ %.



## References

```bib
@inproceedings{cheng2022cenet,
  title = {CENet: Toward Concise and Efficient Lidar Semantic Segmentation for Autonomous Driving},
  author = {Cheng, Hui-Xian and Han, Xian-Feng and Xiao, Guo-Qiang},
  booktitle = {IEEE International Conference on Multimedia and Expo},
  year = {2022},
}
```
