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


## KPConv

### SemanticKITTI-C
| Corruption      | Light | Moderate | Heavy | Average | $\text{CE}_i$ | $\text{RR}_i$ |
| :-------------: | :---: | :------: | :---: | :-----: | :-----------: | :-----------: |
| Fog             | 60.12 | 56.07 | 47.18 | 54.46 | 103.20 | 87.60 |
| Wet Ground      | 59.23 | 57.69 | 56.18 | 57.70 | 91.94  | 92.81 |
| Snow            | 55.04 | 55.08 | 52.34 | 54.15 | 98.14  | 87.10 |
| Motion Blur     | 34.60 | 24.84 | 17.65 | 25.70 | 110.76 | 41.34 |
| Beam Missing    | 59.86 | 59.68 | 52.50 | 57.35 | 97.64  | 92.25 |
| Crosstalk       | 42.36 | 58.16 | 59.63 | 53.38 | 111.91 | 85.86 |
| Incomplete Echo | 58.88 | 55.99 | 52.05 | 55.64 | 97.34  | 89.50 |
| Cross-Sensor    | 61.06 | 50.38 | 50.31 | 53.91 | 85.43  | 86.71 |

- **Summary:** $\text{mIoU}_{\text{clean}} =$ 62.17%, $\text{mCE} =$ 99.54%, $\text{mRR} =$ 82.90%.


## References

```bib
@inproceedings{thomas2019kpconv,
  title = {KPConv: Flexible and Deformable Convolution for Point Clouds},
  author = {Thomas, Hugues and Qi, Charles R. and Deschaud, Jean-Emmanuel and Marcotegui, Beatriz and Goulette, Fran{\c{c}}ois and Guibas, Leonidas J.},
  booktitle = {IEEE/CVF International Conference on Computer Vision},
  year = {2019},
}
```
