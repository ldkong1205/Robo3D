<img src="https://github.com/ldkong1205/Robo3D/blob/main/docs/figs/logo2.png" align="right" width="30%">

# Getting Started

- [**Model Training**](#model-training)
- [**Robustness Evaluation**](#robustness-evaluation)
  - [**Definition**](#definition)
  - [**Calculation**](#calculation)
- [**Baseline**](#baseline)
- [**Custom Model**](#custom-model)

## Model Training

- This task aims to probe the **out-of-distribution (OoD) generalization** of 3D object detection and LiDAR segmentation models. We assume the training data come from the **original** datasets directly, **without** knowing the distributions of corrupted data beforehand.

- Please follow the conventional way of training your 3D object detection and LiDAR segmentation models.

- Kindly use the original validation sets to select your best possible hyperparameters.

- Do not use corruption simulations to fine-tune your trained models.

## Robustness Evaluation

:warning: **Important:** By default, we evaluate the model's robustness on the corruption sets **without** using extra tricks including model ensemble and test-time-augmentation (TTA). To ensure fair comparisons, we suggest you follow the **same** configuration strictly.

Our generated corruption sets follow the same dataset structure as the original validation sets. You can think of each severity level as one validation set and iteratively validate all three levels from every corruption type.

### Definition

- The **Corruption Error (CE)** for a model $A$ under corruption $i$ across three severity levels is:
  $$\text{CE}\_i^{\text{Model}A} = \frac{\sum\_{l=1}((1 - \text{mIoU})\_{i,l}^{\text{Model}A})}{\sum\_{l=1}((1 - \text{mIoU})\_{i,l}^{\text{Baseline}})} . $$

- The average $\text{CE}$ for a model $A$ on all corruptions, *i.e.*, **mean Corruption Error (mCE)**, is calculated as:
  $$\text{mCE} = \frac{1}{N}\sum^N\_{i=1}\text{CE}\_i , $$
  where $N=8$ denotes the number of corruption types in the Robo3D benchmark.

- The **Resilience Rate (RR)** for a model $A$ under corruption $i$ across three severity levels is:
  $$\text{RR}\_i^{\text{Model}A} = \frac{\sum\_{l=1}(\text{mIoU}\_{i,l}^{\text{Model}A})}{3\times\text{mIoU}\_{\text{clean}}^{\text{Model}A}} . $$

- The average $\text{RR}$ for a model $A$ on all corruptions, *i.e.*, **mean Resilience Rate (mRR)**, is calculated as:
  $$\text{mRR} = \frac{1}{N}\sum^N\_{i=1}\text{RR}\_i , $$
  where $N=8$ denotes the number of corruption types in the Robo3D benchmark.

### Calculation
Follow the code blocks below to calculate the mean Corruption Error (mCE) and mean Reseilence Rate (mRR) scores of your 3D object detection and LiDAR segmentation models.

- **mean Corruption Error (mCE):**

  ```python
  def calculate_mCE(model, baseline):
      score = [model[key][0] for key in model.keys() if key != 'clean']
      score = 100 - np.array(score)
      score_baseline = [baseline[key][0] for key in baseline.keys() if key != 'clean']
      score_baseline = 100 - np.array(score_baseline)
      CE = score / score_baseline
  
      mCE = np.mean(CE)
      print("mCE: {:.2f}%.".format(mCE * 100))
      CE = np.round(CE * 100, 2)
      print("CE: {}.".format(CE))
  
      return mCE, CE
  ```
  
- **mean Resilience Rate (mRR):**
  ```python
  def calculate_mRR(model):
      score = [model[key][0] for key in model.keys() if key != 'clean']
      score = np.array(score)
      RR = score / model['clean'][0]
  
      mRR = np.mean(RR)
      print("mRR: {:.2f}%.".format(mRR * 100))
      RR = np.round(RR * 100, 2)
      print("RR: {}.".format(RR))
  
      return mRR, RR
  ```


## Baseline

We select [CenterPoint](https://github.com/ldkong1205/Robo3D/blob/main) and [MinkUNet](https://github.com/ldkong1205/Robo3D/blob/main/docs/results/MinkUNet-18_cr1.0.md) as the baseline models for the 3D object detection and LiDAR segmentation tasks, respectively.

The **scores** of baseline models from each corruption set are attached as follows:

- **SemanticKITTI-C:**
  ```python
  MinkUNet_18_cr10 = {
    # type,             mIoU,
    'clean':           [62.76], 
    'fog':             [55.87],
    'wet_ground':      [53.99],
    'snow':            [53.28],
    'motion_blur':     [32.92],
    'beam_missing':    [56.32],
    'crosstalk':       [58.34],
    'incomplete_echo': [54.43],
    'cross_sensor':    [46.05],
  }
  ```
  ```python
  MinkUNet_18_cr10_mCE, MinkUNet_18_cr10_CE = calculate_mCE(MinkUNet_18_cr10, MinkUNet_18_cr10)
  ```
  ```python
  mCE: 100.00%.
  CE: [ 100. 100. 100. 100. 100. 100. 100. 100. ].
  ```
  ```python
  MinkUNet_18_cr10_mRR, MinkUNet_18_cr10_RR = calculate_mRR(MinkUNet_18_cr10)
  ```
  ```python
  mRR: 81.90%.
  RR: [ 89.02 86.03 84.89 52.45 89.74 92.96 86.73 73.37 ].
  ```

- **KITTI-C:**
  ```python
  CenterPoint = {
    # type,             mAP,
    'clean':           [68.70], 
    'fog':             [53.10],
    'wet_ground':      [68.71],
    'snow':            [48.56],
    'motion_blur':     [47.94],
    'beam_missing':    [49.88],
    'crosstalk':       [66.00],
    'incomplete_echo': [58.90],
    'cross_sensor':    [45.12],
  }
  ```
  ```python
  CenterPoint_mCE = calculate_mCE(CenterPoint, CenterPoint)
  ```
  ```python
  mCE: 100.00%.
  CE: [ 100. 100. 100. 100. 100. 100. 100. 100. ].
  ```
  ```python
  CenterPoint_mRR = calculate_mRR(CenterPoint)
  ```
  ```python
  mRR: 79.73%.
  RR: [ 77.29 100.01 70.68 69.78 72.61 96.07 85.74 65.68 ].
  ```


- **nuScenes-C (Seg3D):**
  ```python
  MinkUNet_18_cr10 = {
    # type,             mIoU,
    'clean':           [75.76], 
    'fog':             [53.64],
    'wet_ground':      [73.91],
    'snow':            [40.35],
    'motion_blur':     [73.39],
    'beam_missing':    [68.54],
    'crosstalk':       [26.58],
    'incomplete_echo': [63.83],
    'cross_sensor':    [50.95],
  }
  ```
  ```python
  MinkUNet_18_cr10_mCE, MinkUNet_18_cr10_CE = calculate_mCE(MinkUNet_18_cr10, MinkUNet_18_cr10)
  ```
  ```python
  mCE: 100.00%.
  CE: [ 100. 100. 100. 100. 100. 100. 100. 100. ].
  ```
  ```python
  MinkUNet_18_cr10_mRR, MinkUNet_18_cr10_RR = calculate_mRR(MinkUNet_18_cr10)
  ```
  ```python
  mRR: 74.44%.
  RR: [ 70.80 97.56 53.26 96.87 90.47 35.08 84.25 67.25 ].


- **nuScenes-C (Det3D):**
  ```python
  CenterPoint_PP = {
    # type,             NDS,
    'clean':           [45.99], 
    'fog':             [35.01],
    'wet_ground':      [45.41],
    'snow':            [31.23],
    'motion_blur':     [41.79],
    'beam_missing':    [35.16],
    'crosstalk':       [35.22],
    'incomplete_echo': [32.53],
    'cross_sensor':    [25.78],
  }
  ```
  ```python
  CenterPoint_mCE = calculate_mCE(CenterPoint, CenterPoint)
  ```
  ```python
  mCE: 100.00%.
  CE: [ 100. 100. 100. 100. 100. 100. 100. 100. ].
  ```
  ```python
  CenterPoint_mRR = calculate_mRR(CenterPoint)
  ```
  ```python
  mRR: 76.68%.
  RR: [ 76.13 98.74 67.91 90.87 76.45 76.58 70.73 56.06 ].
  ```


- **WOD-C (Seg3D):**
  ```python
  MinkU18 = {
    # type,             mIoU,
    'clean':           [69.06], 
    'fog':             [66.99],
    'wet_ground':      [60.99],
    'snow':            [57.75],
    'motion_blur':     [68.92],
    'beam_missing':    [64.15],
    'crosstalk':       [65.37],
    'incomplete_echo': [63.36],
    'cross_sensor':    [56.44],
  }
  ```
  ```python
  MinkUNet_18_mCE, MinkUNet_18_CE = calculate_mCE(MinkUNet_18, MinkUNet_18)
  ```
  ```python
  mCE: 100.00%.
  CE: [ 100. 100. 100. 100. 100. 100. 100. 100. ].
  ```
  ```python
  MinkUNet_18_mRR, MinkUNet_18_RR = calculate_mRR(MinkUNet_18)
  ```
  ```python
  mRR: 91.22%.
  RR: [ 97.00 88.31 83.62 99.80 92.89 94.66 91.75 81.73 ].
  ```


- **WOD-C (Det3D):**
  ```python
  CenterPoint = {
    # type,            mAPH_L2,
    'clean':           [63.59], 
    'fog':             [43.06],
    'wet_ground':      [62.84],
    'snow':            [58.59],
    'motion_blur':     [43.53],
    'beam_missing':    [54.41],
    'crosstalk':       [60.32],
    'incomplete_echo': [57.01],
    'cross_sensor':    [43.98],
  }
  ```
  ```python
  CenterPoint_mCE = calculate_mCE(CenterPoint, CenterPoint)
  ```
  ```python
  mCE: 100.00%.
  CE: [ 100. 100. 100. 100. 100. 100. 100. 100. ].
  ```
  ```python
  CenterPoint_mRR = calculate_mRR(CenterPoint)
  ```
  ```python
  mRR: 83.30%.
  RR: [ 67.72 98.82 92.14 68.45 85.56 94.86 89.65 69.16 ].
  ```


## Custom Model

You can adopt the following **template** to probe the robustness of your custom model:

```python
YOUR_MODEL = {
    # type            metric
    'clean':           [  ], 
    'fog':             [  ],
    'wet_ground':      [  ],
    'snow':            [  ],
    'motion_blur':     [  ],
    'beam_missing':    [  ],
    'crosstalk':       [  ],
    'incomplete_echo': [  ],
    'cross_sensor':    [  ],
}
```

```python
mCE = calculate_mCE(YOUR_MODEL, BASELINE)
```

```python
mRR = calculate_mRR(YOUR_MODEL)
```

