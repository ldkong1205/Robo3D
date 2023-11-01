<img src="https://github.com/ldkong1205/Robo3D/blob/main/docs/figs/logo2.png" align="right" width="30%">

# Getting Started

## Model Training

- This task aims to probe the **out-of-distribution (OoD) generalization** of 3D object detection and LiDAR segmentation models. We assume the training data come from the **original** datasets directly, **without** knowing the distributions of corrupted data beforehand.

- Please follow the conventional way of training your 3D object detection and LiDAR segmentation models.

- Kindly use the original validation sets to select your best possible hyperparameters.

- Do not use corruption simulations to fine-tune your trained models.

## Robustness Evaluation

:warning: **Important:** By default, we evaluate the model on the corruption set after each epoch, **without** using extra tricks including model ensemble and test-time-augmentation (TTA). To ensure fair comparisons, we suggest you follow the **same** configuration strictly.

Our generated corruption sets follow the same dataset structure as the original validation sets. You can think of each severity level as one validation set and interactively validate all three levels from every corruption type.

### Definition

- The **Corruption Error (CE)** for a model $A$ under corruption $i$ across three severity levels is:
  $$\text{CE}\_i^{\text{Model}A} = \frac{\sum\_{l=1}((1 - \text{mIoU})\_{i,l}^{\text{Model}A})}{\sum\_{l=1}((1 - \text{mIoU})\_{i,l}^{\text{Baseline}})} . $$

- The average $\text{CE}$ for a model $A$ on all corruptions, *i.e.*, **mean Corruption Error (mCE)**, is calculated as:
  $$\text{mCE} = \frac{1}{N}\sum^N\_{i=1}\text{CE}\_i , $$
  where $N=8$ denotes the number of corruption types in the Robo3D benchmark.

- The **Resilience Rate (RR)** for a model $A$ under corruption $i$ across three severity levels is:
  $$\text{RR}\_i^{\text{Model}A} = \frac{\sum\_{l=1}(\text{mIoU}\_{i,l}^{\text{Model}A})}{3\times\text{mIoU}\_{\text{clean}}^{\text{Model}A}} . $$

- The average RR for a model $A$ on all corruptions, *i.e.*, **mean Resilience Rate (mRR)**, is calculated as:
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
  CE: [100. 100. 100. 100. 100. 100. 100. 100.].
  ```
  ```python
  MinkUNet_18_cr10_mRR, MinkUNet_18_cr10_RR = calculate_mRR(MinkUNet_18_cr10)
  ```
  ```python
  mRR: 81.90%.
  RR: [89.02 86.03 84.89 52.45 89.74 92.96 86.73 73.37].
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

- nuScenes-C (Seg3D):





