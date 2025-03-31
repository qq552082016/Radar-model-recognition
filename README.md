# 开放环境下的电磁目标型号识别研究
- Original Implementation: https://github.com/qq552082016/Radar-model-recognition

- The complete dataset and pre trained weights can be downloaded from this link:
https://pan.baidu.com/s/17OmwKUtoTOtoLvSJFDKhtw?pwd=2vjt
- Extract the weights folder directly to the project path

# Environments
- libmr==0.1.9
- matplotlib==3.10.1
- numpy==1.25.0
- Pillow==11.1.0
- scikit_learn==1.6.1
- seaborn==0.13.2
- tensorboard==2.18.0
- torch==1.11.0+cu113
- torchvision==0.12.0+cu113
- tqdm==4.67.1

# Installation 
Download the repository and apply `pip install -r requirements.txt` to install the required libraries. 

## 闭集识别
- Train
```
python train.py
```
- Test
```
python test.py
```
## 开集识别
* Step 1: Train a model for the dataset you choice
* Step 2: Load the trained model
* Step 3: Choose an open-set recognition method to test
- Openmax
```
python openset_test/openset_test_openmax.py
```
- Distance
```
python openset_test/openset_test_distance.py
```
- Energy
```
python openset_test/openset_test_energy.py
```

## 类增量识别
* Step 1: Constructing a dataset based on the incremental learning mode
* Step 2: Train pre-trained models using the basic dataset
* Step 3: Select a certain round of incremental learning model for testing
- Train
```
python incremental_learning.py
```
- Test
```
python incremental_test.py
```

If you have any question feel free to create an issue.
