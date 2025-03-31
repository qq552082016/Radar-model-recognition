# 开放环境下的电磁目标型号识别研究
Original Implementation: https://github.com/AIML-UESTC/ZhixinXu-2025

The complete dataset and pre trained weights can be downloaded from this link:
https://pan.baidu.com/s/17OmwKUtoTOtoLvSJFDKhtw?pwd=2vjt
Extract the weights folder directly to the project path

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
