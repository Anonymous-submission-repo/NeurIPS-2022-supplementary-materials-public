# LexicoFlow: Multi-objective Hyperparameter
This repository is the implementation of **LexicoFlow: Multi-objective Hyperparameter
Optimization with Lexicographic Preference**. 

The implementation of our method LexicoFlow is built upon an open-source AutoML library named FLAML. Thus the submitted code includes part of flaml’s code. But we emphasize that the contributors and copyright information about the open-source library FLAML do not necessarily reveal the identities of the authors of this work. We plan to open source the code accompanying the formal publication of this paper.

This version of the code is made to facilitate the peer review of the NIPS-2022 submission of our paper. 
We plan to release the code accompanying the formal publication of this paper. 


## Datasets
In tuning XGboost, we verify the performance of LexiFlow on the datasets shown below. All of these datasets are available on OpenML.
1. [adult](https://www.openml.org/search?type=data&sort=runs&id=179&status=active)
2. [shuttle](https://www.openml.org/search?type=data&sort=runs&id=40685&status=active)
3. [poker](https://www.openml.org/search?type=data&sort=runs&id=354&status=active)

In tuning neural networks, we verify the performance of LexiFLOW on [FashionMnist datasets](https://www.kaggle.com/datasets/zalando-research/fashionmnist).

## Experiments

### **Requirements**

To install requirements:
```setup
pip install -r requirements.txt
```


### **How to run** 

1. LexicoFLOW on Xgboost 

```
cd main_exp
python lexiflow_xgboost.py --data adult --seed 1 --toler_value 0.001 1.0 --time 5000 --c_value 0.00 0.00 
```

2. LexicoFLOW on NN with two objective

```
cd main_exp
python lexiflow_NN_2metric.py --seed 1  --toler_value 0.02 1.0 --c_value 0 0
```

2. LexicoFLOW on NN with three objective

```
cd main_exp
python lexiflow_NN_3metric.py --seed 1  --toler_value 0 0 0 --c_value 0.25 40000 0
```

3. baselines

```
cd main_exp
python botorch_xgboost.py --algorithm parego --seed 1 --data adult --time 5000 
python bortorch_NN_2metric.py --algorithm parego --seed 1 --time 3000

```

## Results

For example, runing the commandline below to show the results of Fig.1 (a).

```
python plot.py
```

