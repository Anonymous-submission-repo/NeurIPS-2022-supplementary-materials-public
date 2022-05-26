"""Require: pip install flaml[test,ray]
"""
import re
from tkinter.tix import Tree
import argparse
from nbformat import current_nbformat
from flaml.searcher.blendsearch import BlendSearch
from sklearn.model_selection import train_test_split
import sklearn.metrics
import sklearn.datasets
import pickle
import sys
import torch
import random
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

try:
    from ray.tune.integration.xgboost import TuneReportCheckpointCallback
except ImportError:
    print("skip test_xgboost because ray tune cannot be imported.")
import xgboost as xgb
import os
import logging
import matplotlib.pyplot as plt
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _test_xgboost(seed, path, data, T, C, time):
    from flaml.data import load_openml_dataset
    from flaml import tune, AutoML, CFO
    from ray import tune as raytune

    set_seed(seed)
    data_func = {"adult":179, "poker":354, "shuttle":40685}
    data_num = data_func[data]
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = load_openml_dataset(
        dataset_id=data_num, data_dir="./download/"
    )
    preference_list = ["val_loss","train_time"]
    mode = {"val_loss": "min","train_time":"min"}
    if data in ["house_16H", "poker"]:
        metric = "r2"
        task = "regression"
    elif data in ["car","shuttle"]:
        metric = "log_loss"
        task = "classification"
    elif data in ["christine","adult"]:
        metric = "roc_auc"
        task = "classification"
    automl = AutoML()
    settings = {
        "time_budget": 0,  
        "metric": metric, 
        "estimator_list": [
            "xgboost"
        ], 
        "task": task,  
        "max_iter": 0,
        "train_time_limit": 0,
        "keep_search_state": True,
        "log_training_metric": True,
        "verbose": 0,
        "seed":seed,
    }

    automl.fit(X_train=X_train, y_train=y_train, **settings)
    real_toler = {}
    C={}
    for key,value in zip(preference_list,toler_value):
        real_toler[key] = value
    for key,value in zip(preference_list,c_value):
        C[key] = value
    automl._state.preference_list = preference_list
    automl._state.use_lexicographic_preference = True
    automl._state.mode = mode
    automl._state.T = real_toler
    automl._state.C = C


    evaluation_function = automl.trainable
    search_space = automl.search_space
    search_space_special = {
        "max_depth": raytune.randint(1, 9)
    }

    for key, value in search_space.items():
        if key in search_space_special.keys():
            search_space[key] = value

    algo = CFO(
        space=search_space,
        use_lexicographic_preference=True,
        metric=preference_list,
        mode=mode,
        seed=seed,
    )

    analysis = tune.run(
        evaluation_function,
        config=search_space,
        metric=preference_list,
        local_dir="logs/",
        num_samples=100000000,
        time_budget_s = time,
        search_alg=algo,
        use_ray=False,
        resources_per_trial={"cpu": 1},
    )

    results = analysis.results
    f = open(os.path.join(path,"result.pckl"), "wb")
    pickle.dump(results, f)
    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="dataset name", type=str, default="adult")
    parser.add_argument("--seed", help="seeds", type=int, nargs="+")
    parser.add_argument("--c_value", help="constraint", type=float, nargs="+")
    parser.add_argument("--toler_value", help="toler range", type=float, nargs="+")
    parser.add_argument("--time", help="time", type=int,default=3000)


    args = parser.parse_args()
    data = args.data
    time = args.time

    for seed in args.seed:
        c_value = args.c_value
        toler_value = args.toler_value
        data = args.data
        toler_list = [str(a)+"-" for a in toler_value]
        toler_str = "".join(toler_list)
        c_list = [str(b)+"-" for b in c_value]
        c_str = "".join(c_list)
        path = "./results/{data}/seed-{seed}_algorithm-CFO_C-{c_str}_toler-{toler_str}/".format(data = data, seed = seed, c_str=c_str, toler_str = toler_str)
        if not os.path.isdir(path):
            os.makedirs(path)
        _test_xgboost(seed, path, data, toler_value, c_value, time)