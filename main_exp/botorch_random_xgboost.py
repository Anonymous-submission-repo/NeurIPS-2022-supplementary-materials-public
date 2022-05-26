from botorch.settings import suppress_botorch_warnings
from botorch.settings import validate_input_scaling
import optuna
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from flaml.data import load_openml_dataset
from flaml import tune, AutoML, CFO
from ray import tune as raytune
from pandas import DataFrame
import argparse
import time
import os
from optuna.multi_objective.samplers import RandomMultiObjectiveSampler
import sys
import pickle
import random
import torch

suppress_botorch_warnings(True)
validate_input_scaling(True)

def is_nan_or_inf(value):
    return np.isnan(value) or np.isinf(value)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="dataset name", type=str, default="adult")
parser.add_argument("--seed", help="seeds", type=int, nargs="+")
parser.add_argument("--time", help="time", type=int,default=3000)
parser.add_argument("--algorithm", help="time", type=str,default="qehvi")
args = parser.parse_args()

data_func = {"adult":179, "poker":354, "shuttle":40685}
data = args.data
data_num = data_func[data]
X_train, X_test, y_train, y_test = load_openml_dataset(
    dataset_id=data_num, data_dir="./download/"
)
if data in ["house_16H", "poker"]:
    metric = "r2"
    task = "regression"
elif data in ["car","shuttle"]:
    metric = "log_loss"
    task = "classification"
elif data in ["christine","adult"]:
    metric = "roc_auc"
    task = "classification"
time_budget = args.time
algorithm = args.algorithm

seeds = args.seed
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
    "seed":0,
}
mode = {"val_loss": "min","train_time":"min"}
automl.fit(X_train=X_train, y_train=y_train, **settings)
automl._state.use_lexicographic_preference = False
automl._state.only_result = True
automl._state.mode = mode
evaluation_function = automl.trainable



seed = seeds[0]
path = "./results_appendix/{data}/seed-{seed}_algorithm-{algorithm}/".format(data = data, seed = seed, algorithm = algorithm)
if not os.path.isdir(path):
    os.makedirs(path)
set_seed(seed)


study = optuna.multi_objective.create_study(["minimize", "minimize"], sampler=RandomMultiObjectiveSampler())

time_start = time.time()
def objective(trial):
    param = {
        'max_leaves': trial.suggest_int("n_estimators", 4, 32768, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        "n_estimators": trial.suggest_int("n_estimators", 4, 32768, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 128, log=True),
        "learning_rate": trial.suggest_float("learning_rate",1/1024,1.0,log=True),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1/1024, 1024, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1/1024, 1024, log=True),
        "learner": "xgboost",
    }
    result = evaluation_function(param)
    trial.set_user_attr("wall_clock_time", time.time()-time_start)
    if is_nan_or_inf(result["val_loss"]) or is_nan_or_inf(result["train_time"]):
        return 100,100
    return result["val_loss"],np.log2(result["train_time"]).item()

study.optimize(objective, timeout=time_budget, n_jobs=1)

trials = sorted(study.get_trials(), key=lambda t: t.user_attrs["wall_clock_time"])
result_all = {}
index = 0
for trial in trials:
    result = {}
    result["val_loss"] = trial.values[0]
    result["train_time"] = trial.values[1]
    result["wall_clock_time"] = trial.user_attrs["wall_clock_time"]
    if result["val_loss"] != 100:
        result_all[str(index)] = result
        index+=1
f = open(os.path.join(path,"result.pckl"), "wb")
pickle.dump(result_all, f)
f.close()
