import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
import optuna
import numpy as np
import os
import pickle
import time
import argparse
from functools import partial
DEVICE = torch.device("cpu")
DIR = ".."
BATCHSIZE = 128
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
def update(result, bound, t, c, metric_order):
    if metric_order == -1: # final
        if result <= bound:
            return [1,metric_order]
        else:
            return [-1,metric_order]  
    else: 
        if bound > c: 
            if result <= bound:
                return [1,metric_order]
            if result > bound + t:
                return [-1,metric_order]
            else:
                return [0,metric_order]
        else: 
            if result <= c:
                return [0,metric_order]
            else:
                return [-1,metric_order]   

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def define_model(configuration):
    n_layers = configuration["n_layers"]
    layers = []
    in_features = 28 * 28
    for i in range(n_layers):
        out_features = configuration["n_units_l{}".format(i)]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = configuration["dropout_{}".format(i)]
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, 10))
    layers.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*layers)

def train_model(model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        F.nll_loss(model(data), target).backward()
        optimizer.step()

def eval_model(model, valid_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
            pred = model(data).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / N_VALID_EXAMPLES
    flops, _ = thop.profile(model, inputs=(torch.randn(1, 28 * 28).to(DEVICE),), verbose=False)
    return flops, 1-accuracy,

def objective(start_time, preference_list, toler_range, C, configuration):
    result = {
        "wall_clock_time": time.time() - start_time,
    }
    T = toler_range
    incumbent_comparision = configuration["incumbent_comparision"]
    del configuration["incumbent_comparision"]
    Lbest = configuration["Lbest"]
    del configuration["Lbest"]
    order = {}
    index_order = 0
    for key in preference_list:
        order[key] = index_order
        index_order  = index_order+1
    last_key = preference_list[-1]
    order[last_key] = -1
    metric_op = {"flops":1, "train_time":1, "error_rate":1}
    train_dataset = torchvision.datasets.FashionMNIST(
        DIR, train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, list(range(N_TRAIN_EXAMPLES))),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    val_dataset = torchvision.datasets.FashionMNIST(
        DIR, train=False, transform=torchvision.transforms.ToTensor()
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_dataset, list(range(N_VALID_EXAMPLES))),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    model = define_model(configuration).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), configuration["lr"]
    )
    train_time_start = time.time()
    n_epoch = configuration["n_epoch"]
    for epoch in range(n_epoch):
        train_model(model, optimizer, train_loader)
    train_time = time.time()-train_time_start
    flops, error_rate = eval_model(model, val_loader)
    result["flops"] = flops
    result["train_time"] = train_time
    result["error_rate"] = error_rate
    if Lbest == None:
        Lbest  = {}
        for key,value in result.items():
            if key in preference_list:
                Lbest[key] = result[key]
        result["Lbest"] = Lbest
    if incumbent_comparision is None:
        result["lexicographic_preference"] = True     
    else:
        for metric in preference_list:
            op = metric_op[metric]
            c = C[metric]
            t = T[metric]
            metric_order = order[metric]
            bound = Lbest[metric]
            comp_result = update(
                result[metric]*op, bound, t, c, metric_order 
            )
            if comp_result[0] == 0:
                continue
            elif comp_result[0] == 1:
                result["lexicographic_preference"] = True
                break
            else:
                result["lexicographic_preference"] = False
                break
        if result["lexicographic_preference"]:
            if comp_result[1] == -1:
                start_index = len(preference_list)-1
            else:
                start_index = comp_result[1]
            for metric_index in range(start_index,len(preference_list)):
                metric_name = preference_list[metric_index]
                Lbest[metric_name]  = result[metric_name]
                result["Lbest"] = Lbest
        else:
                result["Lbest"] = Lbest
    result["incumbent_comparision"] = incumbent_comparision
    return result

def test(seed, c_value, toler_value,preference_list):
    from flaml import tune, CFO
    set_seed(seed)
    real_toler = {}
    C={}
    for key,value in zip(preference_list, toler_value):
        real_toler[key] = value
    for key,value in zip(preference_list, c_value):
        C[key] = value
    mode = {"error_rate":"min","flops":"min","train_time":"min"}
    search_space = {
    "n_layers": tune.randint(lower=1,upper=3),
    "n_epoch":tune.randint(lower=1,upper=20),
    "n_units_l0":tune.randint(lower=4,upper=128),
    "n_units_l1":tune.randint(lower=4,upper=128),
    "n_units_l2":tune.randint(lower=4,upper=128),
    "dropout_0":tune.uniform(lower=0.2,upper=0.5),
    "dropout_1":tune.uniform(lower=0.2,upper=0.5),
    "dropout_2":tune.uniform(lower=0.2,upper=0.5),
    "lr":tune.uniform(lower=1e-5,upper=1e-1),
    }

    start_time = time.time()
    tune_objective = partial(objective,start_time, preference_list, real_toler, C)

    algo = CFO(
        space=search_space,
        use_lexicographic_preference=True,
        metric=preference_list,
        mode=mode,
        seed=seed,
    )
    analysis = tune.run(
        tune_objective,
        config=search_space,
        low_cost_partial_config={
             "n_layers":1,
        },
        local_dir="logs/",
        num_samples=100000000,
        time_budget_s = 3000,
        search_alg=algo,
        use_ray=False,
    )
    result = analysis.results
    toler_list = [str(a)+"-" for a in toler_value]
    toler_str = "".join(toler_list)
    c_list = [str(b)+"-" for b in c_value]
    c_str = "".join(c_list)
    path = "./results/mnist/seed-{seed}_algorithm-CFO_C-{c_str}_toler-{toler_str}/".format(seed = seed,c_str=c_str,toler_str = toler_str)
    if not os.path.isdir(path):
        os.makedirs(path)
    f = open(os.path.join(path,"result.pckl"), "wb")
    pickle.dump(result, f)




preference_list = ["flops","train_time","error_rate"]
parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seeds", type=int, nargs="+")
parser.add_argument("--c_value", help="constraint", type=float, nargs="+")
parser.add_argument("--toler_value", help="toler range", type=float, nargs="+")
args = parser.parse_args()
for i in args.seed:
    test(i,args.c_value,args.toler_value,preference_list)
