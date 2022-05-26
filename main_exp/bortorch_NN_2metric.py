import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import optuna
import random
import numpy as np
import sys
import time
import os
import pickle
from optuna.multi_objective.samplers import RandomMultiObjectiveSampler

DEVICE = torch.device("cpu")
DIR = ".."
BATCHSIZE = 128
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_{}".format(i), 0.2, 0.5)
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
    return np.log2(flops).item(), accuracy

time_start = time.time()
def objective(trial):
    trial.set_user_attr("wall_clock_time", time.time()-time_start)
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
    model= define_model(trial).to(DEVICE)
    
    optimizer = torch.optim.Adam(
        model.parameters(), trial.suggest_float("lr", 1e-5, 1e-1,log=True)
    )
    n_epoch = trial.suggest_int("n_epoch", 1, 20)
    for epoch in range(n_epoch):
        train_model(model, optimizer, train_loader)
    flops, accuracy = eval_model(model, val_loader)
    return accuracy,flops

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seeds", type=int, nargs="+")
parser.add_argument("--time", help="time", type=int,default=300)
parser.add_argument("--algorithm", help="time", type=str,default="qehvi")
args = parser.parse_args()
algorithm = args.algorithm
seed = args.seed[0]
time_budget = args.time

path = "./results/mnist/seed-{seed}_algorithm-{algorithm}/".format(seed = seed, algorithm = algorithm)
if not os.path.isdir(path):
    os.makedirs(path)


if algorithm == "qehvi":
    Func = optuna.integration.botorch._get_default_candidates_func(n_objectives=2)
    sampler = optuna.integration.BoTorchSampler(
        candidates_func=Func,
        n_startup_trials=1,
    )
elif algorithm == "parego":
    Func = optuna.integration.botorch._get_default_candidates_func(n_objectives=4)
    sampler = optuna.integration.BoTorchSampler(
        candidates_func=Func,
        n_startup_trials=1,
    )
elif algorithm == "random":
    sampler = RandomMultiObjectiveSampler()

set_seed(seed)
if algorithm in ["qehvi","parego"]:
    study = optuna.create_study(
        directions=["maximize","minimize"],
        sampler=sampler,
    )
elif algorithm in ["random"]:
    study = optuna.multi_objective.create_study(["maximize","minimize"], sampler=RandomMultiObjectiveSampler())

study.optimize(objective, timeout=time_budget)
trials = sorted(study.trials, key=lambda t: t.user_attrs["wall_clock_time"])
result_all = {}
index = 0
for trial in trials:
    result = {}
    result["accuracy"] = trial.values[0]
    result["flops"] = trial.values[1]
    result["wall_clock_time"] = trial.user_attrs["wall_clock_time"]
    result_all[str(index)] = result
    index+=1

f = open(os.path.join(path,"result.pckl"), "wb")
pickle.dump(result_all, f)
f.close()

