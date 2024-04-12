'''

CL training -- 100 tests runs -- BP, DTP, FWDTP trained in mnist, then fashionmnist, then mnist

Checkpoint tests for accuracy and loss for each model at the start and end of training.

'''

from utils import worker_init_fn, set_seed, combined_loss, set_wandb, set_device
from dataset import make_MNIST, make_FashionMNIST, make_CIFAR10, make_CIFAR100

from Models.BP.bp_nn import bp_net
from Models.TP.tp_nn import tp_net

from initParameters import set_params

import os
import sys
import wandb
import torch
import argparse
import numpy as np
from torch import nn

models = ["BP", "TP", "DTP", "FWDTP"]
# datasets = ["MNIST", "FashionMNIST", "CIFAR10"]
datasets = ["m", "f", "c"]

# TESINGING AND MODEL PARAMETERS
epochs = 15
batch_size = 256
seed = 0
test = "store_true"  # from FWDTP paper's main.py
label_augentation = "store_true"  # from FWDTP paper's main.py
depth = 6
direct_depth = 1

# for TP
lr = 1e-3
lr_backward = 1e-3
std_backward = 0.01
loss_feedback = "DTP"
epochs_backward = 5
sparse_ratio = 0.1 #[0.1, 0.5, 0.9] # for FWDTP

# input and output dimensions depend on the dataset
hid_dim = 256

log = True # for wandb visuals
save = "yes"

TRIALS = 100

for trial in range(TRIALS):
    
    device = set_device()
    print(f"DEVICE: {device}")

    params = {}

    for model in models:
        if model == "BP":
            ...
        elif model == "TP":
            params["ff1"] = {"type": "parameterized",
                            "init": None,
                            "act": "linear-BN"}
            params["ff2"] = {"type": "parameterized",
                            "init": ["parameterized" + "orthogonal"],
                            "act": "tanh-BN"}
            params["bf1"] = {"type": "parameterized",
                            "init": ["parameterized" + "uniform"],
                            "act": "tanh-BN"}
            params["bf2"] = {"type": "parameterized",
                            "init": None,
                            "act": "linear-BN"}
        elif model == "DTP":
            params["ff1"] = {"type": "parameterized",
                            "init": None,
                            "act": "linear-BN"}
            params["ff2"] = {"type": "parameterized",
                            "init": ["parameterized" + "orthogonal"],
                            "act": "tanh-BN"}
            params["bf1"] = {"type": "parameterized",
                            "init": ["parameterized" +"uniform"],
                            "act": "tanh-BN"}
            params["bf2"] = {"type": "difference",
                            "init": None,
                            "act": "linear-BN"}
        elif model == "FWDTP":
            params["ff1"] = {"type": "parameterized",
                            "init": None,
                            "act": "linear-BN"}
            params["ff2"] = {"type": "parameterized",
                            "init": ["parameterized"+ "orthogonal"],
                            "act": "tanh-BN"}
            params["bf1"] = {"type": "random",
                            "init": ["parameterized" + "uniform"] + sparse_ratio,
                            "act": "tanh-BN"}
            params["bf2"] = {"type": "difference",
                            "init": None,
                            "act": "linear-BN"}
            params["last"] = "linear-BN"
        else :
            raise ValueError("Unkown algorithm. Please choose from BP, TP, DTP, FWDTP.")

    if log :
        wandb.init(project="CLtargetprop", config=params)

    ########### DATA ########### AND LEARNING RATE
    for data, d in enumerate(datasets): 
        if data == "m":
            in_dim = 784
            out_dim = 10
            trainset, validset, testset = make_MNIST(label_augentation, out_dim, test)
            
            if model == "BP" :
                stepsize = 0.05
            else :
                stepsize = 0.04
        elif data == "f":
            in_dim = 784
            out_dim = 10
            trainset, validset, testset = make_FashionMNIST(label_augentation, out_dim, test)
            
            if model == "BP" :
                stepsize = 0.05
            else :
                stepsize = 0.004
        elif data == "c":
            in_dim = 3072
            out_dim = 10
            trainset, validset, testset = make_CIFAR10(label_augentation, out_dim, test)
            if model == "BP" :
                stepsize = 0.05
            else :
                stepsize = 0.05


    # make dataloader
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2,
                                              pin_memory=True,
                                              worker_init_fn=worker_init_fn)

    if label_augentation == "store_true":
        loss_function = (lambda pred, label: combined_loss(pred, label, device, out_dim))
    else:
        loss_function = nn.CrossEntropyLoss(reduction="sum")

    ## for saving checkpoints
    str_models_datasets_trials_1 = "-" + datasets[0]
    str_models_datasets_trials_2 = "-" + datasets[0] + "-" + datasets[1]
    str_models_datasets_trials_3 = "-" + datasets[0] + "-" + datasets[1] + "-" + datasets[2]


    ######### MODEL ###########
    for model in models:
        if model == "BP":
            model = bp_net(depth, in_dim, hid_dim, out_dim, loss_function, device, params=params)
            str_models_datasets_trials = model + "-" + data
            if d > 0 :
                if d == 1:
                    data_tm1 = datasets[d-1]
                    str_models_datasets_trials = model + "-" + data_tm1
                    ckpt = "checkpoints/" + model + "/" + str_models_datasets_trials + "-trial" + str(trial) + ".pth"
                elif d == 2:
                    data_tm2 = datasets[d-2]
                    data_tm1 = datasets[d-1]
                    str_models_datasets_trials = model + "-" + data_tm1 + "-" + data_tm2
                    ckpt = "checkpoints/" + model + "/" + str_models_datasets_trials + "-trial" + str(trial) + ".pth"
            
                model.load_state(ckpt, lr)
            model.train_model(train_loader, valid_loader, epochs, lr, log, save, trial=trial, str_prev_checkpoint=str_models_datasets_trials)
        else :
            model = tp_net(depth, direct_depth, in_dim, hid_dim, out_dim, loss_function, device, params=params)
            str_models_datasets_trials = str_models_datasets_trials_1
            if d > 0 :
                if d == 1:
                    str_models_datasets_trials = str_models_datasets_trials_2
                    ckpt = "checkpoints/" + model + "/" + model + str_models_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                elif d == 2:
                    str_models_datasets_trials = str_models_datasets_trials_3
                    ckpt = "checkpoints/" + model + "/" + model + str_models_datasets_trials_3  + "-trial" + str(trial) + ".pth"
            
                model.load_state(ckpt, lr)
            model.train(train_loader, valid_loader, epochs, lr, lr_backward, std_backward, stepsize, log, 
                        {"loss_feedback": loss_feedback, "epochs_backward": epochs_backward}, save,
                        trial=trial, mod_name = model, str_prev_checkpoint=str_models_datasets_trials, data_name=data)


    # Test the model
    loss, acc = model.external_test(test_loader)
    print(f"Test Loss      : {loss}")
    if acc is not None:
        print(f"Test Acc       : {acc}")