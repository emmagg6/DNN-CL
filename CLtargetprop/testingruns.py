'''

CL training -- 100 tests runs -- BP, DTP, FWDTP trained in mnist, then fashionmnist, then mnist

Checkpoint tests for accuracy and loss for each model at the start and end of training.

'''

from utils import worker_init_fn, set_seed, combined_loss, set_wandb, set_device
from dataset import make_MNIST, make_FashionMNIST, make_CIFAR10, make_CIFAR100

from Models.BP.bp_nn import bp_net
from Models.TP.tp_nn import tp_net

import os
import sys
import wandb
import torch
import argparse
import numpy as np
from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

models = ["BP", "TP", "DTP", "FWDTP"]
# datasets = ["MNIST", "FashionMNIST", "CIFAR10"]
# datasets = ["m", "f", "c"]
datasets = ["m", 'f', 'm']

# TESINGING AND MODEL PARAMETERS
epochs = 5
batch_size = int(256)
test = True  # from FWDTP paper's main.py
label_augentation = False  # from FWDTP paper's main.py
depth = 6
direct_depth = 1

# for TP
lr = 1e-3
lr_backward = 1e-3
std_backward = 0.01
loss_feedback = "DTP"
epochs_backward = 5
sparse_ratio = 0.1 #[0.1, 0.5, 0.9] # for FWDTP
sparse_ratio_str = f"-sparse-{sparse_ratio}" if 0 <= sparse_ratio <= 1 else ""

# input and output dimensions depend on the dataset
hid_dim = 256

log = False # for wandb visuals
save = "yes"

TRIALS = 5

device = set_device()
print(f"DEVICE: {device}")

for mod in models:

    for trial in range(TRIALS):

        set_seed(trial)

        params = {}
        print("Parameter Setup ... ")

        if mod == "BP":
            params = {
                "ff1": {
                    "type": "parameterized",
                    "act": "linear-BN",
                    "init": "orthogonal"
                },
                "ff2": {
                    "type": "parameterized",
                    "act": "tanh-BN",
                    "init": "orthogonal"
                }
            }
        elif mod == "TP":
            params["ff1"] = {"type": "parameterized",
                            "init": None,
                            "act": "linear-BN"}
            params["ff2"] = {"type": "parameterized",
                            "init": "orthogonal",
                            "act": "tanh-BN"}
            params["bf1"] = {"type": "parameterized",
                            "init": "uniform",
                            "act": "tanh-BN"}
            params["bf2"] = {"type": "parameterized",
                            "init": None,
                            "act": "linear-BN"}
            params["last"] = "linear"
        elif mod == "DTP":
            params["ff1"] = {"type": "parameterized",
                            "init": None,
                            "act": "linear-BN"}
            params["ff2"] = {"type": "parameterized",
                            "init": "orthogonal",
                            "act": "tanh-BN"}
            params["bf1"] = {"type": "parameterized",
                            "init": "orthogonal",
                            "act": "tanh-BN"}
            params["bf2"] = {"type": "difference",
                            "init": None,
                            "act": "linear-BN"}
            params["last"] = "linear"
        elif mod == "FWDTP":
            params["ff1"] = {"type": "parameterized",
                            "init": None,
                            "act": "linear-BN"}
            params["ff2"] = {"type": "parameterized",
                            "init": "orthogonal",
                            "act": "tanh-BN"}
            params["bf1"] = {"type": "parametrized",
                            "init": "uniform" + sparse_ratio_str,
                            "act": "tanh-BN"}
            params["bf2"] = {"type": "difference",
                            "init": None,
                            "act": "linear-BN"}
            params["last"] = "linear-BN"
        else :
            raise ValueError("Unkown algorithm. Please choose from BP, TP, DTP, FWDTP.")

        if log :
            print("Logging")
            wandb.init(project="CLtargetprop", config=params)

        ########### DATA ########### AND LEARNING RATE
        for d, data in enumerate(datasets): 
            if data == "m":
                print("making MNIST ...")
                in_dim = 784
                out_dim = 10
                trainset, validset, testset = make_MNIST(label_augentation, out_dim, test)
                
                if mod == "BP" :
                    stepsize = 0.04
                else :
                    stepsize = 0.04
            elif data == "f":
                print("making FashionMNIST ...")
                in_dim = 784
                out_dim = 10
                trainset, validset, testset = make_FashionMNIST(label_augentation, out_dim, test)
                
                if mod == "BP" :
                    stepsize = 0.02
                else :
                    stepsize = 0.004
            elif data == "c":
                print("making CIFAR10 ...")
                in_dim = 3072
                out_dim = 10
                trainset, validset, testset = make_CIFAR10(label_augentation, out_dim, test)
                if mod == "BP" :
                    stepsize = 0.04
                else :
                    stepsize = 0.04
            else :
                raise ValueError("Unkown dataset. Please choose from MNIST ('m'), FashionMNIST ('f'), CIFAR10 ('c').")

# <torch.utils.data.dataloader.DataLoader object at 0x1020366a0> 

            if label_augentation :
                loss_function = (lambda pred, label: combined_loss(pred, label, device, out_dim))
            else:
                loss_function = nn.CrossEntropyLoss(reduction="sum")

            train_loader = torch.utils.data.DataLoader(trainset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0, # slower but necessary due to loop of trials and new training
                                                    pin_memory=True,
                                                    worker_init_fn=worker_init_fn)
            valid_loader = torch.utils.data.DataLoader(validset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    worker_init_fn=worker_init_fn)
            test_loader = torch.utils.data.DataLoader(testset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    worker_init_fn=worker_init_fn)


            ## for saving checkpoints
            str_datasets_trials_1 = "-" + datasets[0]
            str_datasets_trials_2 = "-" + datasets[0] + "-" + datasets[1]
            str_datasets_trials_3 = "-" + datasets[0] + "-" + datasets[1] + "-" + datasets[2]


        ######### MODEL ###########
            if mod == "BP":
                model = bp_net(depth, in_dim, hid_dim, out_dim, loss_function, device, params=params)

                str_models_datasets_trials = mod + str_datasets_trials_1
                ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_1 + ".json"
                save_ckpts = "checkpoints/" + mod + "/EVAL-" + mod + "-0"+ str_datasets_trials_1 + ".json"
                if d > 0 :
                    if d == 1:
                        str_models_datasets_trials = str_datasets_trials_2
                        prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                        ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                        save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_2 + ".json"
                        save_ckpts = "checkpoints/" + mod + "/EVAL-" + mod + "-0"+ str_datasets_trials_2 + ".json"
                    elif d == 2:
                        str_models_datasets_trials = str_datasets_trials_3
                        prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                        ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3  + "-trial" + str(trial) + ".pth"
                        save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_3 + ".json"
                        save_ckpts = "checkpoints/" + mod + "/EVAL-" + mod + "-0"+ str_datasets_trials_3 + ".json"
                    model.load_state(prev_ckpt, lr)

                model.train_model(train_loader, valid_loader, epochs, lr, log, save, 
                                trial=trial, save_ckpts=save_ckpts, new_ckpt= ckpt, train_ckpts=save_training)
            else :
                model = tp_net(depth, direct_depth, in_dim, hid_dim, out_dim, loss_function, device, params=params)

                str_models_datasets_trials = mod + str_datasets_trials_1
                ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_1 + ".json"
                save_ckpts = "checkpoints/" + mod + "/EVAL-" + mod + "-0"+ str_datasets_trials_1 + ".json"
                if d > 0 :
                    if d == 1:
                        str_models_datasets_trials = str_datasets_trials_2
                        prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                        ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                        save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_2 + ".json"
                        save_ckpts = "checkpoints/" + mod + "/EVAL-" + mod + "-0"+ str_datasets_trials_2 + ".json"
                    elif d == 2:
                        str_models_datasets_trials = str_datasets_trials_3
                        prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                        ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3  + "-trial" + str(trial) + ".pth"
                        save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_3 + ".json"
                        save_ckpts = "checkpoints/" + mod + "/EVAL-" + mod + "-0"+ str_datasets_trials_3 + ".json"
                    model.load_state(prev_ckpt, lr)

                model.train(train_loader, valid_loader, epochs, lr, lr_backward, std_backward, stepsize, log, 
                            {"loss_feedback": loss_feedback, "epochs_backward": epochs_backward}, save,
                            trial=trial, save_ckpts=save_ckpts, new_ckpt= ckpt, train_ckpts=save_training)


    # Test the model
    # loss, acc = model.external_test(test_loader)
    # print(f"Test Loss      : {loss}")
    # if acc is not None:
    #     print(f"Test Acc       : {acc}")