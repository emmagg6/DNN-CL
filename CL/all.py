'''

CL training

Checkpoint tests for accuracy of each model after each epoch of training.

'''

from utils import *
from dataset import make_MNIST, make_FashionMNIST, make_CIFAR10, make_STL10

from Models.BP.bp_nn import bp_net
from Models.TP.tp_nn import tp_net
# from Models.PC.pc_nn import pc_net
from Models.PC.pc_nn_E import pc_net
from Models.KAN.kan_nn import kan_net
from Models.EP.ep_nn import ep_net
from Models.PC.pc_layers import ConvLayer, MaxPool, ProjectionLayer, FCLayer
from Models.Hnet.hnet_nn import hn, hn_train, hn_evaluate


import os
import sys
import wandb
import torch
import numpy as np
from torch import nn     
import torch.nn.functional as F

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



def main(TRIALS, models, datasets, epochs, epochs_backward, batch_size, 
         test, depth, direct_depth, lr, lr_backward, std_backward, 
         loss_feedback, sparse_ratio_str, hid_dim, log, save,
         num_inference_steps, inference_lr, larger=False):
    # set_seed(1)
    device = set_device()
    print(f"DEVICE: {device}")

    for mod in models:

        for trial in range(0, TRIALS+1):
            print("\n -------------------------------------")
            print(f"TRIAL: {trial}")
            print(" -------------------------------------\n")


            name = mod + "-" + str(trial)
            name = str(name)

            set_seed(trial)
            params = {}
            print("Parameter Setup ... ")

            lr = 0.1
            if mod == "KAN" or mod == "EP":
                lr = 0.005
            stepsize = 0.04
            if mod == "HNET":
                lr = 1e-3

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
                    },
                }
                params["name"] = mod
            elif mod == "TP":
                params["ff1"] = {"type": "identity",
                                "init": None,
                                "act": "linear-BN"}
                params["ff2"] = {"type": "parameterized",
                                "init": "orthogonal",
                                "act": "tanh-BN"}
                params["bf1"] = {"type": "parameterized",
                                "init": "orthogonal",
                                "act": "tanh-BN"}
                params["bf2"] = {"type": "identity",
                                "init": None,
                                "act": "linear-BN"}
                params["last"] = "linear"
                params["name"] = mod
                
            elif mod == "DTP":
                params["ff1"] = {"type": "identity",
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
                params["name"] = str(mod + "-eq")
                name = mod + "-eq-" + str(trial)
                name = str(name)

            elif mod == "FWDTP":
                params["ff1"] = {"type": "identity",
                                "init": None,
                                "act": "linear-BN"}
                params["ff2"] = {"type": "parameterized",
                                "init": "orthogonal",
                                "act": "tanh-BN"}
                params["bf1"] = {"type": "parameterized",
                                "init": "orthogonal" + sparse_ratio_str,
                                "act": "tanh-BN"}
                params["bf2"] = {"type": "difference",
                                "init": None,
                                "act": "linear-BN"}
                params["last"] = "linear-BN"
                params["name"] = mod

            elif mod == "PC":
                params["name"] = mod #mod + "0.0005.0.00005"
                # name = str(name)
            elif mod == "EP":
                params['cost_energy'] = 'cross_entropy'
                params['batch_size'] = batch_size
                params['tyoe'] = 'cond_gaussian'
                params['dynamics'] = {
                                        "dt": 0.1,
                                        "n_relax": 20,
                                        "tau": 1,
                                        "tol": 0
                                    }
                params["name"] = mod
            elif mod == "KAN":
                params["name"] = mod
            elif mod == "HNET":
                params["name"] = mod
                params['n_edges1'] = None
                criterion = nn.CrossEntropyLoss()
            else :
                raise ValueError("Unkown algorithm. Please choose from BP, TP, DTP, FWDTP, KAN.")

            
            if log :
                # print("Logging")
                wandb.init(project="Aug2025", config=params, name=name,  reinit=True)

            ########### DATA ########### AND LEARNING RATE
            count = 0
            type = "start"
            for d, data in enumerate(datasets): 
                count += 1
                if data == "m":
                    type = "m"

                    print("making MNIST ...")
                    in_dim = 784
                    out_dim = 10
                    if mod == "KAN":
                        trainset, validset = make_MNIST(out_dim, test, pc = True)
                    elif mod == "EP":
                        params['dimensions'] = [784, batch_size*out_dim, out_dim]
                        trainset, validset = make_MNIST(out_dim, test, ep = True)
                    else:
                        trainset, validset, testset = make_MNIST(out_dim, test)

                elif data == "f":
                    type = "f"

                    print("making FashionMNIST ...")
                    in_dim = 784
                    out_dim = 10
                    if mod == "KAN":
                        trainset, validset = make_FashionMNIST(out_dim, test, pc = True)
                    elif mod == "EP":
                        params['dimensions'] = [784, batch_size*out_dim, out_dim]
                        trainset, validset = make_FashionMNIST(out_dim, test, ep = True)
                    else:
                        trainset, validset, testset = make_FashionMNIST(out_dim, test)
                    


                elif data == "c":

                    type = "c"

                    print("making CIFAR10 ...")
                    in_dim = 3072
                    out_dim = 10
                    # trainset, validset, testset = make_CIFAR10(out_dim, test)
                    if mod == "KAN":
                        trainset, validset = make_CIFAR10(out_dim, test, True)
                    else:
                        trainset, validset, testset = make_CIFAR10(out_dim, test)

                elif data == "s":

                    type = "s"

                    print("making STL10 ...")
                    in_dim = 3072
                    out_dim = 10
                    # trainset, validset, testset = make_STL10(out_dim, test)
                    if mod == "KAN":
                        trainset, validset = make_STL10(out_dim, test, True)
                    else:
                        trainset, validset, testset = make_STL10(out_dim, test)
                else :
                    raise ValueError("Unkown dataset. Please choose from MNIST ('m'), FashionMNIST ('f'), CIFAR10 ('c').")

    # <torch.utils.data.dataloader.DataLoader object at 0x1020366a0> 

                loss_function = nn.CrossEntropyLoss(reduction="sum")

                
                if mod == "KAN":
                    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, pin_memory=True, shuffle=True)
                    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, pin_memory=True, shuffle=False)
                elif mod == "EP":
                    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True, shuffle=True)
                    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, drop_last=True, shuffle=False)
                else :
                    train_loader = torch.utils.data.DataLoader(trainset,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            num_workers=0, # slower but necessary due to loop of trials and datasets
                                                            pin_memory=True,
                                                            worker_init_fn=worker_init_fn)
                    valid_loader = torch.utils.data.DataLoader(validset,
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            num_workers=0,
                                                            pin_memory=True,
                                                            worker_init_fn=worker_init_fn)


                ## for saving checkpoints
                str_datasets_trials_1 = "-" + datasets[0] # "m-f-m-f-m" + "-" +
                if len(datasets) > 1 :
                    str_datasets_trials_2 = "-" + datasets[0] + "-" + datasets[1]
                if len(datasets) > 2 :
                    str_datasets_trials_3 = "-" + datasets[0] + "-" + datasets[1] + "-" + datasets[2]
                if len(datasets) > 3 :
                    str_datasets_trials_4 = "-" + datasets[0] + "-" + datasets[1] + "-" + datasets[2] + "-" + datasets[3]
                if len(datasets) > 4 :
                    str_datasets_trials_5 = "-" + datasets[0] + "-" + datasets[1] + "-" + datasets[2] + "-" + datasets[3] + "-" + datasets[4]
                if len(datasets) > 5 :
                    str_datasets_trials_6 = "-" + datasets[0] + "-" + datasets[1] + "-" + datasets[2] + "-" + datasets[3] + "-" + datasets[4] + "-" + datasets[5]



            ######### MODEL ###########
                if mod == "BP":

                    model = bp_net(depth, in_dim, hid_dim, out_dim, loss_function, device, params=params)
                    print("Model: ", mod)

                    ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                    save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_1 + ".json"
                    if d > 0 :
                        if d == 1:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_2 + ".json"
                        elif d == 2:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3  + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_3 + ".json"
                        elif d == 3:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_4 + ".json"
                        elif d == 4:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_5 + ".json"
                        elif d == 5:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_6 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_6 + ".json"
                        saved_state = torch.load(prev_ckpt)
                        model.load_state(prev_ckpt, lr)

                    model.train_model(type, count, train_loader, valid_loader, epochs, lr, log, save, 
                                    trial=trial, new_ckpt= ckpt, train_ckpts=save_training)
                    # print("trained BP")

                elif mod == "PC":
                    model = pc_net(depth, in_dim, hid_dim, out_dim, loss_function, device, batch_size, params=params)
                    print("Model: ", mod)

                    ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial)+ ".pth"
                    save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_1 + ".json"
                    prev_ckpt = "None"
                    
                    parent_dir = os.path.dirname(ckpt)
                    if not os.path.exists(parent_dir):
                        os.makedirs(parent_dir)
                    if d > 0 :
                        if d == 1:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial)+ ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial)+ ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_2 + ".json"
                        elif d == 2:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial)+ ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3  + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_3 + ".json"
                        elif d == 3:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3 + "-trial" + str(trial)+ ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial)+ ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_4 + ".json"
                        elif d == 4:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial)+ ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial)+ ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_5 + ".json"
                        elif d == 5:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial)+ ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_6 + "-trial" + str(trial)+ ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_6 + ".json"
                        prev_ckpt = os.path.join(prev_ckpt)
                        ckpt = os.path.join(ckpt)
                        model = pc_net.load_model(prev_ckpt)

                    train_data = list(iter(train_loader))
                    valid_data = list(iter(valid_loader))

                    model.train_model(type, count, train_data, valid_data, epochs, train_loader, valid_loader, batch_size, log, save, trial=trial, new_ckpt= ckpt, train_ckpts=save_training)

                    model.save_model(ckpt)

                elif mod == "DTP" or mod == "FWDTP":
                    model = tp_net(depth, direct_depth, in_dim, hid_dim, out_dim, loss_function, device, params=params)
                    print("Model: ", mod)

                    ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                    save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_1 + ".json"
                    if d > 0 :
                        if d == 1:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_2 + ".json"
                        elif d == 2:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3  + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_3 + ".json"
                        elif d == 3:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_4 + ".json"
                        elif d == 4:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_5 + ".json"
                        elif d == 5:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_6 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_6 + ".json"

                        saved_state = torch.load(prev_ckpt)
                        model.load_state(saved_state)

                    model.train( type, count, train_loader, valid_loader, epochs, lr, lr_backward, std_backward, stepsize, 
                                log, save, hyperparams={"loss_feedback": loss_feedback, "epochs_backward": epochs_backward}, 
                                trial=trial, new_ckpt= ckpt, train_ckpts=save_training)

                elif mod == "KAN":
                    
                    model = kan_net(in_dim, out_dim, loss_function, device, larger)
                    print("Model: ", mod)

                    ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                    save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_1 + ".json"
                    if d > 0 :
                        if d == 1:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_2 + ".json"
                        elif d == 2:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3  + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_3 + ".json"
                        elif d == 3:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_4 + ".json"
                        elif d == 4:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_5 + ".json"
                        elif d == 5:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_6 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_6 + ".json"
                        
                        saved_state = torch.load(prev_ckpt)
                        model.load_state(prev_ckpt, lr)

                    model.train_model(type, count, train_loader, valid_loader, epochs, lr, log, save, 
                              trial=trial, new_ckpt=ckpt, train_ckpts=save_training)
                elif mod == "EP":
                    model = ep_net(type='cond_gaussian', dimensions=params["dimensions"], cost_energy=params["cost_energy"], batch_size=params["batch_size"])
                    print("Model: ", mod)
                    ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                    save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_1 + ".json"
                    if d > 0 :
                        if d == 1:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_2 + ".json"
                        elif d == 2:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3  + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_3 + ".json"
                        elif d == 3:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_4 + ".json"
                        elif d == 4:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_5 + ".json"
                        elif d == 5:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_6 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_6 + ".json"
                        
                        saved_state = torch.load(prev_ckpt)
                        model.load_state(prev_ckpt, lr)

                    model.train_model(type, count, train_loader, valid_loader, epochs, params['dynamics'], lr=lr, log=log, save=save, 
                              trial=trial, new_ckpt=ckpt, train_ckpts=save_training)
                elif mod == "HNET":
                    model = hn().to(device)
                    optim = torch.optim.Adam(model.parameters(), lr=lr)
                    print("Model: ", mod)
                    ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                    save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_1 + ".json"
                    if d > 0 :
                        if d == 1:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_2 + ".json"
                        elif d == 2:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3  + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_3 + ".json"
                        elif d == 3:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_4 + ".json"
                        elif d == 4:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_5 + ".json"
                        elif d == 5:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial) + ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_6 + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_6 + ".json"
                        saved_state = torch.load(prev_ckpt)
                        model.load_state_dict(saved_state['model_state_dict'])
                        optim = torch.optim.Adam(model.parameters(), lr=lr)

                    print(f"Epoch 0 / {epochs} for dataset {data} ...")
                    val_loss, val_acc = hn_evaluate(model, valid_loader, criterion, device)
                    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    if log:
                        log_entry = {
                            "valid accuracy": val_acc,
                            "train loss": 0,
                            "epoch": 0,
                        }
                        wandb.log(log_entry)

                    # if save == "yes":
                    #     log_dir = f"JSON_logs/{mod}/Trial_{trial}"
                    #     os.makedirs(log_dir, exist_ok=True)
                    #     json_log_path = os.path.join(log_dir, f"HNET_{type}_{count}.json")

                    #     print(f"Saving training log to {json_log_path}")
                    #     with open(json_log_path, "w") as f:
                    #         json.dump(log_history, f, indent=4)

                    #     print(f"Wandb log saved to {json_log_path}")

                    #     # original save model
                    #     self.save_model(new_ckpt)
                    #     self.save_training_dynamics(train_acc, val_acc, trial, train_ckpts)

                    for epoch_i in range(epochs):
                        print(f"Epoch {epoch_i + 1} / {epochs} for dataset {data} ...")
                        train_loss = hn_train(model, train_loader, criterion, optim, device)
                        val_loss, val_acc = hn_evaluate(model, valid_loader, criterion, device)
                        print(f"Train Loss: {train_loss:.4f}")
                        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                        if log:
                            log_entry = {
                                "valid accuracy": val_acc,
                                "train loss": train_loss,
                                "epoch": epoch_i + 1,
                            }
                            wandb.log(log_entry)
                    
                    # if save == "yes":
                    #     json_log_path = os.path.join(log_dir, f"HNET_{type}_{count}.json")

                    #     print(f"Saving training log to {json_log_path}")
                    #     with open(json_log_path, "w") as f:
                    #         json.dump(log_history, f, indent=4)

                    #     print(f"Wandb log saved to {json_log_path}")

                    #     # original save model
                    #     self.save_model(new_ckpt)
                    #     self.save_training_dynamics(train_acc, val_acc, trial, train_ckpts)

                else :
                    raise ValueError("Unkown algorithm. Please choose from BP, TP, DTP, FWDTP, or KAN.")
            if log :
                wandb.finish()
            
    print("DONE")

if __name__ == "__main__":
    # models = ["BP", "PC", "EP", "DTP", "KAN", "HNET"]
    # models = ["PC"]
    # models = ["EP"]
    # models = ["BP"]
    # models = ["DTP"]
    models = ["HNET", "BP"]

    # datasets = ['m', 'f']

    datasets = ['m', 'f', 'm', 'f', 'm', 'f']
    # datasets = ['m', 'm', 'm', 'm', 'm', 'm']

    if 'c' in datasets or 's' in datasets:
        larger = True
    else:
        larger = False

    print("Larger input dimensions? : ", larger)

    # TESINGING AND MODEL PARAMETERS

    epochs = 5
    batch_size = 1000

    # epochs = 5
    # epochs = 1
    epochs_backward = 5
    # batch_size = 64
    # batch_size = 5000

    test = True  # from FWDTP paper's main.py
    # label_augentation = False  # from FWDTP paper's main.py
    depth = 6
    direct_depth = 1

    # for TP
    lr = 0.01
    lr_backward = 1e-3
    std_backward = 0.01
    loss_feedback = "DTP"
    sparse_ratio = 0.5 #[0.1, 0.5, 0.9] # for FWDTP
    sparse_ratio_str = f"-sparse-{sparse_ratio}" if 0 <= sparse_ratio <= 1 else ""

    # input and output dimensions depend on the dataset
    hid_dim = 256

    # log = False # for wandb visuals

    log = True
    if len(datasets) > 1:
        save = "yes"
    else:
        save = "no"

    n_inference_steps = 100
    inference_lr = 0.01

    TRIALS = 3
    main(TRIALS, models, datasets, epochs, epochs_backward, batch_size, 
         test, depth, direct_depth, lr, lr_backward, std_backward, 
         loss_feedback, sparse_ratio_str, hid_dim, log, save,
         n_inference_steps, inference_lr, larger=larger)
    