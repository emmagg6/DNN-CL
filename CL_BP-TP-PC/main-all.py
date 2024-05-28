'''

CL training -- 100 tests runs -- BP, DTP, FWDTP trained in mnist, then fashionmnist, then mnist

Checkpoint tests for accuracy and loss for each model at the start and end of training.

'''

from utils import *
from dataset import make_MNIST, make_FashionMNIST, make_CIFAR10, make_STL10

from Models.BP.bp_nn import bp_net
from Models.TP.tp_nn import tp_net
from Models.PC.pc_nn import pc_net
from Models.KAN.kan_nn import kan_net
from Models.PCDTP.pcdtp_nn import pcdtp_net
from Models.PC.pc_layers import ConvLayer, MaxPool, ProjectionLayer, FCLayer
from Models.PCDTP.pcdtp_layers import ConvLayer_dtp, MaxPool_dtp, ProjectionLayer_dtp, FCLayer_dtp

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
    set_seed(1)
    device = set_device()
    print(f"DEVICE: {device}")

    for mod in models:

        for trial in range(0, TRIALS + 1):
            print("\n -------------------------------------")
            print(f"TRIAL: {trial}")
            print(" -------------------------------------\n")


            name = mod + "-" + str(trial)
            name = str(name)

            set_seed(trial)
            # name = {"ff1": "forward_function_1",
            #     "ff2": "forward_function_2",
            #     "bf1": "backward_function_1",
            #     "bf2": "backward_function_2"}
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
                    },
                    'name': {mod}
                }
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
                params["name"] = mod

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
                # lr = 0.0005 # default in the paper
                loss_fn, loss_fn_deriv = parse_loss_function("crossentropy")

                if not larger:

                    l1 = ConvLayer(input_size=28, num_channels=1, num_filters=6, batch_size=batch_size, kernel_size=5, learning_rate=lr, f=relu, df=relu_deriv, device=device)

                    # Max pooling layer with kernel size 2x2
                    l2 = MaxPool(2, device=device)

                    # Convolutional layer with input size 14x14 (after max pooling), 6 input channels, 16 output filters, kernel size 5x5
                    l3 = ConvLayer(input_size=12, num_channels=6, num_filters=16, batch_size=batch_size, kernel_size=5, learning_rate=lr, f=relu, df=relu_deriv, device=device)

                    # Projection layer with input size corresponding to the output size of the previous conv layer, 16x5x5
                    l4 = ProjectionLayer(input_size=(64, 16, 8, 8), output_size=120, f=relu, df=relu_deriv, learning_rate=lr, device=device)

                    # Fully connected layer
                    l5 = FCLayer(input_size=120, output_size=84, batch_size=64, learning_rate=lr, f=relu, df=relu_deriv, device=device)

                    # Final fully connected layer with 10 output classes for MNIST
                    l6 = FCLayer(input_size=84, output_size=10, batch_size=64, learning_rate=lr, f=F.softmax, df=linear_deriv, device=device)

                    # List of layers
                    layers = [l1, l2, l3, l4, l5, l6]

                else:
                ## input for cifar 10, so 28x28 --> 28x28x3 to account for rbg
                    l1 = ConvLayer(input_size=28, num_channels=3, num_filters=6, batch_size=batch_size, kernel_size=5, learning_rate=lr, f=relu, df=relu_deriv, device=device)
                    l2 = MaxPool(2, device=device)
                    l3 = ConvLayer(input_size=12, num_channels=6, num_filters=16, batch_size=batch_size, kernel_size=5, learning_rate=lr, f=relu, df=relu_deriv, device=device)
                    l4 = MaxPool(2, device=device)
                    l5 = ConvLayer(input_size=4, num_channels=16, num_filters=120, batch_size=batch_size, kernel_size=5, learning_rate=lr, f=relu, df=relu_deriv, device=device)
                    l6 = FCLayer(input_size=120, output_size=84, batch_size=64, learning_rate=lr, f=relu, df=relu_deriv, device=device)
                    l7 = FCLayer(input_size=84, output_size=10, batch_size=64, learning_rate=lr, f=F.softmax, df=linear_deriv, device=device)
                    layers = [l1, l2, l3, l4, l5, l6, l7]
                
                params["name"] = mod
            elif mod == "PCDTP":
                loss_fn, loss_fn_deriv = parse_loss_function("crossentropy")
                if not larger:
                    l1 = ConvLayer_dtp(input_size=28, num_channels=1, num_filters=6, batch_size=batch_size, kernel_size=5, learning_rate=lr, f=relu, df=relu_deriv, device=device)
                    l2 = MaxPool_dtp(2, device=device)
                    l3 = ConvLayer_dtp(input_size=12, num_channels=6, num_filters=16, batch_size=batch_size, kernel_size=5, learning_rate=lr, f=relu, df=relu_deriv, device=device)
                    l4 = ProjectionLayer_dtp(input_size=(64, 16, 8, 8), output_size=120, f=relu, df=relu_deriv, learning_rate=lr, device=device)
                    l5 = FCLayer_dtp(input_size=120, output_size=84, batch_size=64, learning_rate=lr, f=relu, df=relu_deriv, device=device)
                    l6 = FCLayer_dtp(input_size=84, output_size=10, batch_size=64, learning_rate=lr, f=F.softmax, df=linear_deriv, device=device)
                    layers = [l1, l2, l3, l4, l5, l6]
                params["name"] = mod
            elif mod == "KAN":
                params["name"] = mod
            else :
                raise ValueError("Unkown algorithm. Please choose from BP, TP, DTP, FWDTP, KAN.")

            
            if log :
                # print("Logging")
                wandb.init(project="DCL_m-f-m-f-m", config=params, name=name,  reinit=True)

            ########### DATA ########### AND LEARNING RATE
            for d, data in enumerate(datasets): 
                if data == "m":
                    print("making MNIST ...")
                    in_dim = 784
                    out_dim = 10
                    if mod == "PC" or mod == "KAN":
                        trainset, validset = make_MNIST(out_dim, test, True)
                    else:
                        trainset, validset, testset = make_MNIST(out_dim, test)
                    
                    stepsize = 0.04
                    if mod == "KAN":
                        lr = 0.005 #0.0005
                    else: 
                        lr = 0.1
                elif data == "f":
                    print("making FashionMNIST ...")
                    in_dim = 784
                    out_dim = 10
                    if mod == "PC" or mod == "KAN":
                        trainset, validset = make_FashionMNIST(out_dim, test, True)
                    else:
                        trainset, validset, testset = make_FashionMNIST(out_dim, test)
                    

                    stepsize = 0.004
                    if mod == "KAN":
                        lr = 0.005
                    else :
                        lr = 0.9


                elif data == "c":
                    print("making CIFAR10 ...")
                    in_dim = 3072
                    out_dim = 10
                    # trainset, validset, testset = make_CIFAR10(out_dim, test)
                    if mod == "PC" or mod == "KAN":
                        trainset, validset = make_CIFAR10(out_dim, test, True)
                    else:
                        trainset, validset, testset = make_CIFAR10(out_dim, test)

                    
                    stepsize = 0.05
                    if mod == "KAN":
                        lr = 0.005
                    else:
                        lr = 0.1

                elif data == "s":
                    print("making STL10 ...")
                    in_dim = 3072
                    out_dim = 10
                    # trainset, validset, testset = make_STL10(out_dim, test)
                    if mod == "PC" or mod == "KAN":
                        trainset, validset = make_STL10(out_dim, test, True)
                    else:
                        trainset, validset, testset = make_STL10(out_dim, test)

                    stepsize = 0.05
                    if mod == "KAN":
                        lr = 0.005
                    else:
                        lr = 0.1
                else :
                    raise ValueError("Unkown dataset. Please choose from MNIST ('m'), FashionMNIST ('f'), CIFAR10 ('c').")

    # <torch.utils.data.dataloader.DataLoader object at 0x1020366a0> 

                loss_function = nn.CrossEntropyLoss(reduction="sum")

                
                if mod == "PC" or mod == "PCDTP":
                    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
                    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
                # elif mod == "PCDTP":
                #     train_loader = torch.utils.data.DataLoader(trainset(torch.stack([data[0] for data in trainset]), torch.stack([data[1] for data in trainset])), batch_size=batch_size, shuffle=True)
                #     valid_loader = torch.utils.data.DataLoader(validset(torch.stack([data[0] for data in validset]), torch.stack([data[1] for data in validset])), batch_size=batch_size, shuffle=False)
                elif mod == "KAN":
                    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, pin_memory=True, shuffle=True)
                    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, pin_memory=True, shuffle=False)
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
                str_datasets_trials_1 = "-" + "m-f-m-f-m" + "-" + datasets[0]
                if len(datasets) > 1 :
                    str_datasets_trials_2 = "-" + "m-f-m-f-m" + "-" + datasets[0] + "-" + datasets[1]
                if len(datasets) > 2 :
                    str_datasets_trials_3 = "-" + "m-f-m-f-m" + "-" + datasets[0] + "-" + datasets[1] + "-" + datasets[2]
                if len(datasets) > 3 :
                    str_datasets_trials_4 = "-" + "m-f-m-f-m" + "-" + datasets[0] + "-" + datasets[1] + "-" + datasets[2] + "-" + datasets[3]
                if len(datasets) > 4 :
                    str_datasets_trials_5 = "-" + "m-f-m-f-m" + "-" + datasets[0] + "-" + datasets[1] + "-" + datasets[2] + "-" + datasets[3] + "-" + datasets[4]
                if len(datasets) > 5 :
                    str_datasets_trials_6 = "-" + "m-f-m-f-m" + "-" + datasets[0] + "-" + datasets[1] + "-" + datasets[2] + "-" + datasets[3] + "-" + datasets[4] + "-" + datasets[5]



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

                    model.train_model(train_loader, valid_loader, epochs, lr, log, save, 
                                    trial=trial, new_ckpt= ckpt, train_ckpts=save_training)
                    # print("trained BP")

                elif mod == "PC" or mod == "PCDTP":
                    if mod == "PC":
                        model = pc_net(layers, num_inference_steps, inference_lr, loss_fn = loss_fn, loss_fn_deriv = loss_fn_deriv, device=device)
                    elif mod == "PCDTP":
                        model = pcdtp_net(layers, num_inference_steps, inference_lr, loss_fn = loss_fn, loss_fn_deriv = loss_fn_deriv, device=device)
                    print("Model: ", mod)

                    ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial)
                    save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_1 + ".json"
                    prev_ckpt = "None"
                    log_dir = "checkpoints/" + mod + "/logs/" + mod + str_datasets_trials_1 + "-trial" + str(trial)
                    # make directories if not there
                    if not os.path.exists(ckpt):
                        os.makedirs(ckpt)
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                    if d > 0 :
                        if d == 1:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial)
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial)
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_2 + ".json"
                            if not os.path.exists(ckpt):
                                os.makedirs(ckpt)
                            if not os.path.exists(log_dir):
                                os.makedirs(log_dir)
                        elif d == 2:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial)
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3  + "-trial" + str(trial) 
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_3 + ".json"
                            if not os.path.exists(ckpt):
                                os.makedirs(ckpt)
                            if not os.path.exists(log_dir):
                                os.makedirs(log_dir)
                        elif d == 3:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3 + "-trial" + str(trial)
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial)
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_4 + ".json"
                            if not os.path.exists(ckpt):
                                os.makedirs(ckpt)
                            if not os.path.exists(log_dir):
                                os.makedirs(log_dir)
                        elif d == 4:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial)
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial)
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_5 + ".json"
                            if not os.path.exists(ckpt):
                                os.makedirs(ckpt)
                            if not os.path.exists(log_dir):
                                os.makedirs(log_dir)
                        elif d == 5:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial)
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_6 + "-trial" + str(trial)
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_6 + ".json"
                            if not os.path.exists(ckpt):
                                os.makedirs(ckpt)
                            if not os.path.exists(log_dir):
                                os.makedirs(log_dir)
                        model.load_model(prev_ckpt)
                        # train(self,dataset,testset,n_epochs,n_inference_steps,logdir,savedir, old_savedir,save_every=1,print_every=10):
                    train_data = list(iter(trainloader))
                    valid_data = list(iter(validloader))
                    model.train(train_data[0:-2], valid_data[0:-2], epochs, num_inference_steps, "log", ckpt, prev_ckpt, log = log)

                elif mod == "TP" or mod == "DTP" or mod == "FWDTP":
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

                    model.train(train_loader, valid_loader, epochs, lr, lr_backward, std_backward, stepsize, 
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

                    model.train_model(train_loader, valid_loader, epochs, lr, log, save, 
                              trial=trial, new_ckpt=ckpt, train_ckpts=save_training)


                else :
                    raise ValueError("Unkown algorithm. Please choose from BP, TP, DTP, FWDTP, or KAN.")
            if log :
                wandb.finish()
            
    print("DONE")

if __name__ == "__main__":
    # models = ["BP", "DTP", "FWDTP", "PC", "KAN"]
    models = ["KAN"]
    datasets = ['m', 'f', 'm', 'f', 'm']

    if 'c' in datasets or 's' in datasets:
        larger = True
    else:
        larger = False

    print("Larger input dimensions? : ", larger)

    # TESINGING AND MODEL PARAMETERS
    epochs = 5
    epochs_backward = 5
    # batch_size = int(256)
    batch_size = 64
    test = True  # from FWDTP paper's main.py
    # label_augentation = False  # from FWDTP paper's main.py
    depth = 6
    direct_depth = 1

    # for TP
    lr = 0.1
    lr_backward = 1e-3
    std_backward = 0.01
    loss_feedback = "DTP"
    sparse_ratio = 0.1 #[0.1, 0.5, 0.9] # for FWDTP
    sparse_ratio_str = f"-sparse-{sparse_ratio}" if 0 <= sparse_ratio <= 1 else ""

    # input and output dimensions depend on the dataset
    hid_dim = 256

    log = True # for wandb visuals
    save = "yes"

    n_inference_steps = 50
    inference_lr = 0.1

    TRIALS = 5
    main(TRIALS, models, datasets, epochs, epochs_backward, batch_size, 
         test, depth, direct_depth, lr, lr_backward, std_backward, 
         loss_feedback, sparse_ratio_str, hid_dim, log, save,
         n_inference_steps, inference_lr, larger=larger)