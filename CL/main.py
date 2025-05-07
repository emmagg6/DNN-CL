'''

CL training

Checkpoint tests for accuracy of each model after each epoch of training.

'''

from utils import *
from dataset import make_MNIST, make_FashionMNIST, make_CIFAR10, make_STL10

from Models.BP.bp_nn import bp_net
from Models.TP.tp_nn import tp_net
# from Models.PC.pc_nn import pc_net
from Models.PC.pc_nn_test import pc_net
from Models.KAN.kan_nn import kan_net
from Models.EP.ep_nn import ep_net
from Models.PC.pc_layers import ConvLayer, MaxPool, ProjectionLayer, FCLayer


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

        for trial in range(1, TRIALS + 1):
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

            lr = 0.1
            if mod == "KAN" or mod == "EP" or mod == "PC":
                lr = 0.005
            stepsize = 0.04

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
                # lr = 0.0005 # default in the paper
                loss_fn, loss_fn_deriv = parse_loss_function("crossentropy")

                if not larger: # then mnist size

                    l1 = ConvLayer(input_size=28, num_channels=1, num_filters=6, batch_size=batch_size, kernel_size=5, learning_rate=lr, f=relu, df=relu_deriv, device=device)

                    # Max pooling layer with kernel size 2x2
                    l2 = MaxPool(2, device=device)

                    # Convolutional layer with input size 14x14 (after max pooling), 6 input channels, 16 output filters, kernel size 5x5
                    l3 = ConvLayer(input_size=12, num_channels=6, num_filters=16, batch_size=batch_size, kernel_size=5, learning_rate=lr, f=relu, df=relu_deriv, device=device)

                    # Projection layer with input size corresponding to the output size of the previous conv layer, 16x5x5
                    l4 = ProjectionLayer(input_size=(64, 16, 8, 8), output_size=120, f=relu, df=relu_deriv, learning_rate=lr, device=device)

                    # Fully connected layer
                    l5 = FCLayer(input_size=120, output_size=84, batch_size=64, learning_rate=lr, f = relu, df = relu_deriv, device=device)

                    # Final fully connected layer with 10 output classes for MNIST
                    l6 = FCLayer(input_size=84, output_size=10, batch_size=64, learning_rate=lr, f = F.softmax, df= linear_deriv, device=device)

                    # List of layers
                    layers = [l1, l2, l3, l4, l5, l6]

                else:
                ## input for cifar 10, so 28x28 --> 32x32x3 to account for rbg

                    l1 = ConvLayer(input_size=32, num_channels=3, num_filters=6, batch_size=batch_size, kernel_size=5, learning_rate=lr, f=relu, df=relu_deriv, device=device)

                    l2 = MaxPool(2, device=device)

                    l3 = ConvLayer(input_size=14, num_channels=6, num_filters=16, batch_size=batch_size, kernel_size=5, learning_rate=lr, f=relu, df=relu_deriv, device=device)

                    l4 = ProjectionLayer(input_size=(64, 16, 10, 10), output_size=200, f=relu, df=relu_deriv, learning_rate=lr, device=device)

                    l5 = FCLayer(input_size=200, output_size=150, batch_size=batch_size, learning_rate=lr, f=relu, df=relu_deriv, device=device)

                    l6 = FCLayer(input_size=150, output_size=10, batch_size=64, learning_rate=lr, f = F.softmax, df = linear_deriv, device=device)

                    layers = [l1, l2, l3, l4, l5, l6]
                
                params["name"] = mod
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
            else :
                raise ValueError("Unkown algorithm. Please choose from BP, TP, DTP, FWDTP, KAN.")

            
            if log :
                # print("Logging")
                wandb.init(project="trials", config=params, name=name,  reinit=True)

            ########### DATA ########### AND LEARNING RATE
            for d, data in enumerate(datasets): 
                if data == "m":
                    print("making MNIST ...")
                    in_dim = 784
                    out_dim = 10
                    if mod == "PC" or mod == "KAN":
                        trainset, validset = make_MNIST(out_dim, test, pc = True)
                    elif mod == "EP":
                        params['dimensions'] = [784, batch_size*out_dim, out_dim]
                        trainset, validset = make_MNIST(out_dim, test, ep = True)
                    else:
                        trainset, validset, testset = make_MNIST(out_dim, test)

                elif data == "f":
                    print("making FashionMNIST ...")
                    in_dim = 784
                    out_dim = 10
                    if mod == "PC" or mod == "KAN":
                        trainset, validset = make_FashionMNIST(out_dim, test, pc = True)
                    elif mod == "EP":
                        params['dimensions'] = [784, batch_size*out_dim, out_dim]
                        trainset, validset = make_FashionMNIST(out_dim, test, ep = True)
                    else:
                        trainset, validset, testset = make_FashionMNIST(out_dim, test)
                    


                elif data == "c":
                    print("making CIFAR10 ...")
                    in_dim = 3072
                    out_dim = 10
                    # trainset, validset, testset = make_CIFAR10(out_dim, test)
                    if mod == "PC" or mod == "KAN":
                        trainset, validset = make_CIFAR10(out_dim, test, True)
                    else:
                        trainset, validset, testset = make_CIFAR10(out_dim, test)

                elif data == "s":
                    print("making STL10 ...")
                    in_dim = 3072
                    out_dim = 10
                    # trainset, validset, testset = make_STL10(out_dim, test)
                    if mod == "PC" or mod == "KAN":
                        trainset, validset = make_STL10(out_dim, test, True)
                    else:
                        trainset, validset, testset = make_STL10(out_dim, test)
                else :
                    raise ValueError("Unkown dataset. Please choose from MNIST ('m'), FashionMNIST ('f'), CIFAR10 ('c').")

    # <torch.utils.data.dataloader.DataLoader object at 0x1020366a0> 

                loss_function = nn.CrossEntropyLoss(reduction="sum")

                
                if mod == "PC":
                    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
                    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
                elif mod == "KAN":
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

                    model.train_model(train_loader, valid_loader, epochs, lr, log, save, 
                                    trial=trial, new_ckpt= ckpt, train_ckpts=save_training)
                    # print("trained BP")

                elif mod == "PC":
                    # model = pc_net(layers, num_inference_steps, inference_lr, loss_fn = loss_fn, loss_fn_deriv = loss_fn_deriv, device=device)
                    model = pc_net(depth, in_dim, hid_dim, out_dim, loss_function, device, batch_size, params=params)
                    print("Model: ", mod)

                    ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial)+ ".pth"
                    save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_1 + ".json"
                    prev_ckpt = "None"
                    log_dir = "checkpoints/" + mod + "/logs/" + mod + str_datasets_trials_1 + "-trial" + str(trial)
                    # make directories if not there
                    # if not os.path.exists(ckpt):
                    #     os.makedirs(ckpt)
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                    if d > 0 :
                        if d == 1:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial)+ ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial)+ ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_2 + ".json"
                            # if not os.path.exists(ckpt):
                            #     os.makedirs(ckpt)
                            if not os.path.exists(log_dir):
                                os.makedirs(log_dir)
                        elif d == 2:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_2 + "-trial" + str(trial)+ ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3  + "-trial" + str(trial) + ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_3 + ".json"
                            # if not os.path.exists(ckpt):
                            #     os.makedirs(ckpt)
                            if not os.path.exists(log_dir):
                                os.makedirs(log_dir)
                        elif d == 3:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_3 + "-trial" + str(trial)+ ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial)+ ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_4 + ".json"
                            # if not os.path.exists(ckpt):
                            #     os.makedirs(ckpt)
                            if not os.path.exists(log_dir):
                                os.makedirs(log_dir)
                        elif d == 4:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_4 + "-trial" + str(trial)+ ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial)+ ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_5 + ".json"
                            # if not os.path.exists(ckpt):
                            #     os.makedirs(ckpt)
                            if not os.path.exists(log_dir):
                                os.makedirs(log_dir)
                        elif d == 5:
                            prev_ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_5 + "-trial" + str(trial)+ ".pth"
                            ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_6 + "-trial" + str(trial)+ ".pth"
                            save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_6 + ".json"
                            # if not os.path.exists(ckpt):
                            #     os.makedirs(ckpt)
                            if not os.path.exists(log_dir):
                                os.makedirs(log_dir)

                        # saved_state = torch.load(prev_ckpt)
                        # model.load_state(saved_state)
                        model.load_state(prev_ckpt)
                        # train(self,dataset,testset,n_epochs,n_inference_steps,logdir,savedir, old_savedir,save_every=1,print_every=10):
                    train_data = list(iter(trainloader))
                    valid_data = list(iter(validloader))
                    # model.train(train_data[0:-2], valid_data[0:-2], epochs, num_inference_steps, "log", ckpt, prev_ckpt, log = log)
                    # model.train_model(trainloader, validloader, epochs, lr, log, save, 
                    #                 trial=trial, new_ckpt= ckpt, train_ckpts=save_training)
                    # model.train_model(train_data, valid_data, epochs, lr, log, save, 
                    #                 trial=trial, new_ckpt= ckpt, train_ckpts=save_training)
                    # model.train_model(train_data, valid_data, epochs, trainloader, validloader, batch_size, log, save, trial=trial, new_ckpt= ckpt, train_ckpts=save_training)
                    # model.train_model(train_data, valid_data, epochs, trainloader, validloader, batch_size, log, save)
                    
                    model.train_model(train_data, valid_data, epochs, trainloader, validloader, batch_size, log, save, trial=trial, new_ckpt= ckpt, train_ckpts=save_training)

                    # need to ad check points/savins
                    # model.train_model(trainloader, validloader, epochs, trainloader, batch_size)

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

                    model.train_model(train_loader, valid_loader, epochs, params['dynamics'], lr=lr, log=log, save=save, 
                              trial=trial, new_ckpt=ckpt, train_ckpts=save_training)

                else :
                    raise ValueError("Unkown algorithm. Please choose from BP, TP, DTP, FWDTP, or KAN.")
            if log :
                wandb.finish()
            
    print("DONE")

if __name__ == "__main__":
    # models = ["BP", "DTP", "FWDTP", "PC", "KAN"]
    models = ["PC"]
    # models = ["BP", "DTP", "EP", "KAN", "FWDTP"]
    # models = ["BP", "PC", "DTP", "EP"]

    datasets = ['m', 'f', 'm', 'f', 'm']

    if 'c' in datasets or 's' in datasets:
        larger = True
    else:
        larger = False

    print("Larger input dimensions? : ", larger)

    # TESINGING AND MODEL PARAMETERS
    epochs = 5
    # epochs = 1
    epochs_backward = 5
    batch_size = 64
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

    TRIALS = 100
    main(TRIALS, models, datasets, epochs, epochs_backward, batch_size, 
         test, depth, direct_depth, lr, lr_backward, std_backward, 
         loss_feedback, sparse_ratio_str, hid_dim, log, save,
         n_inference_steps, inference_lr, larger=larger)
    