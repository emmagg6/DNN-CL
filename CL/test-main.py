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
            params = {}
            print("Parameter Setup ... ")


            lr = 0.01
            stepsize = 0.04
            
            if log :
                # print("Logging")
                wandb.init(project="trials", config=params, name=name,  reinit=True)

            ########### DATA ########### AND LEARNING RATE
            for d, data in enumerate(datasets): 
                if data == "m":
                    print("making MNIST ...")
                    in_dim = 784
                    out_dim = 10
                    trainset, validset, testset = make_MNIST(out_dim, test)

                elif data == "f":
                    print("making FashionMNIST ...")
                    in_dim = 784
                    out_dim = 10
                    trainset, validset, testset = make_FashionMNIST(out_dim, test)
                    

    # <torch.utils.data.dataloader.DataLoader object at 0x1020366a0> 

                loss_function = nn.CrossEntropyLoss(reduction="sum")

                
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
                model = pc_net(depth, in_dim, hid_dim, out_dim, loss_function, device, batch_size, params=params)
                print("Model: ", mod)

                ckpt = "checkpoints/" + mod + "/models/" + mod + str_datasets_trials_1 + "-trial" + str(trial)+ ".pth"
                save_training = "checkpoints/" + mod + "/TRAIN-" + mod + str_datasets_trials_1 + ".json"
                prev_ckpt = "None"
                
                parent_dir = os.path.dirname(ckpt)
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)
                
                # if not os.path.exists(log_dir):
                #     os.makedirs(log_dir)
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
                
                model.train_model(train_data, valid_data, epochs, train_loader, valid_loader, batch_size, log, save, trial=trial, new_ckpt= ckpt, train_ckpts=save_training)

                model.save_model(ckpt)

                
            if log :
                wandb.finish()
            
    print("DONE")

if __name__ == "__main__":
    models = ["PC"]

    datasets = ['m', 'f', 'm', 'f', 'm']
    # datasets = ['m', 'm']

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

    # log = True # for wandb visuals

    log = False
    if len(datasets) > 1:
        save = "yes"
    else:
        save = "no"

    n_inference_steps = 100
    inference_lr = 0.01

    TRIALS = 2
    main(TRIALS, models, datasets, epochs, epochs_backward, batch_size, 
         test, depth, direct_depth, lr, lr_backward, std_backward, 
         loss_feedback, sparse_ratio_str, hid_dim, log, save,
         n_inference_steps, inference_lr, larger=larger)
    