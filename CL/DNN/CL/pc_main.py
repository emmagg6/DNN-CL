'''

CL training

Checkpoint tests for accuracy of each model after each epoch of training.

'''

from utils import *

import torchvision as tv
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST

from Models.PC.pc import pc_cnn

import os
import sys
import wandb
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



def main(TRIALS, datasets, epochs, batch_size, lr, log, 
         save, n_inference_steps, inference_lr, larger=False):
    # set_seed(1)
    device = set_device()
    print(f"DEVICE: {device}")

    for trial in range(1, TRIALS + 1):
        print("\n -------------------------------------")
        print(f"TRIAL: {trial}")
        print(" -------------------------------------\n")


        name = "PC" + "-" + str(trial)
        name = str(name)

        # set_seed(trial)
        # name = {"ff1": "forward_function_1",
        #     "ff2": "forward_function_2",
        #     "bf1": "backward_function_1",
        #     "bf2": "backward_function_2"}
        params = {}
        print("Parameter Setup ... ")

        params["name"] = "PC"

        params["layers"] =nn.Sequential(

                    nn.Sequential(
                    nn.Conv2d(1,10,3),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                    ),

                    nn.Sequential(
                    nn.Conv2d(10,5,3),
                    nn.ReLU(),
                    nn.Flatten()
                    ),

                nn.Sequential(
                    nn.Linear(5*11*11,50),
                    nn.ReLU()
                    ),

                nn.Sequential(
                    nn.Linear(50,30),
                    nn.ReLU()
                    ),


                nn.Sequential(
                nn.Linear(30,10)
                )

                ).to(device)

        LossFun = nn.CrossEntropyLoss()

        
        if log :
            # print("Logging")
            wandb.init(project="IWAI", config=params, name=name,  reinit=True)

        ########### DATA ########### AND LEARNING RATE
        for d, data in enumerate(datasets): 
            if data == "m":
                print("making MNIST ...")
                train_dataset = MNIST('./data',
                    train=True,
                    transform=transforms.ToTensor(),
                    download=True)

                test_dataset = MNIST('./data',
                    train=False,
                    transform=transforms.ToTensor(),
                    download=True)

            elif data == "f":
                print("making FashionMNIST ...")
                train_dataset = FashionMNIST('./data',
                    train=True,
                    transform=transforms.ToTensor(),
                    download=True)

                test_dataset = FashionMNIST('./data',
                    train=False,
                    transform=transforms.ToTensor(),
                    download=True)
                
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)


            n_inference_steps = batch_size

            model = pc_cnn(params["layers"], LearningRate=lr)

            model.train(epochs, n_inference_steps, train_loader, test_loader, LossFun,
                        lr=lr)


        if log :
            wandb.finish()
        
    print("DONE")

if __name__ == "__main__":
    datasets = ['m', 'f', 'm', 'f', 'm']
    datasets = ['m']

    if 'c' in datasets or 's' in datasets:
        larger = True
    else:
        larger = False

    print("Larger input dimensions? : ", larger)

    # TESINGING AND MODEL PARAMETERS
    epochs = 5
    epochs_backward = 5
    batch_size = 64
 
    lr = 0.001

    # input and output dimensions depend on the dataset
    hid_dim = 256

    log = False # for wandb visuals
    if len(datasets) > 1:
        save = "yes"
    else:
        save = "no"

    n_inference_steps = 50
    inference_lr = 0.001

    TRIALS = 1
    main(TRIALS, datasets, epochs, batch_size, lr, log, save,
         n_inference_steps, inference_lr, larger=larger)
    