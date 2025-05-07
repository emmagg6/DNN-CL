'''
Adapted from https://www.kaggle.com/code/mickaelfaust/99-kolmogorov-arnold-network-conv2d-and-mnist?scriptVersionId=177633280 

by emmagg6
'''



from torch.cuda.amp import autocast
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.optim import AdamW
# from tqdm.auto import tqdm

import os
import wandb
import json
# from Models.KAN.kan_layers import KANConv2d, KANLinear2


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import numpy as np
import math

from Models.KAN.kan_layers import KANConv2d, KANLinear2, KolmogorovActivation, KANLinearFFT, KANPreprocessing, KANLinear

###################### prelim fcns ######################
@torch.jit.script
def b_splines(x, grid, k: int):
    x = x.unsqueeze(-1)
    value = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()

    p = 1
    while p <= k:
        # Compute the differences between grid points for the left and right intervals
        diff_left = grid[:, p:-1] - grid[:, : -(p + 1)]
        diff_right = grid[:, p + 1 :] - grid[:, 1:(-p)]

        # Create masks to identify non-zero differences
        mask_left = torch.ne(diff_left, 0)
        mask_right = torch.ne(diff_right, 0)

        # Compute the ratios for the left and right intervals
        ratio_left = torch.where(mask_left, (x - grid[:, : -(p + 1)]) / diff_left, torch.zeros_like(diff_left))
        ratio_right = torch.where(mask_right, (grid[:, p + 1 :] - x) / diff_right, torch.zeros_like(diff_right))

        # Update the value using the weighted average of the left and right intervals
        value = (ratio_left * value[:, :, :-1] + ratio_right * value[:, :, 1:]) / (ratio_left + ratio_right + 1e-8)

        p += 1

    return value

def curve2coeff(x, y, grid, k, eps=1e-8):
    splines = torch.einsum('ijk->jik', b_splines(x, grid, k))

    # Perform Singular Value Decomposition (SVD) on the splines matrix
    # u: left singular vectors
    # s: singular values
    # v: right singular vectors
    u, s, v = torch.linalg.svd(splines, full_matrices=False)

    # Compute the inverse of the singular values
    s_inv = torch.zeros_like(s)
    s_inv[s != 0] = 1 / s[s != 0]
    s_inv = torch.diag_embed(s_inv)

    # Compute the coefficients of the curve using the SVD components and the y values
    value = v.transpose(-2, -1) @ s_inv @ u.transpose(-2, -1) @ y.transpose(0, 1)
    value = value.permute(2, 0, 1)
    
    return value

######################################################

class kan_net(nn.Module):
    def __init__(self, in_dim, out_dim, loss_function, device, larger=False):
        super(kan_net, self).__init__()
        self.loss_function = loss_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not larger:
            self.conv_block_1 = nn.Sequential(
                KANConv2d(in_channels=in_dim, out_channels=16, kernel_size=4, stride=1, padding='same'),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_block_2 = nn.Sequential(
                KANConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding='same'),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=2)
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                KANLinear2(in_dim=32, out_dim=out_dim)
            )
        else:
            self.conv_block_1 = nn.Sequential(
                KANConv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            self.conv_block_2 = nn.Sequential(
                KANConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.conv_block_3 = nn.Sequential(
                KANConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                KANLinear2(in_dim=128, out_dim=out_dim)
            )

        self.opt = False

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        if hasattr(self, 'conv_block_3'):
            x = self.conv_block_3(x)
        x = self.classifier(x)
        return x


    def calculate_accuracy(self, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total

    def train_model(self, train_loader, valid_loader, epochs, lr, log, save, trial=0, new_ckpt='', train_ckpts=''):
        if not self.opt:
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.0005)
        epoch = 0
        self.train()

        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        eps = []
        # print("Calculating initial training loss and accuracy for epoch 0")
        initial_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self(x)
            loss = self.loss_function(y_pred, y)
            initial_train_loss += loss.item()
        
        initial_train_loss /= len(train_loader.dataset)
        initial_train_acc = self.calculate_accuracy(train_loader)

        self.eval()
        initial_test_loss, initial_test_acc = self.external_test(valid_loader)

        print(f"Epoch: {epoch}, Train Acc {initial_train_acc}, Valid Acc: {initial_test_acc}")

        train_losses.append(initial_train_loss)
        train_accuracies.append(initial_train_acc)
        test_losses.append(initial_test_loss)
        test_accuracies.append(initial_test_acc)
        eps.append(epoch)

        if log:
            wandb.log({
                "epoch": epoch,
                "train loss": initial_train_loss,
                "train accuracy": initial_train_acc,
                "valid loss": initial_test_loss,
                "valid accuracy": initial_test_acc
            })
        
        for epoch in range(1, epochs + 1):
            self.train()
            total_train_loss = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self(x)
                loss = self.loss_function(y_pred, y)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            train_loss = total_train_loss / len(train_loader.dataset)
            train_acc = self.calculate_accuracy(train_loader)

            test_loss, test_acc = self.external_test(valid_loader)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            eps.append(epoch)

            if log:
                wandb.log({
                    "epoch": epoch,
                    "train loss": train_loss,
                    "train accuracy": train_acc,
                    "valid loss": test_loss,
                    "valid accuracy": test_acc
                })
            print(f"Epoch: {epoch}, Train Acc: {train_acc}, Valid Acc: {test_acc}")

        if save == 'yes':
            self.save_model(new_ckpt)
            self.save_training_dynamics(train_losses, train_accuracies, test_losses, test_accuracies, trial, train_ckpts)
    
    def save_model(self, new_ckpt):
        path = new_ckpt
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def save_initial_results(self, train_loss, train_acc, valid_loss, valid_acc, trial, ckpt):
        path = ckpt
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r") as file:
                data = json.load(file)
        else:
            data = [{
                "Trial": [],
                "Train Loss": [],
                "Train Acc": [],
                "Valid Loss": [],
                "Valid Acc": []
            }]

        data[0]["Trial"].append(trial)
        data[0]["Train Loss"].append(train_loss)
        data[0]["Train Acc"].append(train_acc)
        data[0]["Valid Loss"].append(valid_loss)
        data[0]["Valid Acc"].append(valid_acc)

        with open(path, "w") as file:
            json.dump(data, file, indent=4)

    def save_training_dynamics(self, train_losses, train_accuracies, test_losses, test_accuracies, trial, ckpt):
        path = ckpt
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r") as file:
                data = json.load(file)
        else:
            data = [{
                "Trial": [],
                "Train Losses": [],
                "Train Accuracies": [],
                "Test Losses": [],
                "Test Accuracies": []
            }]

        data[0]["Trial"].append(trial)
        data[0]["Train Losses"].append(train_losses)
        data[0]["Train Accuracies"].append(train_accuracies)
        data[0]["Test Losses"].append(test_losses)
        data[0]["Test Accuracies"].append(test_accuracies)

        with open(path, "w") as file:
            json.dump(data, file, indent=4)

    def load_state(self, path, lr):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.opt = True
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def external_test(self, test_loader):
        self.eval()
        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self(x)
                loss += self.loss_function(y_pred, y).item()
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = correct / total
        return loss / total, accuracy