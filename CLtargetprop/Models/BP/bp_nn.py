from Models.BP.bp_layers import bp_layers

import torch
from torch import nn
import os
import time
import wandb
import json



class bp_net(nn.Module):
    def __init__(self, depth, in_dim, hid_dim, out_dim, loss_function, device, params=None):
        super(bp_net, self).__init__()
        self.depth = depth
        self.loss_function = loss_function
        self.device = device
        self.layers = nn.ModuleList(self.init_layers(in_dim, hid_dim, out_dim, params))
        self.opt = False

    def init_layers(self, in_dim, hid_dim, out_dim, params):
        layers = []
        dims = [in_dim] + [hid_dim] * (self.depth - 1) + [out_dim]
        for d in range(self.depth):
            layers.append(bp_layers(dims[d], dims[d + 1], self.device, params))
        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # Call the forward method of each layer
        return x

    def calculate_accuracy(self, loader):
        """Calculates accuracy on the given data loader."""
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

    def train_model(self, train_loader, valid_loader, epochs, lr, log, save, 
                    trial = 0, save_ckpts='', new_ckpt= '',train_ckpts = ''):
        if self.opt == False:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        #### for epoch 0
        epoch = 0
        # Set the model to training mode
        self.train()

        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        eps = []
        
        initial_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self(x)
            loss = self.loss_function(y_pred, y)
            initial_train_loss += loss.item()
        
        initial_train_loss /= len(train_loader.dataset)
        initial_train_acc = self.calculate_accuracy(train_loader)

        # Evaluate the model on the validation set before training starts (epoch 0)
        self.eval()
        initial_test_loss, initial_test_acc = self.external_test(valid_loader)
        
        train_losses.append(initial_train_loss)
        train_accuracies.append(initial_train_acc)
        test_losses.append(initial_test_loss)
        test_accuracies.append(initial_test_acc)
        eps.append(epoch)

        # if save == 'yes':
        #     self.save_initial_results(initial_train_loss, initial_train_acc, 
        #                               initial_test_loss, initial_test_acc,
        #                               trial, save_ckpts)

        # Log the initial validation loss and accuracy (epoch 0)
        if log:
            wandb.log({
                "epoch": epoch,
                "train loss": initial_train_loss,
                "train accuracy": initial_train_acc,
                "valid loss": initial_test_loss,
                "valid accuracy": initial_test_acc
            })
        
        print(f"Epoch: {epoch}, Train Loss {initial_train_loss}, Train Acc {initial_train_acc}, Valid Loss: {initial_test_loss}, Valid Acc: {initial_test_acc}")

        # Training loop starts at epoch 1
        for epoch in range(1, epochs + 1):
            self.train()  # Set the model to training mode
            total_train_loss = 0

            # Training phase
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self(x)
                loss = self.loss_function(y_pred, y)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            train_loss = total_train_loss / len(train_loader.dataset)
            train_acc = self.calculate_accuracy(train_loader)  # Calculate training accuracy

            # testing phase
            test_loss, test_acc = self.external_test(valid_loader)  # Calculate validation loss and accuracy

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            trials.append(epoch)

            # Logging after each epoch of training
            if log:
                wandb.log({
                    "epoch": epoch,
                    "train loss": train_loss,
                    "train accuracy": train_acc,
                    "valid loss": test_loss,
                    "valid accuracy": test_acc
                })
            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Train Acc: {train_acc}, Valid Loss: {test_loss}, Valid Acc: {test_acc}")

        if save == 'yes':
            self.save_model(new_ckpt)
            self.save_training_dynamics(train_losses, train_accuracies, test_losses, test_accuracies, trial, train_ckpts)


#--------------------- SAVING ---------------------
# DOES NOT WORK CORRECTLY, OPTIMIZER NEEDS TO BE SAVED AND LOADED AS WELL FOR CL
    # def save_model(self, path="checkpoints/bp/params.pth"):
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     torch.save(self.state_dict(), path)

    # def load_state(self, path):
    #     self.load_state_dict(torch.load(path))
 
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

        # Check if the file exists and has content
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r") as file:
                data = json.load(file)
        else:
            # Initialize the dictionary with lists to store results
            data = [{
                "Trial": [],
                "Train Loss": [],
                "Train Acc": [],
                "Valid Loss": [],
                "Valid Acc": []
            }]

        # Append new results to each list within the first dictionary entry
        data[0]["Trial"].append(trial)
        data[0]["Train Loss"].append(train_loss)
        data[0]["Train Acc"].append(train_acc)
        data[0]["Valid Loss"].append(valid_loss)
        data[0]["Valid Acc"].append(valid_acc)

        # Write the updated dictionary back to the file
        with open(path, "w") as file:
            json.dump(data, file, indent=4)

    def save_training_dynamics(self, train_losses, train_accuracies, test_losses, test_accuracies, trial, ckpt):
        path = ckpt
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Check if the file exists and has content
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r") as file:
                data = json.load(file)
        else:
            # Initialize the dictionary with lists to store results
            data = [{
                "Trial": [],
                "Train Losses": [],
                "Train Accuracies": [],
                "Test Losses": [],
                "Test Accuracies": []
            }]

        # Append new results to each list within the first dictionary entry
        data[0]["Trial"].append(trial)
        data[0]["Train Losses"].append(train_losses)
        data[0]["Train Accuracies"].append(train_accuracies)
        data[0]["Test Losses"].append(test_losses)
        data[0]["Test Accuracies"].append(test_accuracies)

        # Write the updated dictionary back to the file
        with open(path, "w") as file:
            json.dump(data, file, indent=4)

#---------------------------------------------------

    def load_state(self, path, lr):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        # Initialize the optimizer here with the current model parameters
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.opt = True
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def external_test(self, test_loader):
        """Tests the model on the given data loader and returns loss and accuracy."""
        self.eval()  # Set the model to evaluation mode
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
        # Normalize the test loss by the total number of samples
        return loss / total, accuracy