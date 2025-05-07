import torch
from torch import nn
import os
import time
import wandb

from Models.BP.bp_fcns import ParameterizedFunction


class bp_layers(nn.Module):
    def __init__(self, in_dim, out_dim, device, params):
        super(bp_layers, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        # Initialize forward functions as instances of ParameterizedFunction
        self.forward_function_1 = ParameterizedFunction(in_dim, in_dim, params["ff1"])
        self.forward_function_2 = ParameterizedFunction(in_dim, out_dim, params["ff2"])


    def forward(self, x):
        x = self.forward_function_1(x)  # Call the forward method of the first function
        x = self.forward_function_2(x)  # Call the forward method of the second function
        return x

    @staticmethod
    def set_function(in_dim, out_dim, params):
        # This method might not be necessary if you're directly initializing ParameterizedFunction above
        if params["type"] == "parameterized":
            return ParameterizedFunction(in_dim, out_dim, params)
        else:
            raise NotImplementedError()



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

    def train_model(self, train_loader, valid_loader, epochs, lr, log, save):
        if self.opt == False:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        #### for epoch 0
        epoch = 0
        # Set the model to training mode
        self.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self(x)
            loss = self.loss_function(y_pred, y)
            train_loss += loss.item()
        
        train_loss /= len(train_loader.dataset)
        train_acc = self.calculate_accuracy(train_loader)

        # Evaluate the model on the validation set before training starts (epoch 0)
        self.eval()
        initial_test_loss, initial_test_acc = self.external_test(valid_loader)

        # Log the initial validation loss and accuracy (epoch 0)
        if log:
            wandb.log({
                "epoch": epoch,
                "train loss": train_loss,
                "train accuracy": train_acc,
                "valid loss": initial_test_loss,
                "valid accuracy": initial_test_acc
            })
        
        print(f"Epoch: {epoch}, Train Loss {train_loss}, Train Acc {train_acc}, Valid Loss: {initial_test_loss}, Valid Acc: {initial_test_acc}")

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
            self.save_model()

# DOES NOT WORK CORRECTLY, OPTIMIZER NEEDS TO BE SAVED AND LOADED AS WELL FOR CL
    # def save_model(self, path="checkpoints/bp/params.pth"):
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     torch.save(self.state_dict(), path)

    # def load_state(self, path):
    #     self.load_state_dict(torch.load(path))
            
    def save_model(self, path="checkpoints/bp/params.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

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