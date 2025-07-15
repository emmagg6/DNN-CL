import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import wandb  # Make sure to install wandb using 'pip install wandb'

from .hnetcon2d import HNetConv2d  # Import your custom HNetConv2d class

class hnet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 use_norm=True, n_random_edges=None, loss_function=None, device=None, learning_rate=0.001, params=None):
        super(h_nn, self).__init__()

        # Initialize your model
        self.model = HNetConv2d(in_channels, out_channels, kernel_size, stride, padding, use_norm, n_random_edges)

        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Loss function and optimizer
        self.loss_function = loss_function if loss_function is not None else nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Keep track of the current epoch
        self.current_epoch = 0

        # For logging and saving
        self.opt = False  # Flag to check if optimizer has been loaded
        self.params = params  # Any additional parameters you might need

    def forward(self, x):
        return self.model(x)

    def calculate_accuracy(self, loader):
        """Calculates accuracy on the given data loader."""
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.forward(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total if total > 0 else 0

    def external_test(self, loader):
        """Tests the model on the given data loader and returns loss and accuracy."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.forward(x)
                loss = self.loss_function(outputs, y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy

    def train_model(self, train_loader, valid_loader, num_epochs, lr, log=False, save=False,
                    trial=0, new_ckpt='', train_ckpts=''):
        # Initialize the optimizer if not already done
        if not self.opt:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.opt = True

        # Start wandb logging if log is True
        if log:
            wandb.init()  # Ensure you have called wandb.init() somewhere in your code

        train_losses = []
        train_accuracies = []
        valid_losses = []
        valid_accuracies = []
        epochs_list = []

        # Evaluate before training starts (Epoch 0)
        self.model.eval()
        initial_train_loss, initial_train_acc = self.external_test(train_loader)
        initial_valid_loss, initial_valid_acc = self.external_test(valid_loader)

        train_losses.append(initial_train_loss)
        train_accuracies.append(initial_train_acc)
        valid_losses.append(initial_valid_loss)
        valid_accuracies.append(initial_valid_acc)
        epochs_list.append(self.current_epoch)

        # Save initial results if required
        if save and new_ckpt and train_ckpts:
            self.save_initial_results(initial_train_loss, initial_train_acc, initial_valid_loss, initial_valid_acc, trial, train_ckpts)

        # Log initial results
        if log:
            wandb.log({
                "epoch": self.current_epoch,
                "train loss": initial_train_loss,
                "train accuracy": initial_train_acc,
                "valid loss": initial_valid_loss,
                "valid accuracy": initial_valid_acc
            })

        print(f"Epoch: {self.current_epoch}, Train Loss: {initial_train_loss:.4f}, Train Acc: {initial_train_acc:.4f}, "
              f"Valid Loss: {initial_valid_loss:.4f}, Valid Acc: {initial_valid_acc:.4f}")

        # Training loop
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(x)
                loss = self.loss_function(outputs, y)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader.dataset)
            train_acc = self.calculate_accuracy(train_loader)

            # Validation phase
            self.model.eval()
            avg_valid_loss, valid_acc = self.external_test(valid_loader)

            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)
            valid_losses.append(avg_valid_loss)
            valid_accuracies.append(valid_acc)
            epochs_list.append(self.current_epoch + 1)

            # Log after each epoch
            if log:
                wandb.log({
                    "epoch": self.current_epoch + 1,
                    "train loss": avg_train_loss,
                    "train accuracy": train_acc,
                    "valid loss": avg_valid_loss,
                    "valid accuracy": valid_acc
                })

            print(f"Epoch: {self.current_epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

            self.current_epoch += 1

            # Save checkpoint if required
            if save and new_ckpt:
                self.save_model(new_ckpt)

        # Save training dynamics
        if save and train_ckpts:
            self.save_training_dynamics(train_losses, train_accuracies, valid_losses, valid_accuracies, trial, train_ckpts)

    def save_model(self, path):
        """Saves the model and optimizer state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f'Model checkpoint saved at epoch {self.current_epoch} to {path}')

    def load_state(self, path, lr):
        """Loads the model and optimizer state from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Re-initialize optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.opt = True  # Indicate that optimizer has been loaded
        self.model.to(self.device)
        print(f'Checkpoint loaded from {path}, resuming from epoch {self.current_epoch}')

    def save_initial_results(self, train_loss, train_acc, valid_loss, valid_acc, trial, path):
        """Saves the initial results to a JSON file."""
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

    def save_training_dynamics(self, train_losses, train_accuracies, valid_losses, valid_accuracies, trial, path):
        """Saves the training dynamics to a JSON file."""
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
                "Valid Losses": [],
                "Valid Accuracies": []
            }]

        # Append new results to each list within the first dictionary entry
        data[0]["Trial"].append(trial)
        data[0]["Train Losses"].append(train_losses)
        data[0]["Train Accuracies"].append(train_accuracies)
        data[0]["Valid Losses"].append(valid_losses)
        data[0]["Valid Accuracies"].append(valid_accuracies)

        # Write the updated dictionary back to the file
        with open(path, "w") as file:
            json.dump(data, file, indent=4)

    def evaluate(self, test_loader):
        """Evaluates the model on the test set."""
        self.model.eval()
        test_loss, test_acc = self.external_test(test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        return test_loss, test_acc