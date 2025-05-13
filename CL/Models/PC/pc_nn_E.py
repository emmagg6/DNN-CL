import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import json
import random
import Models.PC.predictive_coding as pc
from Models.PC.predictive_coding import PCLayer, PCTrainer
import copy
from tqdm import tqdm
import wandb


# This class contains the parameters of the prior mean \mu parameter (see figure)
class BiasLayer(nn.Module):
    def __init__(self, out_features, offset=0.):
        super(BiasLayer, self).__init__()
        self.bias = nn.Parameter(offset*torch.ones(out_features) if offset is not None else 2*np.sqrt(out_features)*torch.rand(out_features)-np.sqrt(out_features), requires_grad=True)

    def forward(self, x):
        return torch.zeros_like(x) + self.bias  # return the prior mean \mu witht the same shape as the input x to make sure the batch size is the same


class pc_net(nn.Module):
    def __init__(self, depth=2, in_dim=10, hid_dim=256, out_dim=28*28, loss_function=nn.MSELoss(), device='cpu', batch_size=64, params=None):
        super(pc_net, self).__init__()
        self.depth = depth
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.device = device
        self.loss_function = loss_function
        self.batch_size = batch_size


        # Dynamically build the model based on depth
        layers = []
        input_size = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(input_size, hid_dim))
            layers.append(nn.BatchNorm1d(hid_dim)) # batch normalization can help with exploding gradients
            layers.append(nn.ReLU())
            layers.append(pc.PCLayer())
            input_size = hid_dim  # for next layer
        layers.append(nn.Linear(input_size, out_dim))  # Output layer
        # layers.append(nn.Softmax(dim=1))
        self.pc_model = nn.Sequential(*layers).to(self.device)
        self.pc_model.train()

        # PCTrainer configuration
        self.T = 25 # giving fewer inference steps can help mititgate exploding gradients
        self.optimizer_x_fn = optim.Adam
        self.optimizer_x_kwargs = {'lr': 0.020}
        self.optimizer_p_fn = optim.Adam
        self.optimizer_p_kwargs = {"lr": 0.001}

        self.trainer = pc.PCTrainer(
            self.pc_model,
            T=self.T,
            update_x_at="all",
            optimizer_x_fn=self.optimizer_x_fn,
            optimizer_x_kwargs=self.optimizer_x_kwargs,
            optimizer_p_fn=self.optimizer_p_fn,
            optimizer_p_kwargs=self.optimizer_p_kwargs,
            update_p_at="last", #"last:",  #note: spreading out can prevent accumulation of large gradients over the T inference steps
            plot_progress_at=[]
        )

        self.optimizer_x = self.optimizer_x_fn(self.parameters(), **self.optimizer_x_kwargs)
        self.optimizer_p = self.optimizer_p_fn(self.parameters(), **self.optimizer_p_kwargs)

    # def loss_fn(self, output, target):
    #     target = target.view(target.size(0), -1)
    #     return 0.5 * (output - target).pow(2).sum()

    def loss_fn(self, output, target):
        return self.loss_function(output, target)

    def test_normal(self, model, dataset, batch_size, epoch):
        # Ensure the model is in evaluation mode
        model.eval()
        
        test_loader = dataset  # check that 'dataset' is a DataLoader instance

        correct_count, all_count = 0., 0.
        
        with torch.no_grad():  # Disable gradient computation for efficiency
            for data, labels in tqdm(test_loader, desc=f"Epoch: {epoch + 1}"):
                data, labels = data.to(self.device), labels.to(self.device)
                data_flat = data.view(data.size(0), -1)  # flatten the images
                
                # forward pass using the same model structure as training
                output = model(data)
                
                pred = torch.max(output, dim=1)
                correct = (pred.indices == labels).sum().item()
                correct_count += correct
                all_count += labels.size(0)
        
        accuracy = round((correct_count / all_count), 4)
        
        return accuracy


    def train_model(self, train_dataset, val_dataset, epochs, train_loader, valid_loader, batch_size, log, save, trial, new_ckpt='', train_ckpts=''):
        train_acc = []
        val_acc = []
        eps = []

        ### testing untrained model ###
        epoch = 0
        train_acc_epoch = self.test_normal(self.pc_model, train_loader, batch_size, epoch)
        val_acc_epoch = self.test_normal(self.pc_model, valid_loader, batch_size, epoch)
        train_acc.append(train_acc_epoch)
        val_acc.append(val_acc_epoch)
        eps.append(epoch)

        # Logging
        if log:
            wandb.log({
                "epoch": epoch,
                "train_acc": train_acc_epoch,
                "val_acc": val_acc_epoch
                # Add more metrics if tracked
            })
        print(f'Epoch {epoch}: Train Acc: {train_acc_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}, flush=True\n\n')
        ###############################

        grad_norms = []
        for epoch in range(epochs):
            self.pc_model.train()
            epoch_losses = []

            for data, label in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                data, label = data.to(self.device), label.to(self.device)
                data_flat = data.view(data.size(0), -1) # flatten the images and normalize
                labels_one_hot = F.one_hot(label, num_classes=self.out_dim).float()

                # zero the gradients
                self.optimizer_p.zero_grad()
                self.optimizer_x.zero_grad()

                self.trainer.train_on_batch(
                    inputs=data_flat,
                    loss_fn=lambda output: self.loss_fn(output, labels_one_hot),
                    is_log_progress=False,
                    is_return_results_every_t=False,
                    is_checking_after_callback_after_t=False
                )

                # gradient norms
                total_norm = 0
                for p in self.pc_model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                grad_norms.append(total_norm)

            # average gradient norm for the epoch
            avg_grad_norm = sum(grad_norms) / len(grad_norms)
            print(f'Epoch {epoch+1}, Average Gradient Norm: {avg_grad_norm:.6f}')

            train_acc_epoch = self.test_normal(self.pc_model, train_loader, batch_size, epoch)
            val_acc_epoch = self.test_normal(self.pc_model, valid_loader, batch_size, epoch)
            train_acc.append(train_acc_epoch)
            val_acc.append(val_acc_epoch)
            eps.append(epoch)

            if log:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_acc": train_acc_epoch,
                    "val_acc": val_acc_epoch
                    # Add more metrics if tracked
                })
            print(f'Epoch {epoch+1}: Train Acc: {train_acc_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}, flush=True \n\n')

        # Saving the model and training dynamics
        # if save == 'yes':
        #     self.save_model(new_ckpt)
        #     self.save_training_dynamics(train_acc, val_acc, trial, train_ckpts)


    # def save_initial_results(self, train_loss, train_acc, valid_loss, valid_acc, trial, ckpt):
    def save_initial_results(self, train_acc, valid_acc, trial, ckpt):
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
                # "Train Loss": [],
                "Train Acc": [],
                # "Valid Loss": [],
                "Valid Acc": []
            }]

        # Append new results to each list within the first dictionary entry
        data[0]["Trial"].append(trial)
        # data[0]["Train Loss"].append(train_loss)
        data[0]["Train Acc"].append(train_acc)
        # data[0]["Valid Loss"].append(valid_loss)
        data[0]["Valid Acc"].append(valid_acc)

        # Write the updated dictionary back to the file
        with open(path, "w") as file:
            json.dump(data, file, indent=4)
        # self.pc_model.load_state_dict(best_model, strict=False)
        # acc_test = self.test_normal(self.pc_model, test_dataset)
        # print(f'Test accuracy: {acc_test}')

    def save_model(self, new_ckpt):
        path = os.path.join(new_ckpt)
        print(f"Saving model to path: {path}")
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        else:
            print("Warning: Path does not contain a directory. Saving in the current directory.")
        
        torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_x_state_dict': self.optimizer_x.state_dict(),
                'optimizer_p_state_dict': self.optimizer_p.state_dict(),
            }, path)

        # model configuration parameters
        model_config = {
            'depth': self.depth,
            'in_dim': self.in_dim,
            'hid_dim': self.hid_dim,
            'out_dim': self.out_dim,
            'loss_function_name': self.loss_function.__class__.__name__,
            'T': self.T,
            'device': str(self.device),  # convert device to string
            'batch_size': self.batch_size,
            # essential optimizer hyperparameters -- if necessary
            'optimizer_x_lr': self.optimizer_x.param_groups[0]['lr'],
            'optimizer_p_lr': self.optimizer_p.param_groups[0]['lr'],
        }

        # tensors (weights and optimizer states)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_x_state_dict': self.optimizer_x.state_dict(),
            'optimizer_p_state_dict': self.optimizer_p.state_dict(),
        }, path)

        # model configurations -- json file
        config_path = path + '_config.json'
        with open(config_path, 'w') as f:
            json.dump(model_config, f)


    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path, weights_only=True)

        config_path = path + '_config.json'
        with open(config_path, 'r') as f:
            model_config = json.load(f)

        # reconstruct components
        loss_function_class = getattr(nn, model_config['loss_function_name'])
        loss_function = loss_function_class()
        device = torch.device(model_config['device'])

        # model instance
        model = cls(
            depth=model_config['depth'],
            in_dim=model_config['in_dim'],
            hid_dim=model_config['hid_dim'],
            out_dim=model_config['out_dim'],
            loss_function=loss_function,
            device=device,
            batch_size=model_config['batch_size'],
        )

        # model state
        # model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # reconstruct optimizers with saved learning rates
        optimizer_x_lr = model_config.get('optimizer_x_lr', 0.001)
        optimizer_p_lr = model_config.get('optimizer_p_lr', 0.001)
        model.optimizer_x = optim.Adam(model.parameters(), lr=optimizer_x_lr)
        model.optimizer_p = optim.Adam(model.parameters(), lr=optimizer_p_lr)

        # optimizer states
        model.optimizer_x.load_state_dict(checkpoint['optimizer_x_state_dict'])
        model.optimizer_p.load_state_dict(checkpoint['optimizer_p_state_dict'])

        return model


