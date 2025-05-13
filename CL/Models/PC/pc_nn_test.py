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
    def __init__(self, depth, in_dim, hid_dim, out_dim, loss_function, device, batch_size, params=None):
        super(pc_net, self).__init__()
        self.device = device
        self.loss_function = loss_function

        input_size = 10
        hidden_size = 256
        hidden2_size = 256
        output_size = 28 * 28
        activation_fn = nn.ReLU

        self.batch_size = batch_size

        self.pc_model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation_fn(),
            pc.PCLayer(),
            nn.Linear(hidden_size, hidden2_size),
            activation_fn(),
            pc.PCLayer(),
            nn.Linear(hidden2_size, output_size)
        ).to(self.device)
        self.pc_model.train()

        # PCTrainer config
        self.T = 20
        self.optimizer_x_fn = optim.Adam
        self.optimizer_x_kwargs = {'lr': 0.1}
        self.optimizer_p_fn = optim.Adam
        self.optimizer_p_kwargs = {"lr": 0.001, "weight_decay": 0.001}

        self.trainer = pc.PCTrainer(
            self.pc_model,
            T=self.T,
            update_x_at="all",
            optimizer_x_fn=self.optimizer_x_fn,
            optimizer_x_kwargs=self.optimizer_x_kwargs,
            optimizer_p_fn=self.optimizer_p_fn,
            optimizer_p_kwargs=self.optimizer_p_kwargs,
            update_p_at="last",
            plot_progress_at=[]
        )

        self.optimizer_x = self.optimizer_x_fn(self.parameters(), **self.optimizer_x_kwargs)
        self.optimizer_p = self.optimizer_p_fn(self.parameters(), **self.optimizer_p_kwargs)


    def loss_fn(self, output, target):
        target = target.view(target.size(0), -1)
        return 0.5 * (output - target).pow(2).sum()

    def test_normal(self, model, dataset, batch_size, epoch):
        test_loader = dataset

        test_model = nn.Sequential(
            BiasLayer(10, offset=0.),
            pc.PCLayer(),
            model
        ).to(self.device)
        test_model.train()

        trainer_normal_test = pc.PCTrainer(
            test_model,
            T=100,
            update_x_at="all",
            optimizer_x_fn=self.optimizer_x_fn,
            optimizer_x_kwargs=self.optimizer_x_kwargs,
            update_p_at="never",
            optimizer_p_fn=self.optimizer_p_fn,
            optimizer_p_kwargs=self.optimizer_p_kwargs,
            plot_progress_at=[]
        )

        correct_count, all_count = 0., 0.
        # for data, labels in test_loader:
        for data, labels in tqdm(test_loader, desc="Epoch: {}".format(epoch + 1)):

            pseudo_input = torch.zeros(data.shape[0], 10).to(self.device)
            data, labels = data.to(self.device), labels.to(self.device)

            # data = data.view(data.size(0), -1)
            # print(data.shape)
            trainer_normal_test.train_on_batch(
                inputs=pseudo_input,
                loss_fn=self.loss_fn,
                loss_fn_kwargs={"target": data},
                is_log_progress=False,
                is_return_results_every_t=False,
                is_checking_after_callback_after_t=False
            )
            pred = torch.max(test_model[1].get_x(), dim=1)
            correct = (pred.indices == labels).long()
            correct_count += correct.sum()
            all_count += correct.size(0)

        return round((correct_count / all_count).item(), 4)

    # def train_model(self, train_dataset, val_dataset, epochs, train_loader, valid_loader, batch_size, log, save):
    def train_model(self, train_dataset, val_dataset, epochs, train_loader, valid_loader, batch_size, log, save, trial, new_ckpt= '',train_ckpts = ''):

        # best_val_acc = 0
        # best_model = copy.deepcopy(self.pc_model.state_dict())
        # best_model_idx = None


        # train_losses = []
        # train_accuracies = []
        # test_losses = []
        # test_accuracies = []
        eps = []

        # train_acc = [self.test_normal(self.pc_model, train_dataset, batch_size)]
        # val_acc = [self.test_normal(self.pc_model, val_dataset, batch_size)]
        train_acc = [] 
        val_acc = []


        # uncomment
        # train_acc = [self.test_normal(self.pc_model, train_loader, batch_size, -1)]
        # print("finished Train for Epoch 0")
        # val_acc = [self.test_normal(self.pc_model, valid_loader, batch_size, -1)]
        # print("finished Validation for Epoch 0")

        # if log:
        #   wandb.log({
        #       "epoch": 0,
        #       "train_acc": train_acc[0],
        #       "val_acc": val_acc[0]
        #   })
        
        for epoch in range(epochs):
            for data, label in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                data, label = data.to(self.device), label.to(self.device)
                labels_one_hot = F.one_hot(label, num_classes=10).float()
                self.trainer.train_on_batch(
                    inputs=labels_one_hot,
                    loss_fn=lambda output: self.loss_fn(output, data),
                    is_log_progress=False,
                    is_return_results_every_t=False,
                    is_checking_after_callback_after_t=False
                )
            # val_acc.append(self.test_normal(self.pc_model, val_dataset, batch_size))
            # if val_acc[-1] > best_val_acc:
            #     best_val_acc = val_acc[-1]
            #     best_model = copy.deepcopy(self.pc_model.state_dict())
            #     best_model_idx = epoch

            train_acc_epoch = self.test_normal(self.pc_model, train_loader, batch_size, epoch)
            val_acc_epoch = self.test_normal(self.pc_model, valid_loader, batch_size, epoch)
            train_acc.append(train_acc_epoch)
            val_acc.append(val_acc_epoch)


            eps.append(epoch)

            
            # Log metrics to wandb
            if log:
              wandb.log({
                  "epoch": epoch + 1,
                  "train_acc": train_acc_epoch,
                  "val_acc": val_acc_epoch
              })
              print(f'Epoch {epoch+1} - Val acc: {val_acc[-1]}')

        if save == 'yes':
            self.save_model(new_ckpt)
            # self.save_training_dynamics(train_losses, train_accuracies, test_losses, test_accuracies, trial, train_ckpts)
            self.save_training_dynamics(train_acc, val_acc, trial, train_ckpts)

        # izining_dynamics(train_losses, train_accuracies, test_losses, test_accuracies, trial, train_ckpts)
    def save_model(self, new_ckpt):
      path = new_ckpt
      os.makedirs(os.path.dirname(path), exist_ok=True)
      torch.save({
          'model_state_dict': self.state_dict(),
          'optimizer_x_state_dict': self.optimizer_x.state_dict(),
          'optimizer_p_state_dict': self.optimizer_p.state_dict(),
      }, path)


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

    # def save_training_dynamics(self, train_losses, train_accuracies, test_losses, test_accuracies, trial, ckpt):
    def save_training_dynamics(self, train_accuracies, test_accuracies, trial, ckpt):
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
                # "Train Losses": [],
                "Train Accuracies": [],
                # "Test Losses": [],
                "Test Accuracies": []
            }]

        # Append new results to each list within the first dictionary entry
        data[0]["Trial"].append(trial)
        # data[0]["Train Losses"].append(train_losses)
        data[0]["Train Accuracies"].append(train_accuracies)
        # data[0]["Test Losses"].append(test_losses)
        data[0]["Test Accuracies"].append(test_accuracies)

        # Write the updated dictionary back to the file
        with open(path, "w") as file:
            json.dump(data, file, indent=4)

    def load_state(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer_x.load_state_dict(checkpoint['optimizer_x_state_dict'])
        self.optimizer_p.load_state_dict(checkpoint['optimizer_p_state_dict'])