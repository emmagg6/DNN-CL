# https://github.com/smonsays/equilibrium-propagation 

# MIT License

# Copyright (c) 2020 Simon Schug, João Sacramento

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import os
import json
import wandb

from Models.EP.ep_fcns import CEnergy, CrossEntropy, SquaredError, create_cost, create_activations, create_optimizer
from Models.EP.ep_layers import RestrictedHopfield, ConditionalGaussian


class ep_net:
    def __init__(self, type='restr_hopfield', dimensions=[28*28, 640, 10], cost_energy='cross_entropy', batch_size=64, beta = 1, device='cpu'):
        self.type = type
        self.cost_energy = create_cost(cost_energy, beta)
        self.phi = create_activations("sigmoid", len(dimensions))
        if self.type == 'restr_hopfield':
            self.model = RestrictedHopfield(dimensions, self.cost_energy, batch_size, self.phi)
        elif self.type == 'cond_gaussian':
            self.model = ConditionalGaussian(dimensions, self.cost_energy, batch_size, self.phi)
        else:
            raise ValueError('Unknown model type.')
        
        self.device = device
        self.opt = False
        

    def predict_batch(self, x_batch, dynamics, fast_init):
        self.model.reset_state()
        self.model.clamp_layer(0, x_batch.view(-1, self.model.dimensions[0]))
        self.model.set_C_target(None)
        if fast_init:
            self.model.fast_init()
        else:
            self.model.u_relax(**dynamics)
        return torch.nn.functional.softmax(self.model.u[-1].detach(), dim=1)

    def test_model(self, test_loader, dynamics, fast_init):
        test_E, correct, total = 0.0, 0.0, 0.0
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            output = self.predict_batch(x_batch, dynamics, fast_init)
            prediction = torch.argmax(output, 1)
            with torch.no_grad():
                correct += float(torch.sum(prediction == y_batch.argmax(dim=1)))
                test_E += float(torch.sum(self.model.E))
                total += x_batch.size(0)
        return correct / total, test_E / total

    def train_model(self, train_loader, valid_loader, epochs, dynamics, lr = 0.01, fast_init = True, log=False, save=False, trial=0, new_ckpt='', train_ckpts=''):
        if self.opt == False:
            self.optimizer = create_optimizer(self.model, "adam",  lr=lr) # options: sgd, adam, adagrad
        epoch = 0
        test_accs = []
        test_acc, test_E = self.test_model(valid_loader, dynamics, fast_init)

        if log:
            wandb.log({"epoch": epoch, "valid accuracy": test_acc})
        print(f"Epoch: {epoch}, Test Acc: {test_acc}, Test E: {test_E}")
        test_accs.append(test_acc)
        
        for epoch in range(1, epochs+1):
            self.model.train()
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                # reinitialize the neural state variables
                self.model.reset_state()
                # input just the training sample
                self.model.clamp_layer(0, x_batch.view(-1, self.model.dimensions[0]))

                # Free phase
                if fast_init:
                    self.model.fast_init() # fast feed-forward initialization (skip free phase)
                    free_grads = [torch.zeros_like(p) for p in self.model.parameters()]
                else:
                    # run free phase and get gradients
                    self.model.set_C_target(None)
                    dE = self.model.u_relax(**dynamics)
                    free_grads = self.model.w_get_gradients()

                # Nudged phase
                self.model.set_C_target(y_batch)
                dE = self.model.u_relax(**dynamics)
                nudged_grads = self.model.w_get_gradients()

                # Update weights
                self.model.w_optimize(free_grads, nudged_grads, self.optimizer)

            test_acc, test_E = self.test_model(valid_loader, dynamics, fast_init)
            if log:
                wandb.log({"epoch": epoch, "valid accuracy": test_acc})
            print(f"Epoch: {epoch}, Test Acc: {test_acc}, Test E: {test_E}")
            test_accs.append(test_acc)

        if save:
            self.save_model(new_ckpt)
            # self.save_training_dynamics(train_loader, valid_loader, trial, train_ckpts)

    def save_model(self, path="checkpoints/EP/params.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_state(self, path, lr):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = create_optimizer(self.model, "adam", lr=lr)
        self.opt = True
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_training_dynamics(self, test_accuracies, trial, ckpt):
        path = ckpt
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r") as file:
                data = json.load(file)
        else:
            data = [{
                "Trial": [],
                # "Train Losses": [],
                # "Train Accuracies": [],
                # "Test Losses": [],
                "Test Accuracies": []
            }]

        data[0]["Trial"].append(trial)
        # data[0]["Train Losses"].append(train_losses)
        # data[0]["Train Accuracies"].append(train_accuracies)
        # data[0]["Test Losses"].append(test_losses)
        data[0]["Test Accuracies"].append(test_accuracies)

        with open(path, "w") as file:
            json.dump(data, file, indent=4)

