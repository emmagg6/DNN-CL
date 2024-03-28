from Models.bp_layer import bp_layer
from Models.net import net

import time
import wandb
import numpy as np
import torch
from torch import nn
import os


class bp_net(net):
    def __init__(self, depth, in_dim, hid_dim, out_dim, activation_function, loss_function, algorithm, device, continual=False):
        self.device = device
        self.depth = depth
        self.layers = self.init_layers(in_dim, hid_dim, out_dim, activation_function)
        self.loss_function = loss_function
        self.MSELoss = nn.MSELoss(reduction="sum")
        self.algorithm = algorithm
        self.cont = continual

    def init_layers(self, in_dim, hid_dim, out_dim, activation_function):
        layers = [None] * self.depth
        dims = [in_dim] + [hid_dim] * (self.depth - 1) + [out_dim]
        
        # correct activation function for layers with batch normalization
        for d in range(self.depth - 1):
            if "-BN" in activation_function:
                # if includes batch normalization, use the modified string
                act_func = activation_function.replace("-BN", "") + "-BN"
            else:
                act_func = activation_function
            layers[d] = bp_layer(dims[d], dims[d + 1], act_func, self.device, continual = self.cont)
        
        # The final layer should not use batch normalization, assuming it's a linear layer without BN
        layers[-1] = bp_layer(dims[-2], dims[-1], "linear", self.device, continual = self.cont)
        return layers

    def set_bn_eval(self):
        for layer in self.layers:
            if layer.use_batch_norm:
                layer.batch_norm.eval()

    def set_bn_train(self):
        for layer in self.layers:
            if layer.use_batch_norm:
                layer.batch_norm.train()

    def train(self, train_loader, valid_loader, epochs, lr, log, save):
        if self.continual:
            # Set batch norm layers to evaluation mode
            self.set_bn_eval()

            # Load a single batch to check the model output
            x_sample, y_sample = next(iter(train_loader))
            x_sample, y_sample = x_sample.to(self.device), y_sample.to(self.device)
            y_pred_sample = self.forward(x_sample)

            if torch.isnan(y_pred_sample).any() or torch.isinf(y_pred_sample).any():
                raise ValueError("NaN or Inf in model output during evaluation before training starts.")

            # Set batch norm layers back to training mode
            self.set_bn_train()

        # Initial evaluation before any training
        train_loss, train_acc = self.test(train_loader)
        valid_loss, valid_acc = self.test(valid_loader)
        if log:
            log_dict = {
                "epoch": 0,  # Log as epoch 0
                "train loss": train_loss,
                "valid loss": valid_loss,
                "train accuracy": train_acc,
                "valid accuracy": valid_acc
            }
            wandb.log(log_dict)
        else:
            print(f"Epoch 0 - Train Loss: {train_loss}, Valid Loss: {valid_loss}")
            if train_acc is not None:
                print(f"Train Acc: {train_acc}")
            if valid_acc is not None:
                print(f"Valid Acc: {valid_acc}")

        # print("Epoch 0")
        # for e in range(1, epochs + 1):
        #     print(f"Epoch {e}")
        #     start_time = time.time()

        #     # Training forward pass and weight update
        #     for x, y in train_loader:
        #         x, y = x.to(self.device), y.to(self.device)
        #         self.update_weights(x, y, lr)

            # end_time = time.time()
                
        print("Epoch 0")
        for e in range(1, epochs + 1):
            print(f"Epoch {e}")
            start_time = time.time()

            # Training forward pass and weight update
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.update_weights(x, y, lr)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            end_time = time.time()


            # Prediction and logging after each epoch
            with torch.no_grad():
                train_loss, train_acc = self.test(train_loader)
                valid_loss, valid_acc = self.test(valid_loader)
                if log:
                    log_dict = {
                        "epoch": e,
                        "train loss": train_loss,
                        "valid loss": valid_loss,
                        "train accuracy": train_acc,
                        "valid accuracy": valid_acc,
                        "time": end_time - start_time
                    }
                    wandb.log(log_dict)
                else:
                    print(f"Epoch {e} - Train Loss: {train_loss}, Valid Loss: {valid_loss}")
                    if train_acc is not None:
                        print(f"Train Acc: {train_acc}")
                    if valid_acc is not None:
                        print(f"Valid Acc: {valid_acc}")

        # Save the model after training, if specified
        if save == 'yes':
            self.save_model()

    def update_weights(self, x, y, lr):
        y_pred = self.forward(x).requires_grad_()
        y_pred.retain_grad()
        loss = self.loss_function(y_pred, y)
        print(f"Update weights - Loss: {loss.item()}")
        batch_size = len(y)
        self.zero_grad()
        loss.backward()

        for idx, layer in enumerate(self.layers):
            if layer.weight.grad is not None:
                print(f"Layer {idx} - Weight gradient norm: {layer.weight.grad.norm().item()}")  # Print the norm of the gradients
            else:
                print(f"Layer {idx} - No gradients computed (grad is None)")  # This indicates a problem

        g = y_pred.grad
        grad = [None] * self.depth
        with torch.no_grad():
            for d in reversed(range(self.depth)):
                g = g * self.layers[d].activation_derivative(self.layers[d].linear_activation)
                grad[d] = g.T @ self.layers[d - 1].linear_activation if d >= 1 else g.T @ x
                if self.algorithm == "BP":
                    g = g @ self.layers[d].weight
                elif self.algorithm == "FA":
                    g = g @ self.layers[d].fixed_weight
        for d in range(self.depth):
            update_value = (lr / batch_size) * grad[d]
            print(f"Layer {d} - Update norm: {update_value.norm().item()}")
            self.layers[d].weight = (self.layers[d].weight - (lr / batch_size)
                                     * grad[d]).detach().requires_grad_()


    def zero_grad(self):
        for d in range(self.depth):
            if self.layers[d].weight.grad is not None:
                self.layers[d].weight.grad.zero_()

    def forward(self, x, update=True):
        y = x
        for d in range(self.depth):
            y = self.layers[d].forward(y, update=update)
        return y

    def save_model(self, path="checkpoints/bp/params.pth"):
        self.train()  # Ensure the model is in train mode if required
        os.makedirs(os.path.dirname(path), exist_ok=True)
        layer_params = {f'layer_{idx}': layer.get_params() for idx, layer in enumerate(self.layers)}
        torch.save(layer_params, path)


    def load_model(self, path):
        state_dict = torch.load(path)
        for layer_key, layer_params in state_dict.items():
            layer_idx = int(layer_key.split('_')[-1])
            self.layers[layer_idx].load_params(layer_params)

            # printing #
            # print(f"Layer {layer_idx} - Loaded Weight Sample: {self.layers[layer_idx].weight.data[:5]}")
            # if self.layers[layer_idx].use_batch_norm:
            #     print(f"Layer {layer_idx} - Loaded Batch Norm Mean: {self.layers[layer_idx].batch_norm.running_mean}")
            #     print(f"Layer {layer_idx} - Loaded Batch Norm Var: {self.layers[layer_idx].batch_norm.running_var}")

