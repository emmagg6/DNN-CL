from collections import OrderedDict
import torch
from torch import nn
import utils
import numpy as np

# torch.manual_seed(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 100
BATCH_SIZE = 100
N_BATCHES = 500  # 50000/100, total number of samples / batch size

LAYERS_SIZES = [784, 240, 240, 240, 240, 240, 240, 240, 10]


class FFNet(torch.nn.Module):
    def __init__(self, layer_sizes):
        self.f_weights = OrderedDict()
        self.g_weights = OrderedDict()
        super(FFNet, self).__init__()

        for layer_num in range(len(layer_sizes) - 1):
            weight_name = f"Wb_{layer_num+1}"
            irange = np.sqrt(6. / (layer_sizes[layer_num] + layer_sizes[layer_num+1]))
            self.f_weights[weight_name] = [nn.Parameter(
                utils.rand_ortho((layer_sizes[layer_num], layer_sizes[layer_num+1]), irange)),
                nn.Parameter(torch.zeros([layer_sizes[layer_num+1]]))]

        for layer_num in range(len(layer_sizes)-1, 0 - 1):
            weight_name = f"Vc_{layer_num}"
            self.g_weights[weight_name] = [nn.Parameter(
                utils.rand_ortho([layer_sizes[layer_num-1], layer_sizes[layer_num]],
                                 np.sqrt(6. / (layer_sizes[layer_num-1] + layer_sizes[layer_num])))),
                nn.Parameter(torch.zeros([layer_sizes[layer_num]]))]

    def forward(self, X):
        # X: batch_size * 28 * 28
        h_i = X.view(X.size(0), -1)  # batch_size * 784
        for layer, weights in self.f_weights.items():
            # print(layer)
            if layer != len(self.f_weights) - 1:
                h_i = torch.tanh(h_i @ weights[0] + weights[1])
            else:
                h_i = h_i @ weights[0] + weights[1]  # last layer logits output
        return h_i

def train_tp_inverse_epoch(model, train_loader, hparams, logger=None):
    # forward 
    for X, Y in train_loader:

        # forard activations
        h_i = X.view(X.size(0), -1)
        Hs = []
        for layer in range(1, len(model.f_weights)+1): # 1,2,3,4,5,6,7,8 
            if layer != len(model.f_weights): # not output layer
                with torch.no_grad():
                    layer_params = f"Wb_{layer}"
                    h_i =  torch.tanh(h_i @ model.f_weights[layer_params][0] + model.f_weights[layer_params][1])
                    Hs.append(h_i)


def train_tp_forward_epoch(model, train_loader, hparams, logger=None):
    pass
