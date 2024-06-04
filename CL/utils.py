
####################################################################################################
#                                           BP & TP                                                #
####################################################################################################

'''
Following utils on code used for the experiments conducted in the submitted paper 
"Fixed-Weight Difference Target Propagation" by K. K. S. Wu, K. C. K. Lee, and T. Poggio.

'''

import os
import numpy as np
import torch
from torch import nn
import math
import wandb


def combined_loss(pred, label, device="cpu", num_classes=10):
    batch_size = pred.shape[0]
    dim = pred.shape[1]
    E = torch.eye(dim, device=device)
    E1 = E[:, :num_classes]
    E2 = E[:, num_classes:]
    ce = nn.CrossEntropyLoss(reduction="sum")
    mse = nn.MSELoss(reduction="sum")
    return ce(pred @ E1, (label @ E1).max(axis=1).indices) + 1e-3 * mse(pred @ E2, label @ E2)


def calc_accuracy(pred, label):
    max_index = pred.max(axis=1).indices
    return (max_index == label).sum().item() / label.shape[0]


def calc_accuracy_combined(pred, label, num_classes=10):
    data_size = pred.shape[0]
    pred_max = pred[:, :num_classes].max(axis=1).indices
    label_max = label[:, :num_classes].max(axis=1).indices
    return (pred_max == label_max).sum().item() / data_size


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def calc_angle(v1, v2):
    cos = (v1 * v2).sum(axis=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-12)
    cos = torch.clamp(cos, min=-1, max=1)
    acos = torch.acos(cos) * 180 / math.pi
    angle = 180 - torch.abs(acos - 180)
    return angle


def batch_normalization(x, mean=None, std=None):
    if mean is None:
        mean = torch.mean(x, dim=0)
    if std is None:
        std = torch.std(x, dim=0)
    return (x - mean) / (std + 1e-12)


def batch_normalization_inverse(y, mean, std):
    return y * std + mean


def set_wandb(args, params):
    config = args.copy()
    name = {"ff1": "forward_function_1",
            "ff2": "forward_function_2",
            "bf1": "backward_function_1",
            "bf2": "backward_function_2"}
    for n in name.keys():
        config[name[n]] = params[n]["type"]
        config[name[n] + "_init"] = params[n]["init"]
        config[name[n] + "_activation"] = params[n]["act"]
    config["last_activation"] = params["last"]
    wandb.init(config=config)


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        os.environ['OMP_NUM_THREADS'] = '1'
    return device


####################################################################################################
#                                               PC                                                #
####################################################################################################
'''
The following utils are used for the experiments conducted in the submitted paper:
@article{millidge2020predictive,
  title={Predictive Coding Approximates Backprop along Arbitrary Computation Graphs},
  author={Millidge, Beren and Tschantz, Alexander and Buckley, Christopher L},
  journal={arXiv preprint arXiv:2006.04182},
  year={2020}
}
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributions as dist
from copy import deepcopy
import math
import matplotlib.pyplot as plt

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### General Utils ###
def boolcheck(x):
    return str(x).lower() in ["true", "1", "yes"]

def set_tensor(xs):
    return xs.float().to(DEVICE)

def edge_zero_pad(img,d):
  N,C, h,w = img.shape
  x = torch.zeros((N,C,h+(d*2),w+(d*2))).to(DEVICE)
  x[:,:,d:h+d,d:w+d] = img
  return x

def accuracy(out, L):
  B,l = out.shape
  total = 0
  for i in range(B):
    if torch.argmax(out[i,:]) == torch.argmax(L[i,:]):
      total +=1
  return total/ B

def sequence_accuracy(model, target_batch):
    accuracy = 0
    L = len(target_batch)
    _,B = target_batch[0].shape
    s = ""
    for i in range(len(target_batch)): # this loop is over the seq_len
      s += str(torch.argmax(model.mu_y[i][:,0]).item()) + " " + str(torch.argmax(target_batch[i][:,0]).item()) + "  "
      for b in range(B):
        #print("target idx: ", torch.argmax(target_batch[i][:,b]).item())
        #print("pred idx: ", torch.argmax(model.mu_y[i][:,b]).item())
        if torch.argmax(target_batch[i][:,b]) ==torch.argmax(model.mu_y[i][:,b]):
          accuracy+=1
    print("accs: ", s)
    return accuracy / (L * B)

def custom_onehot(idx, shape):
  ret = set_tensor(torch.zeros(shape))
  ret[idx] =1
  return ret

# def onehot(arr, vocab_size):
#   print(arr)
#   print(arr.type())
#   L, B = arr.shape
#   ret = np.zeros([L,vocab_size,B])
#   for l in range(L):
#     for b in range(B):
#       ret[l,int(arr[l,b]),b] = 1
#   return ret

#####################################################################################################
#                       (emmagg6) fix compatibility with custom onehot enc                          #
#####################################################################################################
def onehot(arr, vocab_size):
    arr = arr.unsqueeze(1)  # Ensure arr is 2D: [N, 1]
    one_hot = torch.zeros((arr.size(0), vocab_size), device=arr.device)  # Create a zero tensor of size [N, vocab_size]
    one_hot.scatter_(1, arr, 1)  # Fill with 1s at the indices from arr
    return one_hot

def inverse_list_onehot(arr):
  L = len(arr)
  V,B = arr[0].shape
  ret = np.zeros([L,B])
  for l in range(L):
    for b in range(B):
      for v in range(V):
        if arr[l][v,b] == 1:
          ret[l,b] = v
  return ret

def decode_ypreds(ypreds):
  L = len(ypreds)
  V,B = ypreds[0].shape
  ret = np.zeros([L,B])
  for l in range(L):
    for b in range(B):
      v = torch.argmax(ypreds[l][:,b])
      ret[l,b] =v
  return ret


def inverse_onehot(arr):
  if type(arr) == list:
    return inverse_list_onehot(arr)
  else:
    L,V,B = arr.shape
    ret = np.zeros([L,B])
    for l in range(L):
      for b in range(B):
        for v in range(V):
          if arr[l,v,b] == 1:
            ret[l,b] = v
    return ret

### Activation functions ###
def tanh(xs):
    return torch.tanh(xs)

def linear(x):
    return x

def tanh_deriv(xs):
    return 1.0 - torch.tanh(xs) ** 2.0

def linear_deriv(x):
    return set_tensor(torch.ones((1,)))

def relu(xs):
  return torch.clamp(xs,min=0)

def relu_deriv(xs):
  rel = relu(xs)
  rel[rel>0] = 1
  return rel

# def softmax(xs):
#   return F.softmax(xs)

def sigmoid(xs):
  return F.sigmoid(xs)

def sigmoid_deriv(xs):
  return F.sigmoid(xs) * (torch.ones_like(xs) - F.sigmoid(xs))


### loss functions
def mse_loss(out, label):
      return torch.sum((out-label)**2)

def mse_deriv(out,label):
      return 2 * (out - label)

# ce_loss = nn.CrossEntropyLoss()

def cross_entropy_loss(out,label):
      return nn.CrossEntropyLoss(out,label)

def my_cross_entropy(out,label):
      return -torch.sum(label * torch.log(out + 1e-6))

def cross_entropy_deriv(out,label):
      return out - label

def parse_loss_function(loss_arg):
      if loss_arg == "mse":
            return mse_loss, mse_deriv
      elif loss_arg == "crossentropy":
            return my_cross_entropy, cross_entropy_deriv
      else:
            raise ValueError("loss argument not expected. Can be one of 'mse' and 'crossentropy'. You inputted " + str(loss_arg))


### Initialization Functions ###
def gaussian_init(W,mean=0.0, std=0.05):
  return W.normal_(mean=0.0,std=0.05)

def zeros_init(W):
  return torch.zeros_like(W)

def kaiming_init(W, a=math.sqrt(5),*kwargs):
  return init.kaiming_uniform_(W, a)

def glorot_init(W):
  return init.xavier_normal_(W)

################# self is not defined this is an error - emmagg6 ######
# def kaiming_bias_init(b,*kwargs):
#   fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#   bound = 1 / math.sqrt(fan_in)
#   return init.uniform_(b, -bound, bound)

#the initialization pytorch uses for lstm
def std_uniform_init(W,hidden_size):
  stdv = 1.0 / math.sqrt(hidden_size)
  return init.uniform_(W, -stdv, stdv)




#####################################################################################################
#                                               KAN                                            #
#####################################################################################################

# from https://github.com/IvanDrokin/torch-conv-kan/blob/main/utils/regularization.py
# Based on this implementations: https://github.com/szymonmaszke/torchlayers/blob/master/torchlayers/regularization.py
import abc

import torch
import torch.nn as nn

class WeightDecay(nn.Module):
    def __init__(self, module, weight_decay, name: str = None):
        if weight_decay < 0.0:
            raise ValueError(
                "Regularization's weight_decay should be greater than 0.0, got {}".format(
                    weight_decay
                )
            )

        super().__init__()
        self.module = module
        self.weight_decay = weight_decay
        self.name = name

        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    def remove(self):
        self.hook.remove()

    def _weight_decay_hook(self, *_):
        if self.name is None:
            for param in self.module.parameters():
                if param.grad is None or torch.all(param.grad == 0.0):
                    param.grad = self.regularize(param)
        else:
            for name, param in self.module.named_parameters():
                if self.name in name and (
                    param.grad is None or torch.all(param.grad == 0.0)
                ):
                    param.grad = self.regularize(param)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def extra_repr(self) -> str:
        representation = "weight_decay={}".format(self.weight_decay)
        if self.name is not None:
            representation += ", name={}".format(self.name)
        return representation

    @abc.abstractmethod
    def regularize(self, parameter):
        pass

class L1(WeightDecay):
    """Regularize module's parameters using L1 weight decay.

    Example::

        import torchlayers as tl

        # Regularize all parameters of Linear module
        regularized_layer = tl.L1(tl.Linear(30), weight_decay=1e-5)

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L1` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").

    """

    def regularize(self, parameter):
        return self.weight_decay * torch.sign(parameter.data)