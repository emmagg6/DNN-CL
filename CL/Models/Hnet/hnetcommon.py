"""
non-von HNetComponentBank, EdgePrediction, HNetViaCompH, HNetViaPrimH, LogicFC, Binarize data structures
CONFIDENTIAL
Copyright (c) 2022-2025, Non-Von LLC, all rights reserved.

Data Structures
===============
EdgePrediction: 
HNetComponentBank: 
HNetLayer: 
HNetConv2d: 
HNetViaCompH: 
HNetViaPrimH: 
LogicFC:
Binarize:

Callable Functions
==================
"""
import torch
import torch.nn as nn
import random

from .hnetcustomops import HNetOpPopCount, HNetOpCountBits

SIGMOID_CLIPPING_CONST = 0.25

# === DATA STRUCTURES ===

class HNetComponentBank(nn.Module):
    """HNet object for choosing and modulating the energy method"""
    def __init__(self, n_nodes:int, n_cmp:int, n_random_edges=None):
        super().__init__()
        self.energy = HNetViaCompH(n_nodes, n_cmp, n_random_edges=n_random_edges) 

    def forward(self, x):
        x = self.energy(x)
        return x

class EdgePrediction(nn.Module):
    '''Module for predicting the logic gate for every pair of inputs'''
    def __init__(self, n_edges, n_cmp):
        super().__init__()
        self.n_cmp = n_cmp
        self.n_edges = n_edges

        # Weights representing the logic gate for every edge
        self.edge_weights = nn.Parameter(torch.zeros(n_cmp, n_edges, 4))

        # Weights for "switching" on or off the existence of an edge
        self.edge_switches = nn.Parameter(torch.zeros(n_cmp, n_edges, 1))

        # Continuous differentiation basis for logic gates
        prim_hs = torch.tensor([
                                [-1, 0.5, -1, 1],   # NOR 
                                [ 0, -0.5,  1, 0],  # NCONV
                                [1, -0.5, 0, 0],    # NIMPL 
                                [0, 0.5, 0, 0],     # AND
                                ]).float()

        # Parameter only here for observing logic gate changes over time during training
        self.edge_choices = nn.Parameter(torch.zeros(n_cmp, n_edges))

        self.register_buffer('prim_hs', prim_hs)
        self.register_buffer("pow_2", torch.tensor([1, 2, 4, 8]).float())
        self.init_weights()
    
    def init_weights(self):
        """
        Initialized weights
        """

        #torch.nn.init.uniform_(self.edge_switches, a=0, b=0.1)

        torch.nn.init.xavier_uniform_(self.edge_switches)
        torch.nn.init.xavier_uniform_(self.edge_weights)

        # Sets default gate for all edges to be ~Y (WIP)
        '''torch.nn.init.constant_(self.edge_weights[:,:,0], -0.01)
        torch.nn.init.constant_(self.edge_weights[:,:,1], 0.01)
        torch.nn.init.constant_(self.edge_weights[:,:,2], -0.01)
        torch.nn.init.constant_(self.edge_weights[:,:,3], 0.01)'''

        '''torch.nn.init.constant_(self.edge_weights[:,:,0], -0.01)
        torch.nn.init.constant_(self.edge_weights[:,:,1], 0.01)
        torch.nn.init.xavier_uniform_(self.edge_weights[:,:,2])

        with torch.no_grad():
            self.edge_weights[:,:,3] = -self.edge_weights[:,:,2]'''

        '''
        torch.nn.init.constant_(self.edge_weights[:,:,3], -0.01)
        torch.nn.init.constant_(self.edge_weights[:,:,0], 0.01)
        torch.nn.init.xavier_uniform_(self.edge_weights[:,:,1])
        with torch.no_grad():
            self.edge_weights[:,:,2] = -self.edge_weights[:,:,1]'''

    def log_gates(self, x):
        """
        Converts binary edge weights to numbers and sets it to the parameter edge_choices for logging during training
        
        Inputs
        ======
        x - n_cmp x n_edges x 4 (Tensor) binarized edge weights
        """
        edge_choices = torch.matmul(x, self.pow_2)
        self.edge_choices = nn.Parameter(edge_choices)

    def get_gates(self):
        """
        Converts full precision edge weights to primitive hamiltonian matrices
        Returns
        =======
        prim_hs - n_cmps x n_edges x 2 x 2 (Tensor[double]) primitive hamiltonians for each edge
        switched_off - (float) scalar value representing all the gates that have been turned off 
        """
        x = self.edge_weights

        x = x + (x.clamp(-SIGMOID_CLIPPING_CONST,SIGMOID_CLIPPING_CONST) - x).detach() 

        x = torch.nn.functional.sigmoid(x)

        x = x + (torch.round(x) - x).detach()

        edge_switches = self.edge_switches + (self.edge_switches.clamp(-SIGMOID_CLIPPING_CONST,SIGMOID_CLIPPING_CONST) - self.edge_switches).detach() 

        switches = torch.nn.functional.sigmoid(edge_switches)

        switches = switches + (torch.round(switches) - switches).detach()
        
        prim_hs = x * switches

        switched_off = self.n_edges - switches.sum(dim=(1,2))

        self.log_gates(prim_hs)

        return prim_hs, switched_off

    def forward(self, x):
        """
        Converts full precision edge weights to primitive hamiltonian matrices
        
        Inputs
        ======
        x - None
        Returns
        =======
        x - n_cmp x n_nodes x n_nodes (Tensor[double]) tensor of all composite hamiltonians
        switched_off - n_cmp x n_nodes (Tensor[double]) tensor of the k bias for each composite hamiltonians 
        """
        x, switched_off = self.get_gates()

        x = torch.matmul(x.view(-1,4), self.prim_hs)

        x = x.view(self.n_cmp, -1, 4)

        return x, switched_off

class HNetViaCompH(nn.Module):
    """
    Energy calculation based off of composite hamiltonians 
    """

    def __init__(self, n_nodes, n_cmp, n_random_edges=None, use_custom_op=True, use_pop_count=True, normalize=True, manual_h_generation=False):
        super().__init__()

        self.n_nodes = n_nodes
        self.n_cmp = n_cmp
        self.use_custom_op = use_custom_op
        self.use_pop_count = use_pop_count
        self.manual_h_generation = manual_h_generation
        self.normalize = normalize

        edges = _make_edge_coords(n_nodes, n_random_edges=n_random_edges)
        idx = torch.arange(self.n_nodes)

        self.register_buffer('edges', edges)
        self.register_buffer('idx', idx)
        self.register_buffer('h', torch.zeros((self.n_cmp, self.n_nodes, self.n_nodes)))
        self.register_buffer('k', torch.zeros(self.n_cmp))
        self.register_buffer('switched_off', torch.zeros(self.n_cmp))

        self.edge_pred = EdgePrediction(len(self.edges), n_cmp)

    def _create_h(self):
        """
        Generates primitive hamiltonians for each edge, then composes them into their respective composite hamiltonians
        """
        hs, self.switched_off = self.edge_pred(None)

        self.h, self.k = self._build_h(hs)


    def _build_h(self, hs):
        """
        Builds composite hamiltonians given the primitive hamiltonians
        
        Inputs
        ======
        hs - n_cmp x n_edges x 2 x 2 (Tensor)  tensor containing the primitive hamiltonian for a given edge

        Returns
        =======
        h - n_cmp x n_nodes x n_nodes (Tensor[double]) tensor of all composite hamiltonians
        k - n_cmp x n_nodes (Tensor[double]) tensor of the k bias for each composite hamiltonians 
        """

        device = torch.device("cuda:0") if hs.is_cuda else torch.device("cpu")
        h = torch.zeros((self.n_cmp, self.n_nodes, self.n_nodes), device=device)
        k = torch.zeros(self.n_cmp, device=device)
        intermed_res_a = torch.zeros((self.n_cmp, self.n_nodes), device=device)
        intermed_res_c = torch.zeros((self.n_cmp, self.n_nodes), device=device)

        # Sum up all the k bias from each component
        k += torch.sum(hs[:,:,3], 1)
        
        # Add non-diagonal elements of primitive hamiltonians to the non-diagonal (n)  
        ib0, ib1 = self.edges[:, 0], self.edges[:, 1]
        h[:, ib0, ib1] += hs[:, :, 1]
        h[:, ib1, ib0] += hs[:, :, 1]
        
        x_a = hs[:, :, 0]
        ia = self.edges[:,0].unsqueeze(0).repeat(self.n_cmp, 1)
        intermed_res_a.scatter_add_(-1, ia, x_a)
        h[:, self.idx, self.idx] += intermed_res_a

        x_c = hs[:, :, 2]
        ic = self.edges[:,1].unsqueeze(0).repeat(self.n_cmp, 1)
        intermed_res_c.scatter_add_(-1, ic, x_c)
        h[:, self.idx, self.idx] += intermed_res_c

        return h, k

    def forward(self, node_activations:torch.Tensor):
        """
        Applys composite hnet matrices to node_activations and produces corresponding energy.

        Inputs
        ======
        node_activations - n_pts x n_nodes (Tensor) list of node activations for each datapoint

        Returns
        =======
        energies - n_pts x n_cmp (Tensor[double])
        """

        if self.training and not self.manual_h_generation:
            self._create_h()

        if self.use_custom_op:
            if self.use_pop_count:
                energies = HNetOpPopCount.apply(self.h, self.k, node_activations, len(self.edges), self.switched_off)
            else:
                energies = HNetOpCountBits.apply(self.h, self.k, node_activations)
        else:
            t = torch.einsum('kj,nij,ki->kn',node_activations, self.h, node_activations)

            # Add k
            energies = t + self.k.unsqueeze(0)
            
            if self.use_pop_count:
                # Transform from bitcount to popcount
                energies = 2*energies - len(self.edges) + self.switched_off

        if self.normalize:
            energies = energies / (len(self.edges) - self.switched_off)

        return energies # batch_size, n_cmp
    
class Binarize(nn.Module):
    """Binarizes inputs with step function"""
    def __init__(self, dim=(2,3), get_scale=False):
        super().__init__()
        self.dim = dim
        self.get_scale = get_scale

    def forward(self, x):
        if self.training:
            x = x + ((x > 0).float() - x).detach()
        else:
            x = (x > 0).float()
        
        if self.get_scale:
            scale = x.abs().mean(dim=self.dim, keepdim=True).detach()
            return x, scale
        else:
            return x
        
class SigmoidBinarize(nn.Module):
    """Binarizes inputs with sigmoid round function"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            x = x + (x.clamp(-SIGMOID_CLIPPING_CONST,SIGMOID_CLIPPING_CONST) - x).detach() 
            x = nn.functional.sigmoid(x)
            x = x + (torch.round(x) - x).detach()
        else:
            x = (x > 0).float()
        
        return x

# === FILE-PRIVATE FUNCTIONS ===

def _make_edge_coords(n_nodes, n_random_edges=None):
    edges_generated = set()
    all_edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue

            if (i,j) not in edges_generated:
                edges_generated.add((i,j))
                edges_generated.add((j,i))
                all_edges.append((i,j))

    if n_random_edges is not None:
        edges = random.sample(all_edges, n_random_edges)
    else:
        edges = all_edges

    coords = torch.tensor(edges)
    
    return coords