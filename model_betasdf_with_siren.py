#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class SineLayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, is_last=False, omega_0=30, first_omega=30000):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_last = is_last
        self.first_omega = first_omega
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.first_omega, 
                                             np.sqrt(6 / self.in_features) / self.first_omega)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return self.linear(input) if self.is_last else torch.sin(self.omega_0 * self.linear(input))

class Siren_Decoder(nn.Module):
    def __init__(
        self,
        #latent_size=10,
        dims = [512,512,512,768,512,512,512,512],
        dropout=[0,1,2,3,4,5,6,7],
        dropout_prob=0.2,
        norm_layers=[0,1,2,3,4,5,6,7],
        latent_in=[4],
        num_betas=10,
    ):
        super().__init__()

        dims = [3] + dims + [1]
        self.fc1 = SineLayer(num_betas+3, 256, is_first=True) 
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in

        self.num_betas = num_betas
        """
        TODO : CHECK RESULTS AND COMPARE WITH BOTH ENTRIES IS_FIRST AND SINGLE ENTRY
            : Visualize with marching cubes
            : setup docker wsl
        """
        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0] - 256 # dims[4] - (256 -> layer qui va concatenate + 3(skip connection))
            else:
                out_dim = dims[layer + 1]

            if layer in self.norm_layers:
                setattr(
                    self,
                    "siren" + str(layer),
                    SineLayer(dims[layer], out_dim),
                )
            else:
                setattr(self, "siren" + str(layer), SineLayer(dims[layer], out_dim, is_last=True))

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        #self.th = nn.Tanh()
        #print(self)

    # input: N x (3+num_betas)
    def forward(self, xyz_betas):

        shape_emb = self.fc1(xyz_betas) # fully connected layer with activ fct
        #shape_emb = self.relu(xyz_betas) # activation function

        xyz = xyz_betas[:,:3] # 3 premiere colonne correspondant au 3 valeurs x, y et z
        x = xyz

        for layer in range(0, self.num_layers - 1):
            siren = getattr(self, "siren" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, xyz, shape_emb], 1) #shape_emb :512, xyz = 16384,

            if layer<self.num_layers-2:
                x = siren(x)
            if layer < self.num_layers - 2:
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = siren(x)
        """
        if hasattr(self, "th"):
            x = self.th(x)
        """
        return {'model_in': xyz_betas, 'model_out': x}
