#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(
        self,
        #latent_size=10,
        dims = [512,512,512,768,512,512,512,512],
        dropout=[0,1,2,3,4,5,6,7],
        dropout_prob=0.2,
        norm_layers=[0,1,2,3,4,5,6,7],
        latent_in=[4],
        num_betas=10,
        in_fc1=512
    ):
        super(Decoder, self).__init__()

        dims = [3] + dims + [1]
        self.fc1 = nn.utils.weight_norm(nn.Linear(num_betas+in_fc1, 256)) # != 3 si mapping et 3 sinon

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in

        self.num_betas = num_betas

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0] - 256 # dims[4] - (256 -> layer qui va concatenate + 3(skip connection))
            else:
                out_dim = dims[layer + 1]

            if layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()
        #print(self)

    # input: N x (3+num_betas)
    def forward(self, xyz_betas):
        xyz_betas = xyz_betas.float()
        xyz = xyz_betas[:, :3]  # 3 first columns corresponding to x, y, and z values

        xyz_betas = self.relu(self.fc1(xyz_betas))  # fully connected layer # check re-assignation

        x = xyz

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, xyz, xyz_betas], 1)  # shape_emb: 512, xyz = 16384,

            if layer < self.num_layers - 2:
                x = lin(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = lin(x)

        if hasattr(self, "th"):
            x = self.th(x)

        return {'model_in': xyz_betas, 'model_out': x}

