# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:24:34 2022

@author: Lenovo
"""
import torch
from models_fourier import Siren_model,Siren_2heads_model,Siren_grad
from model_betasdf import Decoder
import numpy as np
from torch.utils.data import DataLoader, TensorDataset 
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import workspace as ws
from utils_reload import *

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mapping=False

#mode="siren simple"
#filename="ex7_dragon_deepsdf_8_8999"
#eval_data="dragon_evalsamples.npy"

mode="beta sdf"
filename="ex17_body_deepsdf_8_1999"
eval_data="data_with_betas_test.npy"

experiment_directory="./"

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     #random.seed(seed)
     torch.backends.cudnn.deterministic = True
# setup_seed
setup_seed(20)


#mapping parameters
mapping_size = 256
mapping_method = 'gaussian'
mapping_scale = 12.


# model   #############################################
nb_layer=8

if mode=="fourier":
    from models_fourier import SingleShapeSDF
    model = SingleShapeSDF([512 for i in range(nb_layer)],mapping_size*2).to(device)
elif mode=="siren fourier":
    model = Siren_model(8, mapping_size*2, 512).to(device)
elif mode=="beta sdf":
    model = Decoder().to(device)
print(model)


with open("data/"+ eval_data, 'rb') as f:
    points = np.load(f)
    sdfs = np.load(f) 

if mapping:    
    features = torch.from_numpy(points/2+np.array([[0.5,0.5,0.5]])).float() # should be like between [0,1] 
    labels = torch.from_numpy(sdfs/2)
    
    def input_mapping(x, B):
        if B is None:
            return x
        else:        
            x_proj = (2. * np.pi * x) @ B.t()
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    #B_gauss = (torch.randn((mapping_size, 3)).to(device) * mapping_scale)
    #torch.save(B_gauss, '{}_B_gauss.pt'.format(experiment))
    
    
else:
    features = torch.from_numpy(points).float() # should be like between [-1,1] for siren
    labels = torch.from_numpy(sdfs)


dataset = TensorDataset(features, labels)

#sampling preparation
sampling_weights = np.ones_like(sdfs)
positive_part = np.sum(sdfs>0) / sdfs.shape[0]
negative_part = np.sum(sdfs<0) / sdfs.shape[0]

sampling_weights[sdfs>0] = negative_part
sampling_weights[sdfs<0] = positive_part

sampler = torch.utils.data.sampler.WeightedRandomSampler(sampling_weights, len(sampling_weights))         

eval_loader = DataLoader(dataset,
                          batch_size=16384,
                          sampler = sampler,
                          num_workers=8)

loss_l1=torch.nn.L1Loss(reduction="sum")



model, saved_epoch=load_model(ws, experiment_directory, filename, model)
print("model loaded")

model.eval()

with torch.no_grad():             
    total_loss = 0
    loss_history = []
    total_samples=0

    for i, data in tqdm(enumerate(eval_loader, 0)):
        x, y = data[0].to(device), data[1].unsqueeze(1).to(device)
        
        total_samples+=x.shape[0]
        
        if mapping:
            model_output = x#model(input_mapping(x, B_gauss))
        else:
            model_output = model(x)  #todo 
        
        #loss = deepsdfloss(y_pred, y)
        loss = loss_l1(model_output['model_out'],y)
        loss_history.append(loss.item())
        total_loss += loss.item()
    print("sdf model output: ", model_output['model_out'])
    print("sdf ground truth: ",sdfs)     
sdf_loss=total_loss/total_samples 
                    
print("**") 
print("total samples: ", total_samples)
print("eval loss: ", sdf_loss)
print("**") 