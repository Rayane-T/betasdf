# Beta SDF 

## Introduction:

This MLP model will generate a 3D mesh. There's types of model you can use, the beta siren model or the betasdf model. For the moment the one showing the best results is the the betasdf model with a mapping (the method is the beta Fourier method).
The goal previously was to train a model to give sdf values out of a grid resulting in another grid to finally print a mesh. [See the report for more explanations](). To generate the mesh, we need then two things. The dataset to train the model and then the grid of points to run the generation with the set of weights previously obtained.


On the NAS, you should have the files for the training 'ex23' (you have to put the dataset in the ./dataset/t_posemesh folder if you download it from the NAS server) and the result as an .obj file.

## Fast usage settings:

To generate your first mesh with this folder, you will have to first modify a few things before doing the generation of the mesh you want.
First you have to check for the .json file you are using. This can be seen at the beginning of the file [single_fourier_grad_reload.py](single_fourier_grad_reload.py).
```python
# load specs
experiment_directory="./"
experiment="ex23" # You have to check there to see what json file you are using
specs = ws.load_experiment_specifications(experiment_directory,experiment) #read json spec info
#continue_from={"model":"ex13_dragon_deepsdf_8_4999" ,"log":"ex13_dragon_history_8_4999"} # every time start from a this should be given
continue_from=None
```
You can modify the json files to customize some training functionality. Still at the beginning, in the setup you will also be able to chose your dataset in the [./dataset/t_posemesh/](./dataset/t_posemesh/) folder. You can test several dataset to train your model here. The default one is the [data_betas_tpose_0_to_2-003.npy]().

```python
#loading data set (current max 15 Million points -> 3 Beta)
with open('./dataset/t_posemesh/data_betas_tpose_0_to_2-003.npy', 'rb') as f: # We chose the dataset here
    points = np.load(f)
    sdfs = np.load(f) 
    normals =np.load(f)
    surfaces =np.load(f)
```

Once done you can execute the file [single_fourier_grad_reload.py](single_fourier_grad_reload.py). When you are doing the training of the model, you have some snapshot through the process. These enable you to directly generate the mesh while while the training is not over yet. When a snapshot is done, it saves a file in the [weights](./weights), [log](./log) and [optimizer](./optimizer) folder.

With the set of weights obtained, you can go to [voxel_marching.py](./voxel_marching.py) file and change the set of weights used in the $torch.load()$ function and the name of the file associated in the $visualize\_marchingcubes\_mesh()$ function.
```python
loadmodel.load_state_dict(torch.load("./weights/name_of_the_set_of_weights.pt"))


visualize_marchingcubes_mesh(loadmodel, "./results/name_of_the_result_file", True, 128)
``` 
Run the file and then you can vizualise the result obtained in the [results](./results) folder (it is an .obj file) by using Blender or simply 3D viewer on Windows (installed by default).

<u>Notes</u>:

In case the fourier mapping is used ('mapping' == true) you need to add the location of the B_gauss.pt file in the [utils.py](/utils.py)
 ```python
 def get_sdfgrid(model, grid_res=20, mapping=True, device='cuda'): #Warning mapping actived
    B = torch.load("B_gauss/location.pt")
    
    ...
    
    beta = np.load('dataset/t_posemesh/t_pose_betas/beta_000.npy')[:10]

    ...
    
    return final_outputs
 
 ``` 

You can also track the loss through the training with the file [loss_curves_b_betasdf.ipynb](./loss_curves_b_betasdf.ipynb).

## Generating the Dataset :

You first need to generate a SMPL mesh with SMPL.py algorithm, sample points around the body with points_generation.py algorithm, do that for as many betas as you want in your dataset, then all_data_with_points.py, take n as an hyperparameter, will concatenate the points generated with their respective betas to generate the complete dataset. (Current [Dataset](https://drive.google.com/drive/folders/1ep9VJdz7qqR5bn6jES7NTwcYmExv57Js?usp=sharing))
If you want to use directly one of the dataset generated so far, you have to check for the folder [./dataset/t_posemesh/](./dataset/t_posemesh/)

[SMPL.py](/SMPL.py)
```python
import torch
import json
import sys
import numpy as np
from util_smpl import batch_global_rigid_transformation, batch_rodrigues, reflect_pose, quat2mat 
from SMPL_interpolation import interpolate_pose
import torch.nn as nn
import os
import trimesh
from util_smpl import axisangle_quat 
...

beta = np.load("betas.npy")[0][:10]
```
<u>Notes</u>:
Change this to your beta parameter.

[points_generation.py](/points_generation.py)
```python
mesh = trimesh.load('./dataset/t_posemesh/your_mesh.obj',process=False)

nb_samples=5000000

points, sdf, gradient = sample_sdf_near_surface(mesh, number_of_points=nb_samples,return_gradients=True)
with open("your_sampled_mesh_location.npy", 'wb') as f:
    np.save(f, points)
    np.save(f, sdf)
    np.save(f,gradient)
    np.save(f,surface)
```
<u>Notes</u>:
Change your mesh location and sampled mesh location

[all_data_with_points.py](/all_data_with_points.py)
```python
number_of_mesh = ...
number_of_beta = ...
```
<u>Notes</u>:
Change this to your number of mesh and betas (should be equal) that the dataset will have

## Training 

[single_fourier_grad_reload](/single_fourier_grad_reload.py)

```python
import torch
from model_betasdf import Decoder
from model_betasdf_with_siren import Siren_Decoder
import numpy as np
from torch.utils.data import DataLoader, TensorDataset 
import torch.nn.functional as F

import workspace as ws
from utils_reload import *

...

# Mapping parameters
mapping_size = ... # 128 / 256
mapping_method = 'gaussian'
mapping_scale = ... # 1. / 10. / 100.

...

# To better understand this, read the paper associated : Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
if mapping: 
    features = torch.from_numpy(points[:, :3] / 2 + np.array([[0.5, 0.5, 0.5]])).double()  # [0,1]
    features = torch.cat([features, torch.from_numpy(points[:, 3:])], dim=1)  # concatenate along dimension 1
    labels = torch.from_numpy(sdfs / 2)
    normals = torch.from_numpy(normals).double()
    surfaces = torch.from_numpy(surfaces)

    def input_mapping(x, B):
        if B is None:
            return x
        else:
            x_proj = (2. * np.pi * x[:,:3].double()) @ B.t().double() # when map size = 3 and batch size = 16384 : 16384 x 3 matmul 3 x 3 =  16384 x 3
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).double() # 16384 x 6
    
    B_gauss = (torch.randn((mapping_size, 3)).to(device) * mapping_scale).double()
    torch.save(B_gauss, '{}_{}_{}_B_gauss.pt'.format(experiment, mapping_size,mapping_scale))
```

<u>Notes</u>:
For the Mapping experiments, you can adjust both the mapping size and mapping scale. 
I didn't manage to obtain good results from siren, leaving further experiments for the next-comers.


## Versions

torch                    2.0.1

numpy                    1.24.3

matplotlib               3.7.1

mesh-to-sdf              0.0.14

trimesh                  3.21.7

onnx                     1.14.0

onnxruntime              1.15.1

opencv-python            4.7.0.72
 
