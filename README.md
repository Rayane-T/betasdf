# Beta SDF 

## Generating the Dataset :

You first need to generate a SMPL mesh with SMPL.py algorithm, sample points around the body with points_generation.py algorithm, do that for how many beta you want in your dataset, then all_data_with_points.py, take n as an hyperparameter, will concatenate the points generated with their respective betas to generate the complete dataset. (Current [Dataset](https://drive.google.com/drive/folders/1ep9VJdz7qqR5bn6jES7NTwcYmExv57Js?usp=sharing))

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


## Analyse 

Monitor the loss curves at [loss_curves_b_betasdf.ipynb](/loss_curves_b_betasdf.ipynb) and [loss_siren_decoder.ipynb](/loss_siren_decoder.ipynb)


## Visualize

Visualize the reconstruction with [voxel_marching.py](/voxel_marching.py), don't forget replacing the state dict with your trained model one : 

```python
loadmodel.load_state_dict(torch.load("PATH/TO/STATE_DICT"))


visualize_marchingcubes_mesh(loadmodel, "PATH/TO/RESULT", True, 128)
```

An .obj file will be generated, you can visualize it with Blender for instance. 

<u>Notes</u>:
If mapping you need to add the location of the B_gauss.pt file in the utils.py
 ```python
 def get_sdfgrid(model, grid_res=20, mapping=True, device='cuda'): #Warning mapping actived
    B = torch.load("B_gauss/location.pt")
    
    ...
    
    beta = np.load('rayane/t_posemesh/t_pose_betas/beta_000.npy')[:10]

    ...
    
    return final_outputs
 
 ``` 

## Versions

torch                    2.0.1

numpy                    1.24.3

matplotlib               3.7.1

mesh-to-sdf              0.0.14

trimesh                  3.21.7

onnx                     1.14.0

onnxruntime              1.15.1

opencv-python            4.7.0.72
 
