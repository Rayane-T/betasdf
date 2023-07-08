import numpy as np
import torch


def input_mapping(x, B):
    if B is None:
        return x
    else:        
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def get_torchgrid(grid_res=20, mapping=True, device='cuda'):
    grid = np.array(
        [[i // (grid_res ** 2), (i // grid_res) % grid_res, i % grid_res]
         for i in range(grid_res ** 3)], dtype=np.single)
    
    if mapping:
        grid = ((grid - grid_res / 2) / grid_res) + np.array([[0.5,0.5,0.5]])  #[0,1]
    else:   
        grid = ((grid - grid_res / 2) / grid_res) * 2.0 # [-1,1]
    
    grid = torch.from_numpy(grid).to(device=device, dtype=torch.float32)
    return grid

def get_sdfgrid(model, grid_res=20, mapping=True, device='cuda'): #Attention mapping activé
    B = torch.load("ex20_3_10.0_B_gauss.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid = get_torchgrid(grid_res, mapping, device)

    #print("Grid built")
    #print("Grid:", grid[1])
    
    beta = np.load('rayane/new/beta_000.npy')[:10]
    beta = torch.from_numpy(beta).to(device=device, dtype=torch.float32)
    beta = beta.repeat(grid_res ** 3, 1)
    #print("Beta:", beta.device, beta.shape)
    if mapping:    
        grid = input_mapping(grid,B)
    grid = torch.cat((grid, beta), dim=1)
    outs = []
    chunk_size = 100  # based on memory constraints

    with torch.no_grad():
        for i in range(0, grid.shape[0], chunk_size):
            grid_chunk = grid[i:i+chunk_size]
            if mapping:
                out = model.forward(grid_chunk)
            else:
                out = model.forward(grid_chunk)
            outs.append(out['model_out'])
        
        final_outputs = torch.cat(outs, dim=0).reshape((grid_res, grid_res, grid_res)).cpu().numpy()
    
    return final_outputs



def get_balancedsampler(labels):
    # TODO: this is still not an ideal solution for balanced sampling
    # balanced sampling
    sampling_weights = np.ones_like(labels)
    positive_part = np.sum(labels > 0) / labels.shape[0]
    negative_part = np.sum(labels < 0) / labels.shape[0]
    sampling_weights[labels > 0] = negative_part
    sampling_weights[labels < 0] = positive_part
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sampling_weights, len(sampling_weights))
    return sampler
