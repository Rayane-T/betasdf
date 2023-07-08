import torch
import numpy as np 

number_of_mesh = 2 
number_of_beta = 2 

poin = np.empty((0, 3))
sdfs = np.empty((0,))
n = np.empty((0,3))
surf = np.empty((0,))

for i in range(9):
    with open('./dataset/t_posemesh/t_posesampled_{}.npy'.format(i), 'rb') as f:
        points = np.load(f)
        sdf = np.load(f) 
        gradient =np.load(f)
        surface =np.load(f)

    poin = np.vstack((poin, points))
    sdfs = np.concatenate((sdfs, sdf))
    n =  np.concatenate((n, gradient))
    surf =  np.concatenate((surf, surface))

poin = torch.from_numpy(poin)

betas = torch.empty(0)

# fit the beta 
for j in range(9):
    with open('./dataset/t_posemesh/t_pose_betas/beta_00{}.npy'.format(j), 'rb') as f:
        beta = np.load(f)[:10]

    beta = torch.from_numpy(beta)
    beta_repeated = beta.repeat(5000000, 1)
    beta_repeated = beta_repeated.reshape(5000000, 10)

    betas = torch.cat((betas, beta_repeated), dim=0)

#beta_repeated.type(torch.float64)

# concatenate points and betas

poin = torch.cat((poin, betas), dim=1)
    
# Save in npy
with open("data_betas_tpose.npy", 'wb') as fs:
    np.save(fs, poin)
    np.save(fs, sdfs)
    np.save(fs,n)
    np.save(fs,surf)

print("finished")