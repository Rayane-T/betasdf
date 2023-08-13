import torch
import json
from model_betasdf import Decoder
from model_betasdf_with_siren import Siren_Decoder
import numpy as np
from torch.utils.data import DataLoader, TensorDataset 
import torch.nn.functional as F
# Import freeze_support for Windows
# from multiprocessing import freeze_support

import workspace as ws
from utils_reload import *
from wnn_callback import WeightForecasting


# freeze_support()  # if using windows

if torch.cuda.is_available():
    device = 'cuda' 
print(device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)


# load specs
experiment_directory="./"
experiment="ex25" 
specs = ws.load_experiment_specifications(experiment_directory,experiment) #read json spec info
#continue_from={"model":"ex13_dragon_deepsdf_8_4999" ,"log":"ex13_dragon_history_8_4999"} # every time start from a this should be given
continue_from=None


#mapping parameters last parameter used
mapping_size = 256
mapping_method = 'gaussian'
mapping_scale = 1.

#experimentation setting
mode = specs["Mode"]
mapping = specs["Mapping"]  #bool
n_epochs = specs["Epochs"]
save_every=specs["SnapshotFrequency"]
save_others=specs["AdditionalSnapshots"]

# model   #############################################
nb_layer=10

if mode=="betasdf": # Vanilla betasdf
    model = Decoder(in_fc1=mapping_size*2).to(device)
elif mode=="beta fourier": #Betasdf with fourier mapping
    model = Decoder(in_fc1=mapping_size*2).to(device)
elif mode=="beta siren": #Betasdf with siren layers
    model = Siren_Decoder().to(device)    

print(model)

#loading data set (current max 15 Million points -> 3 Beta)
with open('./dataset/t_posemesh/data_betas_tpose_0_to_2-003.npy', 'rb') as f: 
    points = np.load(f)
    sdfs = np.load(f) 
    normals =np.load(f)
    surfaces =np.load(f)

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
    
else:
    features = torch.from_numpy(points).float() # should be like between [-1,1] for siren
    labels = torch.from_numpy(sdfs)
    normals = torch.from_numpy(normals).float()
    surfaces = torch.from_numpy(surfaces)


dataset = TensorDataset(features, labels, normals, surfaces)

#sampling preparation
sampling_weights = np.ones_like(sdfs)
positive_part = np.sum(sdfs>0) / sdfs.shape[0]
negative_part = np.sum(sdfs<0) / sdfs.shape[0]

sampling_weights[sdfs>0] = negative_part
sampling_weights[sdfs<0] = positive_part

sampler = torch.utils.data.sampler.WeightedRandomSampler(sampling_weights, len(sampling_weights))         

train_loader = DataLoader(dataset,
                        batch_size=16384,
                        sampler = sampler,
                        num_workers=8)
                                                
        
def deepsdfloss(outputs, targets, delta=0.1): #Deepsdf loss fct used
    return torch.mean(torch.abs(
        torch.clamp(outputs, min=-delta, max=delta) - torch.clamp(targets, min=-delta, max=delta)
    ))

loss_l1=torch.nn.L1Loss(reduction="mean")


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def sdfloss(model_output, gt):
    '''
    x: batch of input coordinates
    y: usually the output of the trial_soln function
    '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    gt_surfaces=gt['surfaces']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']
    grad = gradient(pred_sdf, coords)
    # print("pred_sdf shape:",pred_sdf.shape)
    # print("coords shape:",coords.shape)
    # print("grad shape:",grad.shape)
    # print("gt_normals shape:",gt_normals.shape)
    on_surface_sample=torch.count_nonzero(gt_surfaces).item()  #if samples around the surface it's 1, in the space 0


    # three constraints: on sdf value, grad directoin, grad modulus==1
    sdf_constraint = loss_l1(pred_sdf,gt_sdf)
    # print(grad.shape,gt_normals.shape)
    
    # normal_constraint = 1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None]

    #normal_constraint = torch.where(gt_surfaces != 0, 1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
    #                       torch.zeros_like(grad[..., :1]))                              
    
    grad_constraint = torch.abs(grad.norm(dim=-1) - 1)

    #return sdf_constraint,normal_constraint.sum()/(on_surface_sample+1e-6), grad_constraint.mean()
    return sdf_constraint, grad_constraint.mean()
    
def save_checkpoints(epoch,decoder,optimizer,loss_history,experiment):
    model_ckpt="{}_body_deepsdf_{}_{}".format(experiment,nb_layer,epoch)
    log_ckp="{}_body_history_{}_{}".format(experiment,nb_layer,epoch)
    
    save_model(ws, experiment_directory, model_ckpt, decoder, epoch)
    save_optimizer(ws, experiment_directory,model_ckpt, optimizer, epoch)
    save_loss(ws, experiment_directory, log_ckp, loss_history, epoch)

def save_checkpoints_wnn(epoch,decoder,optimizer,loss_history,experiment):
    model_ckpt="{}_body_deepsdf_{}_{}_wnn".format(experiment,nb_layer,epoch)
    log_ckp="{}_body_history_{}_{}_wnn".format(experiment,nb_layer,epoch)
    
    save_model(ws, experiment_directory, model_ckpt, decoder, epoch)
    save_optimizer(ws, experiment_directory,model_ckpt, optimizer, epoch)
    save_loss(ws, experiment_directory, log_ckp, loss_history, epoch)

def load_checkpoints(continue_from_ckpt, ws, experiment_directory, decoder, optimizer):

    
    model_ckpt =continue_from_ckpt["model"]
    log_ckpt=continue_from_ckpt["log"]
        
    print('continuing from "{}"'.format(model_ckpt))

    decoder, model_epoch =load_model(ws, experiment_directory, model_ckpt, decoder)
    optimizer, optimizer_epoch =load_optimizer(ws, experiment_directory, model_ckpt, optimizer)
    
    loss_log = load_loss(ws, experiment_directory, log_ckpt)

    start_epoch = model_epoch + 1

    return start_epoch, decoder, optimizer, loss_log




lr_schedules = get_learning_rate_schedules(specs)
optimizer= torch.optim.Adam(model.parameters(), lr=lr_schedules[0].get_learning_rate(0))


loss_history = [] #of all epochs
sdf_loss_history=[]
normal_loss_history=[] 
grad_loss_history=[]

start_epoch=0

if continue_from is not None:
        start_epoch, model, optimizer, loss_log \
        = load_checkpoints(continue_from, ws, experiment_directory, model, optimizer)

        loss_history=loss_log["total_loss"]
        sdf_loss_history=loss_log["sdf_loss"]
        normal_loss_history=loss_log["normal_loss"] 
        grad_loss_history=loss_log["grad_loss"]
    

save_others.append(start_epoch+100) 

for epoch in range(start_epoch,n_epochs):
    model.train(True)
    print(f"\nEpoch {epoch}")
    
    running_loss = 0
        
    total_loss = 0
    total_sdf_loss = 0
    total_normal_loss = 0
    total_grad_loss = 0
    
    
    adjust_learning_rate(lr_schedules, optimizer, epoch) #adapt lr 
    
    
    for i, data in enumerate(train_loader, 0):
        x, y = data[0].to(device), data[1].unsqueeze(1).to(device)
        normals, surfaces=data[2].to(device),data[3].unsqueeze(1).to(device)
        # print("normal shape:",normals.shape)
        
        x.requires_grad=True  #for training grad
        gt={"sdf":y,"normals":normals,"surfaces":surfaces}
        # print(x[:,3:].shape)
        
        e = input_mapping(x[:,:3], B_gauss)

        # print("x shape:",x.shape)
        # print("e shape:",e.shape)
        v = torch.cat((e, x[:,3:]), dim=1)
        # print("v shape:",v.shape)
        if mapping:
            model_output = model(v)
        else:
            model_output = model(x)  #todo 
        
        sdf_loss, grad_loss=sdfloss(model_output, gt)
        loss=sdf_loss+grad_loss*0.001                  #weighted loss***********************************
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
                
        total_loss += loss.item()
        total_sdf_loss += sdf_loss.item()
        
        total_grad_loss += grad_loss.item()
        WeightForecasting(model, i)
        
        if i % 5 == 4:
            print(i, running_loss)
            running_loss = 0
    
    #after one epoch
    loss_history.append(total_loss) 
    
    sdf_loss_history.append(total_sdf_loss)
    # normal_loss_history.append(total_normal_loss) 
    grad_loss_history.append(total_grad_loss)
    
        
    print("**") 
    print("end of the epoch loss: ", total_loss)  
    print("end of the epoch sdf: ", total_sdf_loss)
    # print("end of the epoch normal: ", total_normal_loss)
    print("end of the epoch grad: ", total_grad_loss)
    print("**") 

    if (epoch % save_every==0 or epoch in save_others):
        t_history={"total_loss":loss_history,"sdf_loss":sdf_loss_history,\
        "normal_loss":normal_loss_history,"grad_loss":grad_loss_history}    
        
        
        if continue_from is None: 
            save_checkpoints(epoch, model, optimizer,t_history,experiment)
        else:
            experiment_re=experiment+"from"+continue_from["model"].split('_')[0]
            save_checkpoints(epoch, model, optimizer,t_history,experiment_re)
