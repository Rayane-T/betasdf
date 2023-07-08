from test_fnn import visualize_voxels_png,visualize_marchingcubes_mesh,visualize_marchingcubes
from model_betasdf import Decoder
from model_betasdf_with_siren import Siren_Decoder
from models_fourier import Siren_model, Siren_grad
import torch
import trimesh

device = 'cuda' if torch.cuda.is_available() else 'cpu'

nb_layer=8

mode="beta sdf"

if mode=="beta sdf":
    loadmodel= Decoder().to(device)
elif mode=="siren decoder":
    loadmodel = Siren_Decoder().to(device)	
elif mode=="siren simple":
    loadmodel=Siren_model(8,3,512).to(device)
elif mode=="siren grad":
    loadmodel=Siren_grad(8,3,512).to(device)
else:
    from models import SingleShapeSDF
    loadmodel=SingleShapeSDF([512 for i in range(nb_layer)]).to(device)
print(loadmodel)


loadmodel.load_state_dict(torch.load("./weights/ex21_body_deepsdf_9_100.pt"))


visualize_marchingcubes_mesh(loadmodel, "./results/ex21_body_deepsdf_9_100", True, 128)

print("visualization done")
