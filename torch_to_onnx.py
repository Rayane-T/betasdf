import torch
from model_betasdf import Decoder

model = Decoder()
model.load_state_dict(torch.load("./weights/ex20_body_deepsdf_9_100.pt"))

points = torch.tensor([[2.,3.,1.,2.,3.,5.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]])

torch.onnx.export(model,
                 points,
                 "model_betasdf.onnx",
                 verbose=True,
                 export_params=True,
                 )