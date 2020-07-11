from models import *
import torch
from torchsummary import summary

model = mobilenet_v2()
device = torch.device('cuda')
model = model.to(device)
summary(model, (3, 32, 32))