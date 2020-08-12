import torch
from models import *

model_path = "C:\\Users\\26235\\Desktop\\code\\python\\pytorch-net-framework\\results\\SelfBinaring\\checkpoint.pth.tar"

net = SelfBinaring()
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['state_dict'])

for name, param in net.named_parameters():
    print(name, param.data)
    break
