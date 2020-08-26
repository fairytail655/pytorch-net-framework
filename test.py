import torch
from models import *

model_path = "C:\\Users\\26235\\Desktop\\code\\python\\pytorch-net-framework\\results\\SelfBinaring\\checkpoint.pth.tar"

net = resnet20()
# checkpoint = torch.load(model_path)
# net.load_state_dict(checkpoint['state_dict'])

for m in net.modules():
    print(m)
    break

# i = 0
# for m in net.modules():
#     i = i + 1
#     if i == 2:
#         print(m)
#         print("--------- ")
#         print(m.weight)
#         break

# for name, param in net.named_parameters():
#     print(name, param.data)
#     break
