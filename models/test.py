import torch
import torch.nn as nn
from .binarized_modules import *

data = torch.FloatTensor([[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]])
net = SelfBinarizeConv2d(1, 1, 3)
net.is_training = True
for weight in net.wei

print(net)
