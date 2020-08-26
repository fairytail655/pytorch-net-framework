import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

# import torch.nn._functions as tnnf

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        # if input.size(1) != 784:
        input.data=Binarize(input.data)
        # if not hasattr(self.weight,'org'):
        #     self.weight.org=self.weight.data.clone()
        temp = self.weight.data
        self.weight.data=Binarize(temp)

        out = nn.functional.linear(input, self.weight)

        self.weight.data = temp

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        temp = self.weight.data
        self.weight.data=Binarize(temp)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        self.weight.data = temp

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

def SelfBinarize(tensor, epoch, is_training):
    if is_training == False:
        return tensor.sign()
    else:
        return torch.tanh((int(epoch/40)*20+1) * tensor)

class SelfBinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(SelfBinarizeLinear, self).__init__(*kargs, **kwargs)
        self.is_training = True
        self.epoch = 0

    def forward(self, input):
        bw = SelfBinarize(self.weight, self.epoch, self.is_training)
        out = nn.functional.linear(input, bw)

        return out

class SelfBinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(SelfBinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.is_training = True
        self.epoch = 0

    def forward(self, input):
        bw = SelfBinarize(self.weight, self.epoch, self.is_training)
        out = nn.functional.conv2d(input, bw, None, self.stride, self.padding, self.dilation, self.groups)

        return out

class SelfTanh(nn.Module):

    def __init__(self, *kargs, **kwargs):
        super(SelfTanh, self).__init__(*kargs, **kwargs)
        self.is_training = True
        self.epoch = 0

    def forward(self, input):

        return SelfBinarize(input, self.epoch, self.is_training)