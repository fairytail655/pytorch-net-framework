import torch.nn as nn
import torchvision.transforms as transforms
import math
from .binarized_modules import  SelfBinarizeConv2d, SelfBinarizeLinear, SelfTanh

__all__ = ['SelfBinaring']

def Binaryconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return SelfBinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def init_model(model):
    for m in model.modules():
        if isinstance(m, SelfBinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,do_bntan=True):
        super(BasicBlock, self).__init__()

        self.conv1 = Binaryconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = SelfTanh()
        self.conv2 = Binaryconv3x3(planes, planes)
        self.tanh2 = SelfTanh()
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.do_bntan=do_bntan
        self.stride = stride

    def forward(self, x):

        residual = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            if residual.data.max()>1:
                import pdb; pdb.set_trace()
            residual = self.downsample(residual)

        out += residual
        # if self.do_bntan:
        out = self.tanh2(out)

        return out

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.is_training = True
        self.epoch = 0

    def _make_layer(self, block, planes, blocks, stride=1,do_bntan=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SelfBinarizeConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes,do_bntan=do_bntan))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = input[0]
        # is_training = input[1]
        # epoch = input[2]
        for m in self.modules():
            if isinstance(m, SelfBinarizeConv2d):
                m.is_training = self.is_training
                m.epoch = self.epoch
            elif isinstance(m, SelfBinarizeLinear):
                m.is_training = self.is_training
                m.epoch = self.epoch
            elif isinstance(m, SelfTanh):
                m.is_training = self.is_training
                m.epoch = self.epoch

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.bn2(x)
        # x = self.tanh2(x)
        x = self.fc(x)
        # x = self.bn3(x)
        # x = self.logsoftmax(x)

        return x

class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10, block=BasicBlock, in_dim=3, depth=20):
        super(ResNet_cifar10, self).__init__()
        self.inflate = 1
        self.inplanes = 16*self.inflate
        n = int((depth - 2) / 6)
        self.conv1 = SelfBinarizeConv2d(in_dim, 16*self.inflate, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16*self.inflate)
        self.tanh1 = SelfTanh()
        # self.tanh2 = SelfTanh(inplace=True)
        self.layer1 = self._make_layer(block, 16*self.inflate, n)
        self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2)
        self.layer3 = self._make_layer(block, 64*self.inflate, n, stride=2,do_bntan=False)
        # self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        # self.bn2 = nn.BatchNorm1d(64*self.inflate)
        # self.bn3 = nn.BatchNorm1d(10)
        # self.logsoftmax = nn.LogSoftmax()
        self.fc = SelfBinarizeLinear(64*self.inflate, num_classes)

        init_model(self)
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            # 101: {'lr': 1e-3},
            # 142: {'lr': 5e-4},
            # 184: {'lr': 1e-4},
            # 220: {'lr': 1e-5}
        }

        self.input_transform = {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
                transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #R,G,B每层的归一化用到的均值和方差
                                # normalize
            ]),
            'eval': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # normalize
            ])
        }    

def SelfBinaring(**kwargs):
    datasets = kwargs.get('dataset', 'cifar10')
    if datasets == 'mnist':
        num_classes = 10
        in_dim = 1
    elif datasets == 'cifar10':
        num_classes = 10
        in_dim = 3

    return ResNet_cifar10(num_classes=num_classes, block=BasicBlock, in_dim=in_dim)
