import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['lenet']

class LeNet(nn.Module):

    def __init__(self, in_dim=1, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_dim, 6, 5),
            # nn.BatchNorm2d(6),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            # nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*4*4, 120),
            # nn.BatchNorm1d(120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            # nn.BatchNorm1d(84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )

        self.regime = {
            0: {
                'optimizer': 'Adam',
                'lr': 1e-2,
            }
        }

    def forward(self, img):
        feature = self.features(img)
        output = self.classifier(feature.view(img.shape[0], -1))
        return output

def lenet(**kwargs):
    datasets = kwargs.get('dataset', 'mnist')
    if datasets == 'mnist':
        num_classes = 10
        in_dim = 1
    elif datasets == 'cifar10':
        num_classes = 10
        in_dim = 3
    return LeNet(in_dim, num_classes)