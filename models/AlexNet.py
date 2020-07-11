import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['alexnet']

class AlexNet(nn.Module):

    def __init__(self, in_dim=3, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_dim, 96, 11, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes)
        )

        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-2,
                'weight_decay': 5e-4, 'momentum': 0.9},
            10: {'lr': 5e-3},
            15: {'lr': 1e-3, 'weight_decay': 0},
            20: {'lr': 5e-4},
            25: {'lr': 1e-4}
        }

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # normalize = {'mean': [0.5], 'std': [0.5]}
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize
            ]),
            'eval': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize
            ])
        }    

    def forward(self, img):
        feature = self.features(img)
        output = self.classifier(feature.view(img.shape[0], -1))
        return output

def alexnet(**kwargs):
    datasets = kwargs.get('dataset', 'mnist')
    if datasets == 'mnist':
        num_classes = 10
        in_dim = 1
    elif datasets == 'cifar10':
        num_classes = 10
        in_dim = 3
    return AlexNet(in_dim, num_classes)