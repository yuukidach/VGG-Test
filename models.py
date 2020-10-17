'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

class VGG19(nn.Module):
    """ VGG19 modle
    """
    def __init__(self):
        super(VGG19, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Sequential(nn.Linear(512, 4096), nn.ReLU(True), nn.Dropout(),
                                        nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
                                        nn.Linear(4096, 10))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _make_layers(self):
        layers = []
        in_channels = 3
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'] 
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, l, kernel_size=3, padding=1),
                           nn.BatchNorm2d(l),
                           nn.ReLU(inplace=True)]
                in_channels = l
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG34(nn.Module):
    """ VGG34 modle
    """
    def __init__(self):
        super(VGG34, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        layers = []
        # layer 1
        layers += [nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
                   nn.BatchNorm2d(64),
                   nn.ReLU(inplace=True)]
        # pool
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # layer 2-7
        for _ in range(6):
            layers += [nn.Conv2d(64, 64, kernel_size=3, padding=1),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True)]
        # layer 8
        layers += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                   nn.BatchNorm2d(128),
                   nn.ReLU(inplace=True)]
        # layer 9-15
        for _ in range(7):
            layers += [nn.Conv2d(128, 128, kernel_size=3, padding=1),
                       nn.BatchNorm2d(128),
                       nn.ReLU(inplace=True)]
        # layer 16
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                   nn.BatchNorm2d(256),
                   nn.ReLU(inplace=True)]
        # layer 17-27
        for _ in range(11):
            layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1),
                       nn.BatchNorm2d(256),
                       nn.ReLU(inplace=True)]
        # layer 28
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                   nn.BatchNorm2d(512),
                   nn.ReLU(inplace=True)]
        # layer 29 - 33
        for _ in range(5):
            layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1),
                       nn.BatchNorm2d(512),
                       nn.ReLU(inplace=True)]
        # pool
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
