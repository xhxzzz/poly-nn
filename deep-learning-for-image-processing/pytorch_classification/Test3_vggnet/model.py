import torch.nn as nn
import torch
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import numpy as np

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

# x = np.arange(-10, 12, 2)
# y = np.maximum(x, 0)
# poly = lagrange(x, y)
# polyCoefficient = Polynomial(poly).coef

# polyCoefficient = [
#     0.0, 
#     0.49999999999999994, 
#     0.0016330266955266947, 
#     -1.5104317985218285e-21, 
#     -2.7533034336419635e-09, 
#     -1.869600345373798e-26, 
#     2.3990207248264032e-15, 
#     -5.257186895069533e-33, 
#     -9.844542811156595e-22, 
#     2.5053871295601977e-39, 
#     1.8396495301046406e-28, 
#     7.3811070233906075e-47, 
#     -1.254306497798623e-35
# ]

polyCoefficient = [
    3.76760224e-08,
    1.28378431e-21, 
    -8.39637070e-06,  
    1.21549228e-19,
    6.41999421e-04,  
    3.63207728e-18, 
    -2.06404321e-02, 
    -1.64798730e-17,
    3.22817460e-01,  
    5.00000000e-01,  
    0.00000000e+00
]

polyCoe = torch.Tensor(polyCoefficient[::-1]).cuda()

def polyRelu(x):
    c = torch.zeros(x.size()).cuda()
    for i in range(polyCoe.view(1,-1).size()[1]):
        c = c + polyCoe[i] * (x**i)
    return c
    
class polyReluLayer(nn.Module):       
    def __init__(self, **kwargs):
        super(polyReluLayer, self).__init__(**kwargs)

    def forward(self, x):
        return polyRelu(x)

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            # polyReluLayer(),
            # nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            # polyReluLayer(),
            # nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

        # self.classifier2 = nn.Sequential(
        #     polyReluLayer(),
        #     # nn.ReLU(True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, 4096),
        # )

        # self.classifier3 = nn.Sequential(
        #     polyReluLayer(),
        #     # nn.ReLU(True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, num_classes)
        # )

        # self.maxBeforeRelu1 = []
        # self.maxBeforeRelu2 = []
        # self.minBeforeRelu1 = []
        # self.minBeforeRelu2 = []

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        # print(f'maximum before poly relu1: {torch.max(x)}')
        # self.maxBeforeRelu1.append(torch.max(x))
        # self.minBeforeRelu1.append(torch.min(x))
        # x = self.classifier2(x)
        # # print(f'maximum before poly relu2: {torch.max(x)}')
        # self.maxBeforeRelu2.append(torch.max(x))
        # self.minBeforeRelu2.append(torch.min(x))
        # x = self.classifier3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # layers += [conv2d, polyReluLayer()]
            layers += [conv2d]
            # layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model
