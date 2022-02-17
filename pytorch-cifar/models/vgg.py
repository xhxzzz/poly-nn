'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

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
    0.500000000000111,
    0,
    0.0263068516521283
]

polyCoe = torch.Tensor(polyCoefficient).cuda()

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
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x), 
                        #    nn.ReLU(True)]
                           polyReluLayer()]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
