import torch
import torch.nn as nn
import torch.nn.functional as F


polyCoefficient = [
    0,
    0.500000000000345,
    0.000445951593435763
]

polyCoe = torch.Tensor(polyCoefficient).cuda()

# def polyRelu(x):
#     c = torch.zeros(x.size()).cuda()
#     for i in range(polyCoe.view(1,-1).size()[1]):
#         c = c + polyCoe[i] * (x**i)
#     return c

class polyReluLayer(nn.Module):
    def __init__(self, **kwargs):
        super(polyReluLayer, self).__init__(**kwargs)

    def forward(self, x):
        c = torch.zeros(x.size()).cuda()
        for i in range(polyCoe.view(1,-1).size()[1]):
            c = c + polyCoe[i] * (x**i)
        return c



class polyNet(nn.Module):
    def __init__(self):
        super(polyNet, self).__init__()
        # 3 -> 96
        self.conv1 = nn.Conv2d(
            3, 96, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(96)

        # 96 -> 96
        self.conv2 = nn.Conv2d(
            96, 96, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(
            96, 96, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(96)
        self.relu1 = polyReluLayer()

        # 96 -> 192
        self.conv4 = nn.Conv2d(
            96, 192, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(192)

        # 192 -> 192
        self.conv5 = nn.Conv2d(
            192, 192, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.relu2 = polyReluLayer()
        self.bn5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(
            192, 192, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn6 = nn.BatchNorm2d(192)
        self.conv7 = nn.Conv2d(
            192, 192, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(
            192, 192, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.relu3 = polyReluLayer()
        self.bn8 = nn.BatchNorm2d(192)

        self.classifier = nn.Linear(196608, 10)
        self.avgpool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        out = self.bn1(self.conv1(x))

        out = self.bn2(self.conv2(out))
        out = self.bn3(self.relu1(self.conv3(out)))

        out = self.bn4(self.conv4(out))

        out = self.bn5(self.relu2(self.conv5(out)))

        out = self.bn6(self.conv6(out))
        out = self.bn7(self.conv7(out))
        out = self.bn8(self.relu3(self.conv8(out)))

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        # out = self.avgpool(out)
        out = F.softmax(out)

        return out



        
