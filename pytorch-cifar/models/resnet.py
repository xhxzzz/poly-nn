'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def count(t, label):
    ge10 = torch.sum(torch.ge(t, 10))  # >= 10
    ge5 = torch.sum(torch.ge(t, 5))    # >= 5
    ge0 = torch.sum(torch.ge(t, 0))    # >= 0
    ge_5 = torch.sum(torch.ge(t, -5))  # >= -5
    ge_10 = torch.sum(torch.ge(t, -10))# >= -10
    le_10 = torch.sum(torch.lt(t, -10))# < -10
    with open('resnet18-stat.txt', 'a') as f:
        f.write(f'{label}\n')
        f.write(f'>= 10: {ge10}\n')
        f.write(f'5 ~ 10: {ge5 - ge10}\n')
        f.write(f'0 ~ 5: {ge0 - ge5}\n')
        f.write(f'-5 ~ 0: {ge_5 - ge0}\n')
        f.write(f'-10 ~ -5: {ge_10 - ge_5}\n')
        f.write(f'<= -10: {le_10}\n')
        f.write(f'-----------------------------\n')


# polyCoefficient = [
#     0.0, 
#     0.5000000000000001,
#     0.4229738360571231,
#     1.0209560586549857e-15,
#     -0.05181217276513935,
#     1.922278905409424e-16,
#     0.003755139551398267,
#     -2.9976358510402246e-17,
#     -0.00015542914766067772,
#     -1.2613715307256731e-17,
#     3.9668640059761604e-06,
#     -8.244772775241301e-19,
#     -6.611937843267827e-08,
#     -1.8140098300635756e-20,
#     7.482664585831936e-10,
#     -1.878757911502137e-22,
#     -5.893442474031118e-12,
#     -1.3102757644730043e-24,
#     3.27356832474586e-14,
#     -6.020909032444876e-27,
#     -1.285688357993083e-16,
#     -1.5619668656266033e-29,
#     3.5376371913616156e-19,
#     -1.9265767708798748e-32,
#     -6.650853631290366e-22,
#     -7.58788385670289e-36,
#     8.116836166593117e-25,
#     -2.863459015785196e-40,
#     -5.781502656158069e-28,
#     2.0850910371944984e-44,
#     1.8204742745683492e-31
# ]

# polyCoefficient = [
#     52,
#     0.5,
#     0.0012
# ]

polyCoefficient = [
    0,
    0.500000000000345,
    0.000445951593435763
]

polyCoe = torch.Tensor(polyCoefficient).cuda()

# def polyReluFunc(x):
#     c = torch.zeros(x.size()).cuda()
#     for i in range(polyCoe.view(1,-1).size()[1]):
#         c = c + polyCoe[i] * (x**i)
#     return c
    
class polyRelu(nn.Module):       
    def __init__(self, **kwargs):
        super(polyRelu, self).__init__(**kwargs)

    def polyReluFunc(self, x):
        c = torch.zeros(x.size()).cuda()
        for i in range(polyCoe.view(1,-1).size()[1]):
            c = c + polyCoe[i] * (x**i)
        return c

    def forward(self, x):
        return self.polyReluFunc(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.reluLayer = polyRelu()

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(x))
        # TODO: count out
        # count(out, 'before first relu in basic block')
        # out = F.relu(out)
        out = self.reluLayer(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # TODO: count out
        # count(out, 'before second relu in basic block')
        # out = F.relu(out)
        out = self.reluLayer(out)

        # prod = cz * self.shortcut(x)
        # prod = self.bn3(self.conv3(prod))
        # prod2 = cz * self.shortcut(x) * self.shortcut(x)
        # prod2 = self.bn4(self.conv4(prod2))

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # count(out, 'before first relu in bottleneck')
        out = F.relu(out)

        out = self.bn2(self.conv2(out))
        # count(out, 'before second relu in bottleneck')
        out = F.relu(out)

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        # count(out, 'before third relu in bottleneck')
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.reluLayer = polyRelu()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(x))
        # TODO: count out
        # count(out, 'before relu in resnet')
        # out = F.relu(out)
        out = self.reluLayer(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # out = F.softmax(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
