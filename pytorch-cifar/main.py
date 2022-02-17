'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

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
        f.write(f'maximum: {torch.max(t)}\n')
        f.write(f'minimum: {torch.min(t)}\n')
        f.write(f'avg: {torch.mean(t)}\n')
        f.write(f'-----------------------------\n')


polyCoefficient = [
    0.0, 
    0.5000000000000001,
    0.4229738360571231,
    1.0209560586549857e-15,
    -0.05181217276513935,
    1.922278905409424e-16,
    0.003755139551398267,
    -2.9976358510402246e-17,
    -0.00015542914766067772,
    -1.2613715307256731e-17,
    3.9668640059761604e-06,
    -8.244772775241301e-19,
    -6.611937843267827e-08,
    -1.8140098300635756e-20,
    7.482664585831936e-10,
    -1.878757911502137e-22,
    -5.893442474031118e-12,
    -1.3102757644730043e-24,
    3.27356832474586e-14,
    -6.020909032444876e-27,
    -1.285688357993083e-16,
    -1.5619668656266033e-29,
    3.5376371913616156e-19,
    -1.9265767708798748e-32,
    -6.650853631290366e-22,
    -7.58788385670289e-36,
    8.116836166593117e-25,
    -2.863459015785196e-40,
    -5.781502656158069e-28,
    2.0850910371944984e-44,
    1.8204742745683492e-31
]

polyFlag = False
polyCoe = torch.Tensor(polyCoefficient).cuda()

# def polyRelu(x):
#     c = torch.zeros(x.size()).cuda()
#     for i in range(polyCoe.view(1,-1).size()[1]):
#         c = c + polyCoe[i] * (x**i)
#     return c

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, polyFlag=False):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes)
#         self.conv4 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn4 = nn.BatchNorm2d(planes)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#         self.polyFlag = polyFlag

#     def forward(self, x):
#         #out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn1(self.conv1(x))
#         # TODO: count out
#         if not polyFlag:
#             count(out, 'before first relu in basic block in train')
#         else:
#             count(out, 'before first relu in basic block in test')
#         if not polyFlag:
#             out = F.relu(out)
#         else:
#             out = polyRelu(out)
#         # out = polyRelu(out)
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         # TODO: count out
#         if not polyFlag:
#             count(out, 'before second relu in basic block in train')
#         else:
#             count(out, 'before second relu in basic block in test')
#         if not polyFlag:
#             out = F.relu(out)
#         else:
#             out = polyRelu(out)

#         # prod = cz * self.shortcut(x)
#         # prod = self.bn3(self.conv3(prod))
#         # prod2 = cz * self.shortcut(x) * self.shortcut(x)
#         # prod2 = self.bn4(self.conv4(prod2))

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion *
#                                planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = self.bn1(self.conv1(x))
#         count(out, 'before first relu in bottleneck')
#         out = F.relu(out)

#         out = self.bn2(self.conv2(out))
#         count(out, 'before second relu in bottleneck')
#         out = F.relu(out)

#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         count(out, 'before third relu in bottleneck')
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#         self.polyFlag = False

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         # out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn1(self.conv1(x))
#         # TODO: count out
#         if not polyFlag:
#             count(out, 'before relu in resnet in train')
#         else:
#             count(out, 'before relu in resnet in test')
#         if not polyFlag:
#             out = F.relu(out)
#         else:
#             out = polyRelu(out)
#         # out = polyRelu(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         # out = F.softmax(out)
#         return out


# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])


# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])


# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = ResNet50()
# net = ResNet34()
# net = ResNet101()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = polyNet()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        #print(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    polyFlag = True
    test(epoch)
    polyFlag = False
    scheduler.step()
