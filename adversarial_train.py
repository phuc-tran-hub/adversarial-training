import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
#import resnet
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np

##Defining basic blocks for model

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

##Defining ResNet class

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

##Defining a model of ResNet class

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


##Initializing parameters

learning_rate = 0.1
epsilon = 0.0314
k = 7
alpha = 0.00784
file_name = 'adversarial_training'
mixup_alpha = 1.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

##Training and Test set Transformations
#************************* Write your code here *********************

transform_train = transforms.Compose([transforms.RandomCrop(32, padding = 4),
                                            transforms.RandomHorizontalFlip(p = 0.5),
                                            transforms.ToTensor()])

transform_test = transforms.Compose([transforms.ToTensor()])

##************************* Your code ends here***********************
#Download dataset and splits 

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

## Create your train and test loaders using DataLoader, with specified parameters
#************************* Write your code here *********************

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers = 4, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, num_workers = 4, shuffle=True)
##************************* Your code ends here***********************

##Generate mixed inputs with pairs of targets

def mixup_data(x, y):
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    batch_size = x.size()[0]
    #index = torch.randperm(batch_size).cuda()
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

##Generate adversarial examples using Projected Gradient Descent (PGD) Attack

class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

##Defining perturb Function

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        
        ##Add epsilon to x 
#************************* Write your code here *********************
        x = x + epsilon 
##************************* Your code ends here***********************

        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]

            ##Detach x and add with alpha * torch.sign(grad.detach())
            ##take min(max(x, x_natural-epsilon), x_natural+epsilon)
            ##clamp x between 0,1

#************************* Write your code here *********************

            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
            
##************************* Your code ends here***********************

        return x

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

adversary = LinfPGDAttack(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

def train(epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        optimizer.zero_grad()

        benign_inputs, benign_targets_a, benign_targets_b, benign_lam = mixup_data(inputs, targets)
        benign_outputs = net(benign_inputs)
        loss1 = mixup_criterion(criterion, benign_outputs, benign_targets_a, benign_targets_b, benign_lam)
        benign_loss += loss1.item()

        _, predicted = benign_outputs.max(1)
        benign_correct += (benign_lam * predicted.eq(benign_targets_a).sum().float() + (1 - benign_lam) * predicted.eq(benign_targets_b).sum().float())
        if batch_idx % 10 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current benign train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current benign train loss:', loss1.item())

        ##Adversarial example generation and training
        ##call perturbation function to obtain perturbed sets
        ##use mixup_data to get mixed inputs, pairs of targets
        ##obtain adv_outputs
        ##compute loss2 using criterion
        ##assign loss2 as adv_loss
#************************* Write your code here *********************
        adv = adversary.perturb(inputs, targets)
        adv_inputs, adv_targets_a, adv_targets_b, adv_lam = mixup_data(adv, targets)
        adv_outputs = net(adv_inputs)
        loss2 = mixup_criterion(criterion, adv_outputs, adv_targets_a, adv_targets_b, adv_lam)
        adv_loss += loss2.item()
##************************* Your code ends here***********************
        _, predicted = adv_outputs.max(1)
        adv_correct += (adv_lam * predicted.eq(adv_targets_a).sum().float() + (1 - adv_lam) * predicted.eq(adv_targets_b).sum().float())
        if batch_idx % 10 == 0:
                print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current adversarial train loss:', loss2.item())

        loss = (loss1 + loss2) / 2
        loss.backward()
        optimizer.step()

    print('\nTotal benign train accuracy:', 100. * benign_correct / total)
    print('Total adversarial train accuracy:', 100. * adv_correct / total)
    print('Total benign train loss:', benign_loss)
    print('Total adversarial train loss:', adv_loss)

def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current benign test loss:', loss.item())
            
            ##On test set get predictions
            ##call perturbation function to obtain perturbed sets
            ##obtain adv_outputs
            ##compute loss using criterion
            ##assign loss as adv_loss
#************************* Write your code here *********************
            adv = adversary.perturb(inputs, targets)
            adv_outputs = net(adv)
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.item()
##************************* Your code ends here***********************

            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current adversarial test loss:', loss.item())

    print('\nTotal benign test accuracy:', 100. * benign_correct / total)
    print('Total adversarial test Accuracy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

    state = {
        'net': net.state_dict()
    }

    ##Save the state checkpoints with provided file_name

    #************************* Write your code here *********************
    torch.save(state, file_name)
    ##************************* Your code ends here***********************
    print('Model Saved!')

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(0, 25):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test(epoch)