import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchvision
import numpy as np


class MRInet(nn.Module):
    def __init__(self):
        super(MRInet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 32,kernel_size = 3,padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.Flatten = nn.Flatten()
        # self.fc1 = nn.Linear(32*37*37,1024)
        # self.fc2 = nn.Linear(1024,2)
        self.fc = nn.Linear(64*37*37,2)

        #self.initialize_network()
        
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))),2,stride = 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))),2,stride = 2)
        x = self.Flatten(x)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc(x)
        
        return x

    def initialize_network(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MRIVGG16(nn.Module):

    def __init__(self, pretrained=False):
        super(MRIVGG16, self).__init__()
        self.model = torchvision.models.vgg16(pretrained=pretrained)
        if (pretrained):
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier._modules['6'] = nn.Linear(4096, 2)
        nn.init.kaiming_normal_(self.model.classifier._modules['6'].weight)

    def forward(self, x):
        return self.model(x)

            


def build_network(arch):

    if (arch == 'MRInet'):
        return MRInet()

    elif (arch == 'VGG16'):
        return MRIVGG16()

    elif (arch == 'VGG16-pretrained'):
        return MRIVGG16(pretrained = True)

    raise NotImplementedError
