import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as ap

from torch.autograd import Function

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha=1): #changes the direction of the gradient
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = -grad_output * ctx.alpha
        return output, None

def gradient_reversal_layer(x, alpha=0):
    return GradientReversalFn.apply(x, alpha)


# Network Definitions



class DomainClassifier(nn.Module):
    def __init__(self, length_of_features):
        super(DomainClassifier, self).__init__()
        self.max_pool = nn.MaxPool2d(4, stride=4)
        self.dmoain_fc = nn.Sequential(
            nn.Linear(length_of_features, 100), #we could potentially reduce the size by adding max pool
            nn.ReLU(),
            nn.Linear(100, 2)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.max_pool(x)
        #print(x.shape)
        #exit()
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)
        #exit()
        return self.fc(x)

# Define Downsample Blocks

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DownsampleBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(conv1x1(in_channels, out_channels, stride), nn.BatchNorm2d(out_channels))
        self.stride = stride

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

# Define Upsample Blocks

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv_block = DownsampleBlock(in_channels+out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv_block(x)
        return x


# Build the ResUNet Model

class ResUNet(nn.Module):
    def __init__(self, n_class):
        super(ResUNet, self).__init__()
        self.base_model = models.resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # Size: (/2, /2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # Size: (/4, /4)
        self.layer2 = self.base_layers[5]  # Size: (/8, /8)
        self.layer3 = self.base_layers[6]  # Size: (/16, /16)
        self.layer4 = self.base_layers[7]  # Size: (/32, /32)

        self.up4 = UpsampleBlock(512, 256)
        self.up3 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.up1 = UpsampleBlock(64, 64)
        self.outc = nn.Conv2d(64, n_class, 1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.dmoain_fc = nn.Sequential(
            nn.Linear(512*2*2, 2024), # this one was for !!128x128, for 256x256 it should be 512*4*4, for !!512x512 it should be 512*8*8 (see here) 
            nn.ReLU(),
            nn.Linear(2024, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x,alpha=0.0, domain=False):
        x0 = self.layer0(x)  # Size: (/2, /2)
        x1 = self.layer1(x0)  # Size: (/4, /4)
        x2 = self.layer2(x1)  # Size: (/8, /8)
        x3 = self.layer3(x2)  # Size: (/16, /16)
        x4 = self.layer4(x3)  # Size: (/32, /32)

        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)

        x = self.outc(x)
        if (domain):
            x4 = self.max_pool(x4)
            x4 = x4.view(x4.size(0), -1)
            x4 = gradient_reversal_layer(x4, alpha)
            x4 = self.dmoain_fc(x4)
            return x, x4
        
        return x


model = ResUNet(n_class=1)
#output, features = model(torch.rand([1,3,256,256]), domain=True)
#print(features.shape)

#reverse = gradient_reversal_layer(features)
#domain_classifier = DomainClassifier(features.shape[1]*features.shape[2]*features.shape[3])
#domain_classifier(reverse)
#optimizer = optim.Adam(list(model.parameters())  + list(domain_classifier.parameters()), lr=0.001)
#loss = domain_classifier(reverse).sum()
#loss.backward()
#optimizer.step()