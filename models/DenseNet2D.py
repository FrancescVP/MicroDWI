"""
DenseNet121 Implementation in PyTorch

This script contains a PyTorch implementation of the DenseNet-121 architecture.
The DenseNet architecture is a deep convolutional neural network architecture
known for its densely connected layers. Specifically, each layer receives the feature-maps 
from all preceding layers and passes on its own feature-maps to all subsequent layers.

The architecture comprises the following key components:
1. Initial Convolution and Max Pooling layers
2. Dense Blocks with multiple convolutional layers
3. Transition Layers that reduce dimensionality of the input
4. Classification Layer for output predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Double Convolution Block
    
    A block comprising two consecutive convolutional layers, each followed by 
    batch normalization and ReLU activation.
    
    Attributes:
    conv (nn.Sequential): Sequential list containing the convolutional, batch normalization
                          and activation layers.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        )

    def forward(self, x):
        """Forward pass for the DoubleConv block."""
        return self.conv(x)

class DenseBlock(nn.Module):
    """
    Dense Block
    
    A dense block consisting of multiple double convolutional layers, 
    where each layer receives all preceding feature maps.
    
    Attributes:
    layers (nn.ModuleList): List containing multiple DoubleConv layers.
    """
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            DoubleConv(in_channels + i*growth_rate, growth_rate) 
            for i in range(n_layers)
        ])

    def forward(self, x):
        """Forward pass for the Dense Block."""
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x

class TransitionLayer(nn.Module):
    """
    Transition Layer
    
    A layer that reduces the dimensionality of the input using a 1x1 convolution
    followed by 2x2 average pooling.
    
    Attributes:
    reduce (nn.Conv2d): Convolutional layer for reducing dimensions.
    pool (nn.AvgPool2d): Average pooling layer.
    """
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        """Forward pass for the Transition Layer."""
        x = self.reduce(x)
        x = self.pool(x)
        return x

class ClassLayer(nn.Module):
    """
    Classification Layer
    
    A layer that produces the final output predictions by using global average pooling
    and a fully connected layer.
    
    Attributes:
    pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
    fc (nn.Linear): Fully connected layer for classification.
    """
    def __init__(self, in_channels, num_classes):
        super(ClassLayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        """Forward pass for the Classification Layer."""
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class DenseNet2D(nn.Module):
    """
    DenseNet-121 Architecture
    
    A deep convolutional neural network composed of initial convolutional and max-pooling layers,
    multiple dense blocks, transition layers, and a classification layer for final output.
    """
    
    def __init__(self, in_channels, num_classes, config):
        super(DenseNet2D, self).__init__()
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.init_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.dense1 = DenseBlock(64, 32, config[0])
        self.trans1 = TransitionLayer(64 + 32 * config[0], 128)
        
        self.dense2 = DenseBlock(128, 32, config[1])
        self.trans2 = TransitionLayer(128 + 32 * config[1], 256)
        
        self.dense3 = DenseBlock(256, 32, config[2])
        self.trans3 = TransitionLayer(256 + 32 * config[2], 512)

        self.dense4 = DenseBlock(512, 32, config[3])
        self.trans4 = TransitionLayer(512 + 32 * config[3], 1024)

        self.classlayer = ClassLayer(1024, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.init_pool(x)
        
        x = self.dense1(x)
        x = self.trans1(x)
        
        x = self.dense2(x)
        x = self.trans2(x)

        x = self.dense3(x)
        x = self.trans3(x)

        x = self.dense4(x)
        x = self.trans4(x)
        
        x = self.classlayer(x)
        return x