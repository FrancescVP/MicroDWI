import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    """
    Implements the bottleneck layer containing two consecutive 3D convolutional layers,
    each followed by batch normalization and ReLU activation as described in the original 
    DenseNet paper.
    
    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


class DenseBlock3D(nn.Module):
    """
    Implements a 3D dense block as described in the original DenseNet paper.
    
    Args:
        in_channels: number of input channels.
        growth_rate: how many filters to add each layer (k in paper).
        n_layers: number of DoubleConv3D layers in the block.
    """
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList([
            DoubleConv3D(in_channels + i * growth_rate, growth_rate)
            for i in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x


class TransitionLayer3D(nn.Module):
    """
    Implements a 3D transition layer as described in the original DenseNet paper.
    
    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer3D, self).__init__()
        self.reduce = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool3d(2, stride=2)

    def forward(self, x):
        x = self.reduce(x)
        x = self.pool(x)
        return x


class ClassLayer3D(nn.Module):
    """
    Implements the final 3D classification layer of the DenseNet architecture.
    
    Args:
        in_channels: number of input channels.
        num_classes: number of output classes.
    """
    def __init__(self, in_channels, num_classes):
        super(ClassLayer3D, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DenseNet3D(nn.Module):
    """
    Implements the 3D version of the DenseNet-121 architecture as described in the original paper.
    
    Args:
        in_channels: number of input channels.
        num_classes: number of output classes.
    """
    def __init__(self, in_channels, num_classes, config):
        super(DenseNet3D, self).__init__()
        self.init_conv = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.init_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.dense1 = DenseBlock3D(64, 32, config[0])
        self.trans1 = TransitionLayer3D(64 + 32 * config[0], 128)
        
        self.dense2 = DenseBlock3D(128, 32, config[1])
        self.trans2 = TransitionLayer3D(128 + 32 * config[1], 256)
        
        self.dense3 = DenseBlock3D(256, 32, config[2])
        self.trans3 = TransitionLayer3D(256 + 32 * config[2], 512)

        self.dense4 = DenseBlock3D(512, 32, config[3])
        self.trans4 = TransitionLayer3D(512 + 32 * config[3], 1024)

        self.classlayer = ClassLayer3D(1024, num_classes)

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
    
def DenseNet121(num_classes=4, channels=3):
    DenseNet3D(channels, num_classes, config = [6, 12, 24, 16])


def DenseNet169(num_classes=4, channels=3):
    DenseNet3D(channels, num_classes, config = [6, 12, 32, 32])


def DenseNet201(num_classes=4, channels=3):
    DenseNet3D(channels, num_classes, config = [6, 12, 48, 32])


def DenseNet264(num_classes=4, channels=3):
    DenseNet3D(channels, num_classes, config = [6, 12, 64, 48])