import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with optional SE attention."""
    
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        out += identity
        out = F.relu(out)
        
        return out


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (from MobileNet)."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride, 
                                   padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class FaceVerificationNet(nn.Module):
    """
    Lightweight version for faster training on limited data.
    
    Uses fewer blocks and channels, plus depthwise separable convolutions.
    """
    
    def __init__(self, embedding_dim=128, dropout=0.3):
        super(FaceVerificationNet, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Efficient blocks using depthwise separable convolutions
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            ResidualBlock(64, 64, stride=1, use_se=True)
        )
        
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1, use_se=True)
        )
        
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1, use_se=True)
        )
        
        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1, use_se=True)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Embedding layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, embedding_dim)
        self.bn_fc = nn.BatchNorm1d(embedding_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_once(self, x):
        """Extract embedding for a single image."""
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Efficient blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Embedding
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_fc(x)
        
        # L2 normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def forward(self, input1, input2):
        """
        Forward pass for Siamese network.
        Returns embeddings for both inputs.
        """
        embedding1 = self.forward_once(input1)
        embedding2 = self.forward_once(input2)
        
        return embedding1, embedding2


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)