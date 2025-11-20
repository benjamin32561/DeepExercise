import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetV2(nn.Module):
    """
    LIGHTWEIGHT Siamese Network V2 - For small datasets!
    
    Philosophy: MUCH smaller network that can actually learn from 1870 training pairs
    
    Architecture (REDUCED channels and FC size):
    - Conv1: 32 filters (was 64), 10x10 kernel, ReLU, MaxPool 2x2
    - Conv2: 64 filters (was 128), 7x7 kernel, ReLU, MaxPool 2x2
    - Conv3: 64 filters (was 128), 4x4 kernel, ReLU, MaxPool 2x2
    - Conv4: 128 filters (was 256), 4x4 kernel, ReLU
    - FC: 512 units (was 4096!) + Dropout
    - L1 Distance + FC layer for final prediction
    
    Parameters: ~2M (was 38M!) - 95% reduction!
    """
    
    def __init__(self, input_channels=3, use_batchnorm=True, fc_dropout=0.3):
        """
        Initialize the LIGHTWEIGHT Siamese Network V2.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            use_batchnorm: Whether to use batch normalization
            fc_dropout: Dropout rate for FC layer (0.3 recommended for small datasets)
        """
        super(SiameseNetV2, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        self.fc_dropout_rate = fc_dropout
        
        # Convolutional Block 1: 32 filters (HALVED), 10x10 kernel
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=10, stride=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else None
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2: 64 filters (HALVED), 7x7 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else None
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 3: 64 filters (HALVED), 4x4 kernel
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1)
        self.bn3 = nn.BatchNorm2d(64) if use_batchnorm else None
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 4: 128 filters (HALVED), 4x4 kernel (no pooling)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        self.bn4 = nn.BatchNorm2d(128) if use_batchnorm else None
        
        # Fully Connected Layer - DRASTICALLY REDUCED (8x smaller!)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)  # Was 256*6*6 -> 4096, now 128*6*6 -> 512
        self.bn_fc = nn.BatchNorm1d(512) if use_batchnorm else None
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout > 0 else None
        
        # Final classification layer
        self.fc_out = nn.Linear(512, 1)  # Was 4096 -> 1
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He initialization for ReLU, Xavier for sigmoid output."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m is self.fc_out:
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward_once(self, x):
        """Forward pass through one tower - extracts features from single image."""
        # Conv block 1
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        if self.bn3 is not None:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Conv block 4
        x = self.conv4(x)
        if self.bn4 is not None:
            x = self.bn4(x)
        x = F.relu(x)
        
        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.bn_fc is not None:
            x = self.bn_fc(x)
        # NO ReLU here - keep embeddings unbounded like original
        
        # Apply dropout (only during training)
        if self.fc_dropout is not None:
            x = self.fc_dropout(x)
        
        return x
    
    def forward(self, input1, input2):
        """
        Forward pass through both towers + similarity prediction.
        For BCE/Focal loss (outputs similarity score 0-1).
        """
        # Get embeddings from both towers
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        # Compute L1 distance
        l1_distance = torch.abs(output1 - output2)
        
        # Predict similarity score
        similarity = self.fc_out(l1_distance)
        similarity = torch.sigmoid(similarity)
        
        return similarity
    
    def get_embeddings(self, input1, input2):
        """
        Get raw embeddings for both inputs (for embedding-based losses).
        For contrastive/cosine/triplet loss.
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# Dummy parameters for compatibility with old code
# (these are ignored in the simplified version)
class DummyModule:
    pass

SqueezeExcitation = DummyModule
ConvBlock = DummyModule
ResidualBlock = DummyModule
DepthwiseSeparableConv = DummyModule
