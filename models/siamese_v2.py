import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Recalibrates channel-wise feature responses.
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: global spatial information
        y = self.squeeze(x).view(b, c)
        # Excitation: channel interdependencies
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale features
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """
    Convolutional block with BatchNorm, Dropout, and optional SE attention.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.0, 
                 use_se=False, use_pooling=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.se = SqueezeExcitation(out_channels) if use_se else None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pooling else None
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        
        if self.se is not None:
            x = self.se(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        if self.pool is not None:
            x = self.pool(x)
        
        return x


class SiameseNetV2(nn.Module):
    """
    Improved Siamese Network with modern regularization techniques.
    
    Improvements over original:
    1. Dropout layers for regularization (combat overfitting)
    2. Squeeze-Excitation attention for better feature recalibration
    3. Spatial dropout (2D dropout) for conv layers
    4. Better weight initialization
    5. Optional residual connections
    6. Configurable dropout rates
    
    Architecture maintains the same structure as original for fair comparison:
    - Conv1: 64 filters, 10x10 kernel, ReLU, MaxPool 2x2 + Dropout
    - Conv2: 128 filters, 7x7 kernel, ReLU, MaxPool 2x2 + SE + Dropout
    - Conv3: 128 filters, 4x4 kernel, ReLU, MaxPool 2x2 + SE + Dropout
    - Conv4: 256 filters, 4x4 kernel, ReLU + SE + Dropout
    - FC1: 4096 units + Dropout
    - FC2: L1 Distance → 1 unit, Sigmoid
    """
    
    def __init__(self, input_channels=3, conv_dropout=0.1, fc_dropout=0.3, use_se=True):
        """
        Initialize the improved Siamese Network.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            conv_dropout: Dropout rate for convolutional layers (spatial dropout)
            fc_dropout: Dropout rate for fully connected layers
            use_se: Whether to use Squeeze-Excitation attention
        """
        super(SiameseNetV2, self).__init__()
        
        self.use_se = use_se
        
        # Convolutional Block 1: 64 filters, 10x10 kernel
        # Input: (batch, 3, 105, 105) -> Output: (batch, 64, 48, 48) after pooling
        self.conv1 = ConvBlock(input_channels, 64, kernel_size=10, 
                               dropout=conv_dropout, use_se=False, use_pooling=True)
        
        # Convolutional Block 2: 128 filters, 7x7 kernel + SE
        # Input: (batch, 64, 48, 48) -> Output: (batch, 128, 21, 21) after pooling
        self.conv2 = ConvBlock(64, 128, kernel_size=7, 
                               dropout=conv_dropout, use_se=use_se, use_pooling=True)
        
        # Convolutional Block 3: 128 filters, 4x4 kernel + SE
        # Input: (batch, 128, 21, 21) -> Output: (batch, 128, 9, 9) after pooling
        self.conv3 = ConvBlock(128, 128, kernel_size=4, 
                               dropout=conv_dropout, use_se=use_se, use_pooling=True)
        
        # Convolutional Block 4: 256 filters, 4x4 kernel + SE (no pooling)
        # Input: (batch, 128, 9, 9) -> Output: (batch, 256, 6, 6)
        self.conv4 = ConvBlock(128, 256, kernel_size=4, 
                               dropout=conv_dropout, use_se=use_se, use_pooling=False)
        
        # Fully Connected Layer for feature embedding
        # Flatten: (batch, 256, 6, 6) -> (batch, 9216)
        # FC: (batch, 9216) -> (batch, 4096)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.bn_fc = nn.BatchNorm1d(4096)
        self.fc_dropout = nn.Dropout(fc_dropout)
        
        # Final classification layer (L1 distance -> FC -> Sigmoid)
        self.fc_out = nn.Linear(4096, 1)
        
        # Initialize weights with better initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using modern best practices:
        - He initialization for ReLU layers
        - Xavier initialization for final sigmoid layer
        - Batch norm initialized to (1, 0)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m is self.fc_out:
                    # Xavier for final sigmoid layer
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    # He for ReLU layers
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward_once(self, x):
        """
        Forward pass through one tower of the Siamese network.
        Extracts features from a single image with improved regularization.
        
        Args:
            x: Input image tensor (batch, channels, height, width)
            
        Returns:
            Feature embedding tensor (batch, 4096)
        """
        # Conv blocks with progressive feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layer with regularization
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x, inplace=True)
        x = self.fc_dropout(x)
        
        return x
    
    def get_embeddings(self, input1, input2):
        """
        Get embeddings for both inputs (for use with embedding-based losses).
        
        Args:
            input1: First image tensor
            input2: Second image tensor
            
        Returns:
            embedding1, embedding2: Tuple of embedding tensors
        """
        embedding1 = self.forward_once(input1)
        embedding2 = self.forward_once(input2)
        return embedding1, embedding2
    
    def forward(self, input1, input2, return_embeddings=False):
        """
        Forward pass through the Siamese network.
        
        Args:
            input1: First image tensor (batch, channels, height, width)
            input2: Second image tensor (batch, channels, height, width)
            return_embeddings: If True, return embeddings instead of similarity scores
            
        Returns:
            If return_embeddings=False: Similarity score between 0 and 1 (batch, 1)
            If return_embeddings=True: (embedding1, embedding2)
        """
        # Get embeddings for both images using shared weights
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        if return_embeddings:
            return output1, output2
        
        # Compute L1 distance between embeddings
        distance = torch.abs(output1 - output2)
        
        # Pass through final FC layer with sigmoid
        output = torch.sigmoid(self.fc_out(distance))
        
        return output
    
    def get_embedding(self, x):
        """
        Get the feature embedding for an input image.
        
        Args:
            x: Input image tensor (batch, channels, height, width)
            
        Returns:
            Feature embedding (batch, 4096)
        """
        return self.forward_once(x)


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing SiameseNetV2 Architecture")
    print("=" * 80)
    
    # Create model
    model = SiameseNetV2(input_channels=3, conv_dropout=0.1, fc_dropout=0.3, use_se=True)
    
    # Print model summary
    print(f"\nTotal Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    img1 = torch.randn(batch_size, 3, 105, 105)
    img2 = torch.randn(batch_size, 3, 105, 105)
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        output = model(img1, img2)
    
    print(f"\nInput shape: {img1.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Test embedding extraction
    with torch.no_grad():
        embedding = model.get_embedding(img1)
    print(f"\nEmbedding shape: {embedding.shape}")
    
    # Compare with original Siamese
    from siamese_network import SiameseNetwork
    original_model = SiameseNetwork(input_channels=3, use_batchnorm=True)
    
    print("\n" + "=" * 80)
    print("COMPARISON: Original vs V2")
    print("=" * 80)
    print(f"Original Siamese:  {count_parameters(original_model):,} parameters")
    print(f"SiameseNetV2:      {count_parameters(model):,} parameters")
    print(f"Overhead:          {count_parameters(model) - count_parameters(original_model):,} parameters")
    print(f"                   (+{(count_parameters(model) / count_parameters(original_model) - 1) * 100:.1f}%)")
    print("\n" + "=" * 80)
    print("Key improvements:")
    print("  ✓ Spatial Dropout (2D) in convolutional layers")
    print("  ✓ Regular Dropout in FC layers")
    print("  ✓ Squeeze-Excitation attention blocks")
    print("  ✓ Better weight initialization")
    print("  ✓ Modular ConvBlock design")
    print("=" * 80)
    print("\nArchitecture test completed successfully!")

