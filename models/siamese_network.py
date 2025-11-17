import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """
    Siamese Neural Network for facial verification and one-shot learning.
    
    Architecture based on Koch et al. (2015):
    - Conv1: 64 filters, 10x10 kernel, ReLU, MaxPool 2x2
    - Conv2: 128 filters, 7x7 kernel, ReLU, MaxPool 2x2
    - Conv3: 128 filters, 4x4 kernel, ReLU, MaxPool 2x2
    - Conv4: 256 filters, 4x4 kernel, ReLU
    - FC: 4096 units, Sigmoid
    - L1 Distance + FC layer for final prediction
    
    Reasoning:
    1. Progressive feature extraction: Filters increase in depth while spatial dimensions decrease
    2. Large initial kernels (10x10) capture global face structure
    3. Smaller kernels (4x4) in later layers capture fine details
    4. Deep architecture (4 conv + 2 FC) learns hierarchical features
    5. L1 distance metric naturally measures similarity between embeddings
    """
    
    def __init__(self, input_channels=3, use_batchnorm=True):
        """
        Initialize the Siamese Network.
        
        Args:
            input_channels: Number of input channels (3 for RGB, 1 for grayscale)
            use_batchnorm: Whether to use batch normalization (improves training stability)
        """
        super(SiameseNetwork, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        
        # Convolutional Block 1: 64 filters, 10x10 kernel
        # Input: (batch, 3, 105, 105) -> Output: (batch, 64, 48, 48) after pooling
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=10, stride=1)
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else None
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2: 128 filters, 7x7 kernel
        # Input: (batch, 64, 48, 48) -> Output: (batch, 128, 21, 21) after pooling
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1)
        self.bn2 = nn.BatchNorm2d(128) if use_batchnorm else None
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 3: 128 filters, 4x4 kernel
        # Input: (batch, 128, 21, 21) -> Output: (batch, 128, 9, 9) after pooling
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batchnorm else None
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 4: 256 filters, 4x4 kernel (no pooling)
        # Input: (batch, 128, 9, 9) -> Output: (batch, 256, 6, 6)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1)
        self.bn4 = nn.BatchNorm2d(256) if use_batchnorm else None
        
        # Fully Connected Layer for feature embedding
        # Flatten: (batch, 256, 6, 6) -> (batch, 9216)
        # FC: (batch, 9216) -> (batch, 4096)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.bn_fc = nn.BatchNorm1d(4096) if use_batchnorm else None
        
        # Final classification layer (L1 distance -> FC -> Sigmoid)
        # Takes L1 distance of two embeddings (4096) and outputs similarity score
        self.fc_out = nn.Linear(4096, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using He initialization for ReLU activations.
        Xavier initialization for sigmoid output layer.
        This is important for deep networks to prevent vanishing/exploding gradients.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Use Xavier for final sigmoid layer, He for others
                if m is self.fc_out:
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward_once(self, x):
        """
        Forward pass through one tower of the Siamese network.
        This extracts features from a single image.
        
        Args:
            x: Input image tensor (batch, channels, height, width)
            
        Returns:
            Feature embedding tensor (batch, 4096)
        """
        # Conv Block 1
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Conv Block 4
        x = self.conv4(x)
        if self.use_batchnorm:
            x = self.bn4(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layer - NO activation here!
        # Paper applies sigmoid only at the END after distance computation
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.bn_fc(x)
        # NO sigmoid here - embeddings should be unbounded
        
        return x
    
    def forward(self, input1, input2):
        """
        Forward pass through the Siamese network.
        Processes both images through shared weights and computes similarity.
        
        Args:
            input1: First image tensor (batch, channels, height, width)
            input2: Second image tensor (batch, channels, height, width)
            
        Returns:
            Similarity score between 0 and 1 (batch, 1)
        """
        # Get embeddings for both images using shared weights
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        # Compute L1 distance between embeddings
        # Paper: p = sigmoid(sum(alpha_j * |h1_j - h2_j|))
        # where alpha is learned weights in fc_out layer
        distance = torch.abs(output1 - output2)
        
        # Pass through final FC layer with sigmoid
        # fc_out learns the weights alpha for each dimension
        # Output is SIMILARITY (high = similar, low = different)
        output = torch.sigmoid(self.fc_out(distance))
        
        return output
    
    def get_embedding(self, x):
        """
        Get the feature embedding for an input image.
        Useful for visualization and analysis.
        
        Args:
            x: Input image tensor (batch, channels, height, width)
            
        Returns:
            Feature embedding (batch, 4096)
        """
        return self.forward_once(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function for Siamese Networks.
    
    Loss = (1 - Y) * 0.5 * D^2 + Y * 0.5 * max(margin - D, 0)^2
    
    Where:
    - Y = 1 if pair is dissimilar, 0 if similar
    - D = distance between embeddings
    - margin = minimum distance for dissimilar pairs
    
    Reasoning:
    - Pulls similar pairs closer (minimize distance)
    - Pushes dissimilar pairs apart (maximize distance up to margin)
    - Margin prevents the network from pushing dissimilar pairs infinitely far
    """
    
    def __init__(self, margin=1.0):
        """
        Initialize Contrastive Loss.
        
        Args:
            margin: Minimum distance for dissimilar pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        """
        Compute contrastive loss.
        
        Args:
            output1: Embedding from first image (batch, embedding_dim)
            output2: Embedding from second image (batch, embedding_dim)
            label: 0 if same person, 1 if different person (batch,)
            
        Returns:
            Loss value
        """
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Contrastive loss
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing Siamese Network Architecture")
    print("=" * 60)
    
    # Create model
    model = SiameseNetwork(input_channels=3, use_batchnorm=True)
    
    # Print model summary
    print(f"\nModel Architecture:")
    print(model)
    
    print(f"\n\nTotal Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    img1 = torch.randn(batch_size, 3, 105, 105)
    img2 = torch.randn(batch_size, 3, 105, 105)
    
    output = model(img1, img2)
    print(f"\nInput shape: {img1.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Test embedding extraction
    embedding = model.get_embedding(img1)
    print(f"\nEmbedding shape: {embedding.shape}")
    
    print("\n" + "=" * 60)
    print("Architecture test completed successfully!")

