import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BackboneNetwork(nn.Module):
    """
    Face verification network using pretrained backbone.
    
    Supports various pretrained backbones from torchvision:
    - ResNet18, ResNet34, ResNet50
    - MobileNetV3 (Small, Large)
    - EfficientNet-B0, B1, B2
    
    Args:
        backbone: Backbone architecture name (default: 'resnet18')
        embedding_dim: Dimension of face embedding (default: 128)
        pretrained: Use pretrained ImageNet weights (default: True)
        dropout: Dropout rate (default: 0.3)
    """
    
    def __init__(self, backbone='resnet18', embedding_dim=128, pretrained=True, dropout=0.3):
        super(BackboneNetwork, self).__init__()
        
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        
        # Load backbone
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            feature_dim = 512
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            feature_dim = 512
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        elif backbone == 'mobilenet_v3_small':
            base_model = models.mobilenet_v3_small(pretrained=pretrained)
            feature_dim = 576
            self.backbone = base_model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
        
        elif backbone == 'mobilenet_v3_large':
            base_model = models.mobilenet_v3_large(pretrained=pretrained)
            feature_dim = 960
            self.backbone = base_model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
        
        elif backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
            self.backbone = base_model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Embedding head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        
        # Initialize embedding head
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc.bias, 0)
    
    def forward_once(self, x):
        """Extract embedding for a single image."""
        # Extract features using backbone
        x = self.backbone(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Embedding
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        
        # L2 normalize
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


class BackboneNetworkWithClassifier(nn.Module):
    """
    Backbone network with classifier head (for BCE loss).
    
    This version outputs similarity scores instead of embeddings.
    """
    
    def __init__(self, backbone='resnet18', embedding_dim=128, pretrained=True, dropout=0.3):
        super(BackboneNetworkWithClassifier, self).__init__()
        
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        
        # Load backbone
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            feature_dim = 512
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            feature_dim = 512
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        elif backbone == 'mobilenet_v3_small':
            base_model = models.mobilenet_v3_small(pretrained=pretrained)
            feature_dim = 576
            self.backbone = base_model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
        
        elif backbone == 'mobilenet_v3_large':
            base_model = models.mobilenet_v3_large(pretrained=pretrained)
            feature_dim = 960
            self.backbone = base_model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
        
        elif backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
            self.backbone = base_model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Embedding head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(feature_dim, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        
        # Classifier head
        self.fc_out = nn.Linear(embedding_dim, 1)
        
        # Initialize
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_normal_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)
    
    def forward_once(self, x):
        """Extract embedding for a single image."""
        # Extract features using backbone
        x = self.backbone(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Embedding
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn(x)
        
        return x
    
    def forward(self, input1, input2):
        """
        Forward pass for Siamese network.
        Returns similarity score.
        """
        embedding1 = self.forward_once(input1)
        embedding2 = self.forward_once(input2)
        
        # L1 distance
        distance = torch.abs(embedding1 - embedding2)
        
        # Classifier
        output = torch.sigmoid(self.fc_out(distance))
        
        return output