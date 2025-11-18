import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEFromEmbeddings(nn.Module):
    """
    BCE Loss for models that output embeddings.
    Converts distance to similarity score.
    """
    
    def __init__(self, distance_threshold=1.0):
        super(BCEFromEmbeddings, self).__init__()
        self.distance_threshold = distance_threshold
        self.bce = nn.BCELoss()
    
    def forward(self, embedding1, embedding2, label):
        """
        Args:
            embedding1: First embedding (batch_size, embedding_dim)
            embedding2: Second embedding (batch_size, embedding_dim)
            label: Labels (batch_size,) - 1 for same person, 0 for different
        
        Returns:
            loss: Scalar loss value
        """
        # Calculate distance
        distance = torch.norm(embedding1 - embedding2, p=2, dim=1)
        
        # Convert distance to similarity score (0-1)
        # Using sigmoid transformation: similarity = sigmoid(-distance + threshold)
        similarity = torch.sigmoid(-distance + self.distance_threshold)
        
        # BCE loss
        label_expanded = label.unsqueeze(1) if label.dim() == 1 else label
        loss = self.bce(similarity.unsqueeze(1), label_expanded)
        
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.
    
    For similar pairs (label=1): minimize distance
    For dissimilar pairs (label=0): maximize distance (up to margin)
    
    Automatically normalizes embeddings to handle high-dimensional vectors.
    
    Args:
        margin: Minimum distance for dissimilar pairs (default: 2.0)
        normalize: Whether to L2-normalize embeddings (default: True)
    """
    
    def __init__(self, margin=2.0, normalize=True):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.normalize = normalize
    
    def forward(self, embedding1, embedding2, label):
        """
        Args:
            embedding1: First embedding (batch_size, embedding_dim)
            embedding2: Second embedding (batch_size, embedding_dim)
            label: Labels (batch_size,) - 1 for same person, 0 for different
        
        Returns:
            loss: Scalar loss value
        """
        # Normalize embeddings to unit vectors (helps with high-dimensional embeddings)
        if self.normalize:
            embedding1 = F.normalize(embedding1, p=2, dim=1)
            embedding2 = F.normalize(embedding2, p=2, dim=1)
        
        # Calculate Euclidean distance
        euclidean_distance = F.pairwise_distance(embedding1, embedding2, p=2)
        
        # Contrastive loss
        # label = 1 (same): minimize distance
        # label = 0 (different): maximize distance up to margin
        loss_positive = label * torch.pow(euclidean_distance, 2)
        loss_negative = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        loss = torch.mean(0.5 * (loss_positive + loss_negative))
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Works with score outputs (like BCE but with focusing parameter).
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (batch_size, 1) - probabilities [0, 1]
            target: Labels (batch_size, 1) - 0 or 1
        
        Returns:
            loss: Scalar loss value
        """
        # Ensure correct shape
        if target.dim() == 1:
            target = target.unsqueeze(1)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        
        # Focal loss formula
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class CosineEmbeddingLoss(nn.Module):
    """
    Cosine Embedding Loss - uses cosine similarity instead of Euclidean distance.
    """
    
    def __init__(self, margin=0.5):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        """
        Args:
            embedding1: First embedding (batch_size, embedding_dim)
            embedding2: Second embedding (batch_size, embedding_dim)
            label: Labels (batch_size,) - 1 for same person, -1 for different
        
        Returns:
            loss: Scalar loss value
        """
        # Convert labels: 1 for same person, -1 for different
        # (PyTorch's cosine_embedding_loss expects this format)
        label_converted = label.clone()
        label_converted[label == 0] = -1
        
        # Cosine similarity loss
        loss = F.cosine_embedding_loss(embedding1, embedding2, label_converted, margin=self.margin)
        
        return loss


if __name__ == '__main__':
    print("=" * 80)
    print("Testing Contrastive Loss")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 32
    embedding_dim = 128
    
    embedding1 = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    embedding2 = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # Test Contrastive Loss
    contrastive_loss = ContrastiveLoss(margin=2.0)
    loss = contrastive_loss(embedding1, embedding2, labels)
    print(f"Loss: {loss.item():.4f}")
    
    # Calculate distances for similar vs dissimilar
    distances = F.pairwise_distance(embedding1, embedding2, p=2)
    similar_dist = distances[labels == 1].mean()
    dissimilar_dist = distances[labels == 0].mean()
    print(f"Avg distance (similar): {similar_dist:.4f}")
    print(f"Avg distance (dissimilar): {dissimilar_dist:.4f}")
    
    print("\n" + "=" * 80)
    print("Contrastive loss working correctly!")
    print("=" * 80)
