import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from mri_gist import split_hemispheres, extract_left_hemisphere, extract_right_hemisphere


# `slow_r50` model 
ResNet3D = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

class OneClassSiameseSymmetry(nn.Module):
    """
    Learn symmetric embedding where left ≈ right for normal brains
    
    Key insight: For normal brains, left and right hemispheres should 
    produce similar features despite appearance variations (domain randomization)
    """
    
    def __init__(self):
        self.encoder = ResNet3D(output_dim=256)  # Shared for left/right
        self.projection_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def forward(self, left_volume, right_volume):
        # Extract features
        left_feat = self.encoder(left_volume)
        right_feat = self.encoder(right_volume)
        
        # Project to embedding space
        left_embed = self.projection_head(left_feat)
        right_embed = self.projection_head(right_feat)
        
        return left_embed, right_embed


def contrastive_symmetry_loss(left_embed, right_embed, temperature=0.5):
    """
    Contrastive loss: left and right from SAME brain should be close
    left and right from DIFFERENT brains should be far
    
    This is similar to SimCLR but for hemispheric symmetry
    """
    # Normalize embeddings
    left_norm = F.normalize(left_embed, dim=1)
    right_norm = F.normalize(right_embed, dim=1)
    
    # Positive pairs: left-right from same brain
    positive_similarity = torch.sum(left_norm * right_norm, dim=1)
    
    # Negative pairs: left from brain A, right from brain B
    # (Create batch of negative pairs)
    negatives = torch.mm(left_norm, right_norm.t())  # All pairs in batch
    
    # Contrastive loss
    logits = torch.cat([positive_similarity.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()  # First is positive
    
    loss = F.cross_entropy(logits / temperature, labels)
    return loss


def train_one_class_siamese(model, dhcp_normal_brains, synthesizer):
    """
    Training loop: Learn symmetric embeddings from normal brains only
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for iteration in range(100000):
        # Sample random normal brain
        label_map = random.choice(dhcp_normal_brains)
        
        # Split into hemispheres
        left_labels, right_labels = split_hemispheres(label_map)
        
        # Apply SAME domain randomization to both (preserves symmetry)
        aug_params = synthesizer.sample_augmentation_params()
        left_img = synthesizer.apply_domain_randomization(left_labels, aug_params)
        right_img = synthesizer.apply_domain_randomization(right_labels, aug_params)
        
        # CRITICAL: Despite extreme appearance changes (domain randomization),
        # left and right should still produce similar embeddings
        
        left_tensor = torch.from_numpy(left_img).unsqueeze(0).unsqueeze(0).float().cuda()
        right_tensor = torch.from_numpy(right_img).unsqueeze(0).unsqueeze(0).float().cuda()
        
        # Forward pass
        left_embed, right_embed = model(left_tensor, right_tensor)
        
        # Contrastive loss (batch requires multiple samples)
        # For simplicity, accumulate batch
        # ... (implement batching)
        
        loss = contrastive_symmetry_loss(left_embed, right_embed)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item():.4f}")


def detect_asymmetry(model, test_mri, threshold=0.7):
    """
    At test time: Compare left-right similarity
    
    Normal brain: left ≈ right (high similarity)
    Asymmetric brain: left ≠ right (low similarity)
    """
    left_hemisphere = extract_left_hemisphere(test_mri)
    right_hemisphere = extract_right_hemisphere(test_mri)
    
    left_embed, right_embed = model(left_hemisphere, right_hemisphere)
    
    # Cosine similarity
    similarity = F.cosine_similarity(left_embed, right_embed, dim=1)
    
    # Threshold for asymmetry
    is_asymmetric = similarity < threshold  # Low similarity = asymmetry
    
    return {
        'is_asymmetric': is_asymmetric,
        'similarity_score': similarity.item(),
        'confidence': abs(similarity.item() - threshold)
    }