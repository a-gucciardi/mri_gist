# Quick start implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ResNet

class OneClassSymmetryDetector:
    """
    One-class learning for brain symmetry
    Training: Normal brains only
    Testing: Flag deviations as asymmetric
    """
    
    def __init__(self):
        # Shared encoder for left and right hemispheres
        self.encoder = ResNet(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=256,  # Embedding dimension
            block='bottleneck',
            layers=[3, 4, 6, 3],
            widen_factor=1
        ).cuda()
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).cuda()
    
    def get_embeddings(self, left_volume, right_volume):
        """Extract embeddings for left and right hemispheres"""
        left_feat = self.encoder(left_volume)
        right_feat = self.encoder(right_volume)
        
        left_embed = self.projection(left_feat)
        right_embed = self.projection(right_feat)
        
        return left_embed, right_embed
    
    def train_on_normal(self, dhcp_label_maps, num_iterations=100000):
        """
        Stage 1: Pre-train on normal brains
        """
        from domain_randomization import DomainRandomizedSynthesis
        
        synthesizer = DomainRandomizedSynthesis()
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.projection.parameters()),
            lr=1e-4
        )
        
        for iteration in range(num_iterations):
            # Sample normal brain
            label_map = dhcp_label_maps[np.random.randint(len(dhcp_label_maps))]
            
            # Generate symmetric pair with domain randomization
            left_img, right_img, _ = synthesizer.synthesize_symmetric_pair(label_map)
            
            # Convert to tensors
            left_t = torch.from_numpy(left_img).unsqueeze(0).unsqueeze(0).float().cuda()
            right_t = torch.from_numpy(right_img).unsqueeze(0).unsqueeze(0).float().cuda()
            
            # Get embeddings
            left_embed, right_embed = self.get_embeddings(left_t, right_t)
            
            # Contrastive loss (simple version: maximize similarity)
            loss = 1 - F.cosine_similarity(left_embed, right_embed, dim=1).mean()
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iteration % 1000 == 0:
                print(f"Iter {iteration}, Loss: {loss.item():.4f}")
    
    def detect_asymmetry(self, mri_volume, threshold=0.75):
        """
        Test: Detect if brain is asymmetric
        """
        left = extract_left_hemisphere(mri_volume)
        right = extract_right_hemisphere(mri_volume)
        
        left_embed, right_embed = self.get_embeddings(left, right)
        
        similarity = F.cosine_similarity(left_embed, right_embed, dim=1).item()
        
        return {
            'is_asymmetric': similarity < threshold,
            'similarity': similarity,
            'anomaly_score': 1 - similarity
        }


# Usage
detector = OneClassSymmetryDetector()

# Stage 1: Train on normal dHCP
detector.train_on_normal(dhcp_label_maps)

# Stage 2 (optional): Few-shot fine-tune on BONBID-HIE

# Test
result = detector.detect_asymmetry(test_mri)
print(f"Asymmetric: {result['is_asymmetric']}")
print(f"Anomaly score: {result['anomaly_score']:.3f}")