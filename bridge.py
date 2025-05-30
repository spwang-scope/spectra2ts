"""
CORAL Domain Bridging Module

Implements Correlation Alignment (CORAL) for cross-domain feature bridging
between Vision Transformer features and Time Series features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrelationAlignment(nn.Module):
    """
    CORAL: Correlation Alignment for Domain Adaptation.
    
    Simple and effective domain bridging using correlation alignment.
    Matches second-order statistics (covariance) between domains.
    
    Based on "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
    """
    
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Single linear transformation layer
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        
        # Initialize as identity-like transformation when possible
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        if self.input_dim == self.output_dim:
            # Initialize as identity matrix for same dimensions
            nn.init.eye_(self.linear.weight)
        else:
            # Use orthogonal initialization for different dimensions
            nn.init.orthogonal_(self.linear.weight)
            
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CORAL alignment layer.
        
        Args:
            x: Input features of shape (batch_size, input_dim)
            
        Returns:
            Aligned features of shape (batch_size, output_dim)
        """
        return self.linear(x)
    
    def coral_loss(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """
        Compute CORAL alignment loss between source and target features.
        
        This loss minimizes the difference between covariance matrices
        of source and target domain features.
        
        Args:
            source_features: Source domain features (batch_size, feature_dim)
            target_features: Target domain features (batch_size, feature_dim)
            
        Returns:
            CORAL alignment loss (scalar)
        """
        if source_features.size(1) != target_features.size(1):
            raise ValueError("Source and target features must have same feature dimension")
            
        source_cov = self._compute_covariance(source_features)
        target_cov = self._compute_covariance(target_features)
        
        # Frobenius norm of covariance difference
        loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        
        # Normalize by feature dimension squared
        normalized_loss = loss / (4 * source_features.size(1) ** 2)
        
        return normalized_loss
    
    def _compute_covariance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute covariance matrix of features.
        
        Args:
            features: Feature tensor of shape (batch_size, feature_dim)
            
        Returns:
            Covariance matrix of shape (feature_dim, feature_dim)
        """
        batch_size = features.size(0)
        
        if batch_size == 1:
            # Handle single sample case
            return torch.zeros(
                features.size(1), features.size(1), 
                dtype=features.dtype, device=features.device
            )
        
        # Center the features (subtract mean)
        centered_features = features - features.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        covariance = torch.mm(centered_features.t(), centered_features) / (batch_size - 1)
        
        return covariance
    
    def get_feature_statistics(self, features: torch.Tensor) -> dict:
        """
        Get statistical information about features for analysis.
        
        Args:
            features: Feature tensor of shape (batch_size, feature_dim)
            
        Returns:
            Dictionary containing mean, std, and covariance matrix
        """
        return {
            'mean': features.mean(dim=0),
            'std': features.std(dim=0),
            'covariance': self._compute_covariance(features),
            'shape': features.shape
        }


class MultiScaleCORAL(nn.Module):
    """
    Multi-scale CORAL alignment for handling features at different scales.
    
    Useful when bridging very different feature spaces (e.g., vision to time series).
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        
        # Create multiple CORAL layers for different scales
        self.coral_layers = nn.ModuleList([
            CorrelationAlignment(input_dim, output_dim)
            for _ in range(num_scales)
        ])
        
        # Scale-specific weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale CORAL.
        
        Args:
            x: Input features
            
        Returns:
            Weighted combination of multi-scale aligned features
        """
        outputs = []
        
        for coral_layer in self.coral_layers:
            aligned = coral_layer(x)
            outputs.append(aligned)
        
        # Weighted combination
        stacked_outputs = torch.stack(outputs, dim=0)  # (num_scales, batch_size, output_dim)
        weights = F.softmax(self.scale_weights, dim=0).view(-1, 1, 1)
        
        weighted_output = (weights * stacked_outputs).sum(dim=0)
        
        return weighted_output
    
    def coral_loss(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale CORAL loss."""
        total_loss = 0.0
        
        for coral_layer in self.coral_layers:
            # Apply each CORAL layer to get aligned features
            source_aligned = coral_layer(source_features)
            target_aligned = coral_layer(target_features)
            
            # Compute CORAL loss for this scale
            scale_loss = coral_layer.coral_loss(source_aligned, target_aligned)
            total_loss += scale_loss
            
        return total_loss / self.num_scales


def create_coral_bridge(input_dim: int, output_dim: int, bridge_type: str = "standard") -> nn.Module:
    """
    Factory function to create different types of CORAL bridges.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension  
        bridge_type: Type of bridge ("standard" or "multiscale")
        
    Returns:
        CORAL bridging module
    """
    if bridge_type == "standard":
        return CorrelationAlignment(input_dim, output_dim)
    elif bridge_type == "multiscale":
        return MultiScaleCORAL(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown bridge type: {bridge_type}")


if __name__ == "__main__":
    # Test CORAL alignment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create CORAL bridge
    coral = CorrelationAlignment(768, 256).to(device)
    
    # Test data
    source_features = torch.randn(32, 768, device=device)
    target_features = torch.randn(32, 768, device=device)
    
    # Forward pass
    aligned_source = coral(source_features)
    aligned_target = coral(target_features)
    
    print(f"Input shape: {source_features.shape}")
    print(f"Output shape: {aligned_source.shape}")
    
    # Compute CORAL loss
    loss = coral.coral_loss(aligned_source, aligned_target)
    print(f"CORAL loss: {loss.item():.6f}")
    
    # Check parameter count
    param_count = sum(p.numel() for p in coral.parameters())
    print(f"CORAL parameters: {param_count:,}")
