"""
Custom Vision Transformer (ViT) implementation with support for rectangular images.
Based on the original ViT paper and adapted for flexible input dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, parse_shape
from einops.layers.torch import Rearrange
import math


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""
    
    def __init__(self, image_height=64, image_width=96, patch_size=8, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.num_patches_h = image_height // patch_size
        self.num_patches_w = image_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
    def forward(self, x):
        return self.projection(x)


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for rectangular images."""
    
    def __init__(self, embed_dim, max_height=100, max_width=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create 2D positional encoding
        pe = torch.zeros(max_height, max_width, embed_dim)
        
        # Height position encoding (uses half of embed_dim)
        pos_h = torch.arange(0, max_height).unsqueeze(1).float()
        pos_w = torch.arange(0, max_width).unsqueeze(1).float()
        
        # Divide embed_dim into two parts for height and width
        d_h = embed_dim // 2
        d_w = embed_dim - d_h
        
        div_term_h = torch.exp(torch.arange(0, d_h, 2).float() * 
                               -(math.log(10000.0) / d_h))
        div_term_w = torch.exp(torch.arange(0, d_w, 2).float() * 
                               -(math.log(10000.0) / d_w))
        
        # Height encoding
        pe[:, :, 0:d_h:2] = torch.sin(pos_h * div_term_h).unsqueeze(1).expand(-1, max_width, -1)
        pe[:, :, 1:d_h:2] = torch.cos(pos_h * div_term_h).unsqueeze(1).expand(-1, max_width, -1)
        
        # Width encoding
        pe[:, :, d_h::2] = torch.sin(pos_w * div_term_w).unsqueeze(0).expand(max_height, -1, -1)
        pe[:, :, d_h+1::2] = torch.cos(pos_w * div_term_w).unsqueeze(0).expand(max_height, -1, -1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, num_patches_h, num_patches_w, device=None):
        """Get positional encoding for specific patch grid size."""
        pos_enc = self.pe[:num_patches_h, :num_patches_w, :]
        if device is not None:
            pos_enc = pos_enc.to(device)
        pos_enc = rearrange(pos_enc, 'h w d -> (h w) d')
        return pos_enc


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Aggregate
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RectangularViT(nn.Module):
    """
    Vision Transformer with support for rectangular images.
    Optimized for image widths of 96, 192, 336, and 720 pixels.
    """
    
    def __init__(
        self,
        image_height=128,
        image_width=128,
        patch_size=8,
        in_channels=1,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1,
        embed_dropout=0.1,
    ):
        super().__init__()
        
        # Validate image dimensions
        assert image_height % patch_size == 0, f"Image height {image_height} must be divisible by patch size {patch_size}"
        assert image_width % patch_size == 0, f"Image width {image_width} must be divisible by patch size {patch_size}"
        
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.num_patches_h = image_height // patch_size
        self.num_patches_w = image_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_height, image_width, patch_size, in_channels, embed_dim
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional encoding (2D for rectangular support)
        self.pos_encoding = PositionalEncoding2D(
            embed_dim, 
            max_height=100,  # Support up to 100x100 patches
            max_width=100,
            dropout=embed_dropout
        )
        
        self.embed_dropout = nn.Dropout(embed_dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head (can be replaced for other tasks)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, x, return_features=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_features: If True, return features instead of classification output
            
        Returns:
            Output tensor or features
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add positional encoding
        pos_enc = self.pos_encoding(self.num_patches_h, self.num_patches_w, device=x.device)
        pos_enc = torch.cat([torch.zeros_like(cls_tokens[0, 0, :]).unsqueeze(0), pos_enc], dim=0)
        x = x + pos_enc.unsqueeze(0).to(x.device)
        
        x = self.embed_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        if return_features:
            return x  # Return all token features
        
        # Classification (use CLS token)
        cls_output = x[:, 0]
        return self.head(cls_output)
    
    def get_last_hidden_state(self, x):
        """Get the last hidden state (all tokens) from the transformer."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        pos_enc = self.pos_encoding(self.num_patches_h, self.num_patches_w, device=x.device)
        pos_enc = torch.cat([torch.zeros_like(cls_tokens[0, 0, :]).unsqueeze(0), pos_enc], dim=0)
        x = x + pos_enc.unsqueeze(0).to(x.device)
        
        x = self.embed_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        return x  # Return all tokens including CLS


def create_rectangular_vit(image_height=128, image_width=128, **kwargs):
    """
    Factory function to create a rectangular ViT model.
    
    Args:
        image_height: Height of input images
        image_width: Width of input images (96, 192, 336, or 720 recommended)
        **kwargs: Additional arguments for RectangularViT
        
    Returns:
        RectangularViT model
    """
    
    patch_size = 8  # Default
        
    return RectangularViT(
        image_height=image_height,
        image_width=image_width,
        patch_size=patch_size,
        **kwargs
    )
