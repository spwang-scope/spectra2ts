"""
ViT to TimeSeries Model with Transformer Decoder and Cross-Attention

Combines custom rectangular ViT encoder with Transformer decoder
using CORAL domain bridging and proper cross-attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import numpy as np
import math
import sys
import os

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vit_encoder import RectangularViT, create_rectangular_vit
from pytorch_stft import get_STFT_spectra  # Importing the STFT function from pytorch_stft.py

import numpy as np
import matplotlib.pyplot as plt

class ValueEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ValueEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

'''
class DecoderPositionalEmbedding(nn.Module):
    """
    Dynamic positional encoding for transformer decoder that computes encodings on-the-fly.
    Allows for variable prediction lengths without fixed buffer size limitations.
    Maintains backward compatibility with checkpoints trained using fixed PositionalEmbedding.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        # Pre-compute div_term for efficiency (this doesn't depend on sequence length)
        self.register_buffer('div_term', torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        ))
        
        # Cache for computed positional encodings to avoid recomputation
        self._pe_cache = {}
    
    def _compute_pe(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute positional encoding for given sequence length."""
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
            
        # Check cache first
        cache_key = (seq_len, device.type, device.index if device.index is not None else 0)
        if cache_key in self._pe_cache:
            return self._pe_cache[cache_key]
        
        # Compute positional encoding
        pe = torch.zeros(seq_len, self.d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = self.div_term.to(device)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
        
        # Cache the result (limit cache size to prevent memory issues)
        if len(self._pe_cache) < 100:  # Reasonable cache limit
            self._pe_cache[cache_key] = pe
            
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        """
        seq_len = x.size(1)
        device = x.device
        
        # Get or compute positional encoding for this sequence length
        pe = self._compute_pe(seq_len, device)
        
        # Add positional encoding and apply dropout
        x = x + pe
        return self.dropout(x)
    
    def clear_cache(self):
        """Clear the positional encoding cache."""
        self._pe_cache.clear()
'''

class CachedMultiHeadAttention(nn.Module):
    """
    MultiHead Attention with KV caching support for autoregressive generation.
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            query: (batch_size, tgt_len, d_model)
            key: (batch_size, src_len, d_model) 
            value: (batch_size, src_len, d_model)
            attn_mask: (tgt_len, src_len) or None
            use_cache: Whether to use/update cache
            cache: Dict with 'k' and 'v' tensors for caching
            
        Returns:
            output: (batch_size, tgt_len, d_model)
            updated_cache: Updated cache dict if use_cache=True
        """
        batch_size, tgt_len, _ = query.shape
        
        # Apply linear projections
        Q = self.q_linear(query)  # (batch, tgt_len, d_model)
        K = self.k_linear(key)    # (batch, src_len, d_model) 
        V = self.v_linear(value)  # (batch, src_len, d_model)
        
        # Handle KV caching
        updated_cache = None
        if use_cache:
            if cache is not None and 'k' in cache and 'v' in cache:
                # Concatenate new K,V with cached K,V
                K = torch.cat([cache['k'], K], dim=1)  # (batch, cached_len + src_len, d_model)
                V = torch.cat([cache['v'], V], dim=1)  # (batch, cached_len + src_len, d_model)
            
            # Update cache with new K,V
            updated_cache = {'k': K.clone(), 'v': V.clone()}
        
        src_len = K.size(1)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, tgt_len, self.nhead, self.head_dim).transpose(1, 2)  # (batch, nhead, tgt_len, head_dim)
        K = K.view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)  # (batch, nhead, src_len, head_dim)
        V = V.view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)  # (batch, nhead, src_len, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch, nhead, tgt_len, src_len)
        
        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:  # (tgt_len, src_len)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, src_len)
            attn_scores = attn_scores + attn_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, nhead, tgt_len, head_dim)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        output = self.out_proj(attn_output)
        
        return output, updated_cache


class TransformerDecoderLayer(nn.Module):
    """
    Custom transformer decoder layer with self-attention and cross-attention.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        # Self-attention with caching support
        self.self_attn = CachedMultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention with caching support
        self.cross_attn = CachedMultiHeadAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        self_attn_cache: Optional[Dict[str, torch.Tensor]] = None,
        cross_attn_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            tgt: Target sequence from decoder (batch_size, tgt_len, d_model)
            memory: Encoder output (batch_size, src_len, d_model)
            tgt_mask: Causal mask for target sequence
            use_cache: Whether to use KV caching
            self_attn_cache: Cache for self-attention
            cross_attn_cache: Cache for cross-attention
            
        Returns:
            output: Output tensor (batch_size, tgt_len, d_model)
            updated_self_cache: Updated self-attention cache
            updated_cross_cache: Updated cross-attention cache
        """
        # Self-attention with residual connection
        tgt2, updated_self_cache = self.self_attn(
            query=tgt, key=tgt, value=tgt, 
            attn_mask=tgt_mask, 
            use_cache=use_cache, 
            cache=self_attn_cache
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with residual connection
        # Q from decoder (tgt), K and V from encoder (memory)
        tgt2, updated_cross_cache = self.cross_attn(
            query=tgt, key=memory, value=memory, 
            attn_mask=None,  # No mask for cross-attention
            use_cache=use_cache,
            cache=cross_attn_cache
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward with residual connection
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, updated_self_cache, updated_cross_cache

class TransformerDecoderWithCrossAttention(nn.Module):
    """
    Transformer decoder with proper cross-attention mechanism.
    
    Q comes from decoder self-attention
    K, V come from encoder (ViT) output
    Additional conditioning from time series context
    """
    
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        prediction_length: int = 96,
        context_length: int = 96,
        pred_dim: int = 1,
        encoder_dim: int = 768,  # ViT encoder output dimension
    ):
        super().__init__()
        
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.pred_dim = pred_dim
        
        # Embedding for time series values
        self.value_embedding = nn.Linear(pred_dim, d_model)
        
        # Dynamic positional encoding for variable prediction lengths
        self.pos_encoding = PositionalEmbedding(d_model, dropout=dropout)
        
        # Project encoder output to decoder dimension for cross-attention
        self.encoder_projection = nn.Linear(encoder_dim, d_model)
        
        # Project context condition (last feature of context) to decoder dimension
        self.context_condition_projection = nn.Linear(context_length, d_model)
        
        # Project encoder CLS token to time series start token
        self.start_token_projection = nn.Linear(encoder_dim, pred_dim)
        
        # Custom transformer decoder layers with cross-attention
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_dim)
        )
        
        # Static cross-attention memory and cache buffers
        self.static_cross_kv = None
        self.cross_attn_cache = None  # Static + dynamic K,V for cross-attention
        self.self_attn_cache = None   # Dynamic K,V for self-attention
        
        # Initialize parameters with better initialization schemes
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """
        Initialize decoder parameters with better schemes for gradient sensitivity.
        """
        # Initialize value embedding layer
        nn.init.xavier_uniform_(self.value_embedding.weight)
        nn.init.constant_(self.value_embedding.bias, 0.0)
        
        # Initialize encoder projection
        nn.init.xavier_uniform_(self.encoder_projection.weight)
        nn.init.constant_(self.encoder_projection.bias, 0.0)
        
        # Initialize start token projection
        nn.init.xavier_uniform_(self.start_token_projection.weight)
        nn.init.constant_(self.start_token_projection.bias, 0.0)
        
        # Initialize transformer decoder layers with scaled initialization
        for layer in self.decoder_layers:
            # Self-attention
            self._init_multihead_attention(layer.self_attn)
            # Cross-attention  
            self._init_multihead_attention(layer.cross_attn)
            # Feed-forward layers
            self._init_feedforward(layer.ffn)
            # Layer norms
            nn.init.constant_(layer.norm1.weight, 1.0)
            nn.init.constant_(layer.norm1.bias, 0.0)
            nn.init.constant_(layer.norm2.weight, 1.0)
            nn.init.constant_(layer.norm2.bias, 0.0)
            nn.init.constant_(layer.norm3.weight, 1.0)
            nn.init.constant_(layer.norm3.bias, 0.0)
        
        # Initialize output projection with smaller weights for stable training
        for i, layer in enumerate(self.output_projection):
            if isinstance(layer, nn.Linear):
                if i == len(self.output_projection) - 1:  # Final layer
                    # Smaller initialization for final output layer
                    nn.init.xavier_uniform_(layer.weight, gain=0.01)
                    nn.init.constant_(layer.bias, 0.1)
                else:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
    
    def _init_multihead_attention(self, attention_layer):
        """Initialize multihead attention layer with proper scaling."""
        # Access the underlying linear layers
        # MultiheadAttention has in_proj_weight (for Q, K, V) and out_proj
        if hasattr(attention_layer, 'in_proj_weight') and attention_layer.in_proj_weight is not None:
            # Combined Q, K, V projection
            nn.init.xavier_uniform_(attention_layer.in_proj_weight)
        else:
            # Separate Q, K, V projections
            if hasattr(attention_layer, 'q_proj_weight'):
                nn.init.xavier_uniform_(attention_layer.q_proj_weight)
            if hasattr(attention_layer, 'k_proj_weight'):
                nn.init.xavier_uniform_(attention_layer.k_proj_weight) 
            if hasattr(attention_layer, 'v_proj_weight'):
                nn.init.xavier_uniform_(attention_layer.v_proj_weight)
        
        # Initialize biases
        if hasattr(attention_layer, 'in_proj_bias') and attention_layer.in_proj_bias is not None:
            nn.init.constant_(attention_layer.in_proj_bias, 0.0)
        
        # Output projection
        nn.init.xavier_uniform_(attention_layer.out_proj.weight, gain=1/math.sqrt(2))  # Residual scaling
        nn.init.constant_(attention_layer.out_proj.bias, 0.0)
    
    def _init_feedforward(self, ffn_module):
        """Initialize feed-forward network with residual scaling."""
        for i, layer in enumerate(ffn_module):
            if isinstance(layer, nn.Linear):
                if i == len(ffn_module) - 1:  # Final layer in FFN
                    # Scale down final layer for residual connections
                    nn.init.xavier_uniform_(layer.weight, gain=1/math.sqrt(2))
                else:
                    nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def setup_encoder_memory(self, encoder_output: torch.Tensor, context_condition: torch.Tensor):
        """
        Pre-compute static cross-attention K,V from encoder output.
        This should be called once before teacher forcing or inference.
        
        Args:
            encoder_output: Output from ViT encoder (batch_size, num_patches+1, encoder_dim)
            context_condition: Last feature column of context (batch_size, context_length)
        """
        # Project encoder output to decoder dimension  
        static_memory = self.encoder_projection(encoder_output)  # (batch, num_patches+1, d_model)
        
        # Project context condition
        context_vec = self.context_condition_projection(context_condition)  # (batch, d_model)
        context_vec = context_vec.unsqueeze(1)  # (batch, 1, d_model)
        
        # Combine static memory for cross-attention
        self.static_cross_kv = torch.cat([static_memory, context_vec], dim=1)  # (batch, num_patches+2, d_model)
    
    def get_start_token(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Project encoder CLS token to time series start token.
        
        Args:
            encoder_output: Output from ViT encoder (batch_size, num_patches+1, encoder_dim)
            
        Returns:
            Start token tensor (batch_size, 1, pred_dim)
        """
        cls_token = encoder_output[:, -1, -self.pred_dim:]  # (batch, encoder_dim) - CLS token is last
        start_token = self.start_token_projection(cls_token)  # (batch, pred_dim)
        return start_token.unsqueeze(1)  # (batch, 1, pred_dim)
    
    def clear_dynamic_cache(self):
        """
        Clear prediction parts of cache, keep encoder parts.
        Should be called after inference to reset cache for next sequence.
        """
        if self.static_cross_kv is not None and self.cross_attn_cache is not None:
            # Keep static part (encoder + context), clear dynamic part
            static_size = self.static_cross_kv.size(1)  # num_patches + 2
            if self.cross_attn_cache.size(1) > static_size:
                self.cross_attn_cache = self.static_cross_kv.clone()  # Reset to static only
        
        # Clear self-attention cache completely
        self.self_attn_cache = None
    
    def forward(
        self, 
        encoder_output: torch.Tensor,
        context_condition: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with KV caching support.
        
        Args:
            encoder_output: Output from ViT encoder (batch_size, num_patches+1, encoder_dim)
            context_condition: Last feature column of context (batch_size, context_length)
            target: Ground truth for teacher forcing (batch_size, prediction_length, pred_dim)
            use_teacher_forcing: Whether to use teacher forcing (True during training)
            
        Returns:
            Predictions (batch_size, prediction_length, pred_dim)
        """
        # Setup static encoder memory once
        self.setup_encoder_memory(encoder_output, context_condition)
        
        if use_teacher_forcing and target is not None:
            return self._forward_teacher_forcing(encoder_output, target)
        else:
            return self._forward_inference(encoder_output)
    
    def _forward_teacher_forcing(self, encoder_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Teacher forcing forward pass with static memory and causal mask."""
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Get start token from encoder CLS token
        start_token = self.get_start_token(encoder_output)  # (batch, 1, pred_dim)
        
        # Pre-allocate decoder input tensor for efficiency
        decoder_input = torch.zeros(batch_size, self.prediction_length, self.pred_dim,
                                   device=device, dtype=target.dtype)
        decoder_input[:, 0, :] = start_token.squeeze(1)  # Set start token
        decoder_input[:, 1:, :] = target[:, :-1, :]  # Set shifted target
        
        # Embed and add positional encoding
        decoder_input = self.value_embedding(decoder_input)  # (batch_size, pred_len, d_model)
        decoder_input = self.pos_encoding(decoder_input)
        
        # Create causal mask
        tgt_len = decoder_input.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)
        
        # Pass through decoder layers with static memory (no caching)
        output = decoder_input
        for layer in self.decoder_layers:
            output, _, _ = layer(
                tgt=output, 
                memory=self.static_cross_kv, 
                tgt_mask=tgt_mask,
                use_cache=False,  # No caching during teacher forcing
                self_attn_cache=None,
                cross_attn_cache=None
            )
        
        # Project to output dimension
        output = self.output_projection(output)  # (batch_size, pred_len, pred_dim)
        return output
    
    def _forward_inference(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Inference forward pass with KV caching."""
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Get start token from encoder CLS token
        start_token = self.get_start_token(encoder_output)  # (batch, 1, pred_dim)
        
        # Initialize caches with static memory
        self.cross_attn_cache = self.static_cross_kv.clone()  # Start with static memory
        self.self_attn_cache = None  # Will be initialized on first use
        
        # Pre-allocate output tensor
        output = torch.zeros(batch_size, self.prediction_length, self.pred_dim,
                           device=device, dtype=torch.float32)
        
        # Initialize current input with start token
        current_input = start_token  # (batch, 1, pred_dim)
        
        for step in range(self.prediction_length):
            # Embed current input (just the latest token)
            embedded = self.value_embedding(current_input)  # (batch, 1, d_model)
            embedded = self.pos_encoding(embedded)
            
            # Pass through decoder layers with caching
            layer_output = embedded
            layer_self_cache = self.self_attn_cache
            layer_cross_cache = self.cross_attn_cache
            
            for layer_idx, layer in enumerate(self.decoder_layers):
                layer_output, updated_self_cache, updated_cross_cache = layer(
                    tgt=layer_output, 
                    memory=current_input,  # For cross-attention K,V (new prediction)
                    tgt_mask=None,  # No mask needed for single token
                    use_cache=True,
                    self_attn_cache=layer_self_cache,
                    cross_attn_cache=layer_cross_cache
                )
                
                # Update caches (each layer shares the same cache structure)
                if layer_idx == 0:  # Only update on first layer to avoid redundant updates
                    self.self_attn_cache = updated_self_cache
                    self.cross_attn_cache = updated_cross_cache
            
            # Get prediction for current step
            next_pred = self.output_projection(layer_output)  # (batch, 1, pred_dim)
            
            # Store prediction in output tensor
            output[:, step, :] = next_pred.squeeze(1)
            
            # Update current_input for next iteration
            current_input = next_pred  # Next iteration will process this prediction
        
        # Clear dynamic cache after inference
        self.clear_dynamic_cache()
        
        return output


class ViTToTimeSeriesModel(nn.Module):
    """
    Architecture combining rectangular ViT encoder with Transformer decoder.
    
    Uses linear projection for feature adaptation and cross-attention mechanism.
    Supports teacher forcing during training and autoregressive generation during inference.
    """
    
    def __init__(
        self,
        seq_dim: int = 1,
        prediction_length: int = 96,
        context_length: int = 96,
        feature_projection_dim: int = 128,
        pred_dim: int = 1,
        d_model: int = 64,
        ts_num_heads: int = 8,
        ts_num_layers: int = 4,
        ts_dim_feedforward: int = 1024,
        ts_dropout: float = 0.1,
    ):
        """
        Initialize the model.
        
        Args:
            seq_dim: Number of channels in input sequence
            prediction_length: Length of time series to predict
            context_length: Length of context window
            feature_projection_dim: Dimension for feature projection
            pred_dim: Dimension of time series (usually 1 for univariate)
            d_model: Hidden dimension for transformer decoder
            ts_num_heads: Number of attention heads
            ts_num_layers: Number of decoder layers
            ts_dim_feedforward: Feed-forward dimension
            ts_dropout: Dropout rate
            image_mean: Mean values for image normalization
            image_std: Std values for image normalization
        """
        super().__init__()
        
        # Store configuration
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.pred_dim = pred_dim
        self.feature_projection_dim = feature_projection_dim
        self.d_model = d_model
        self.seq_dim = seq_dim
        self.context_proj_dim = 32

        self.value_embedding = ValueEmbedding(seq_dim, self.context_proj_dim)
        self.position_embedding = PositionalEmbedding(self.context_proj_dim, dropout=ts_dropout)
        
        
        # Rectangular ViT Encoder (128x128 spectrograms)
        self.vit_encoder = create_rectangular_vit(
            image_height=128,  # Updated height for resized spectrograms
            image_width=128,  # Updated width for resized spectrograms
            embed_dim=768,
            depth=3,
            num_heads=6,
            mlp_ratio=4,
            dropout=0.1
        )
        
        # Linear projection for encoder features
        vit_hidden_size = 768  # Default ViT hidden size
        self.encoder_projection = nn.Linear(vit_hidden_size, feature_projection_dim)
        
        
        # Transformer Decoder with Cross-Attention
        self.ts_decoder = TransformerDecoderWithCrossAttention(
            d_model=d_model,
            nhead=ts_num_heads,
            num_layers=ts_num_layers,
            dim_feedforward=ts_dim_feedforward,
            dropout=ts_dropout,
            prediction_length=prediction_length,
            context_length=context_length,
            pred_dim=pred_dim,
            encoder_dim=feature_projection_dim,  # After linear projection
        )
        
        # TSLib standard: no additional normalization in model (handled in data loader)
    
    def forward(self, context: torch.Tensor, tf_target: torch.Tensor = None, mode: str = 'train') -> torch.Tensor:
        """
        Forward pass with TSLib standard preprocessing (normalized input from data loader).
        
        Args:
            context: Input context time series (batch_size, context_length, num_features) - already StandardScaler normalized
            tf_target: Target time series for teacher forcing (batch_size, prediction_length, num_features) - already normalized
            mode: 'train' for teacher forcing, 'inference' for autoregressive generation
            
        Returns:
            Predicted time series (batch_size, prediction_length, 1) - normalized scale (TSLib standard)
        """
        device = next(self.parameters()).device
        batch_size = context.size(0)

        print('context shape:', context.shape)
        context = self.value_embedding(context)  # (batch, context_len, context_proj_dim)
        print('context shape:', context.shape)
        context = self.position_embedding(context)  # (batch, context_len, context_proj_dim)
        print('context shape:', context.shape)
        
        # Step 1: Get last feature column of context as condition (already normalized by StandardScaler)
        context_condition = context[:, :, -1]  # (batch, context_len)
        
        # Step 2: Generate spectrograms from normalized context
        spectra_list = []
        for item in context:
            spectra = get_STFT_spectra(item, device=device)
            spectra_list.append(spectra)
        
        # Stack into batch tensor
        spectra_tensor = torch.stack(spectra_list, dim=0)  # (batch, channels, 64, context_length)
        
        # Step 3: Process through ViT encoder
        vit_features = self.vit_encoder.get_last_hidden_state(spectra_tensor)  # (batch, num_patches+1, 768)
        encoder_features = self.encoder_projection(vit_features)  # (batch, num_patches+1, feature_projection_dim)
        
        # Step 4: Decoder forward pass
        if mode == 'train':
            # Teacher forcing: prepare target features
            if tf_target is not None:
                # Target already normalized by StandardScaler
                # CRITICAL: Select target feature for univariate prediction
                if self.pred_dim == 1:
                    # Single variable: use last feature only
                    target_features = tf_target[:, :, -1:]  # (batch, pred_len, 1)
                else:
                    # Multi-variable: use last pred_dim features
                    target_features = tf_target[:, :, -self.pred_dim:]  # (batch, pred_len, pred_dim)
            else:
                raise ValueError("tf_target must be provided in training mode")
            
            predictions = self.ts_decoder(
                encoder_output=encoder_features,
                context_condition=context_condition,  # Pass context condition for cross-attention
                target=target_features,
                use_teacher_forcing=True
            )
        else:
            # Inference: autoregressive generation
            predictions = self.ts_decoder(
                encoder_output=encoder_features,
                context_condition=context_condition,  # Pass context condition for cross-attention
                target=None,
                use_teacher_forcing=False
            )
        
        # Return normalized predictions (TSLib standard for loss computation)
        return predictions
    
    def inference(self, context: torch.Tensor) -> torch.Tensor:
        """
        Inference mode without teacher forcing.
        
        Args:
            context: Input context time series (batch_size, context_length, features)
            
        Returns:
            Predicted time series (batch_size, prediction_length, pred_dim)
        """
        return self.forward(context=context, tf_target=None, mode='inference')
    
    def freeze_vit_encoder(self, freeze: bool = True):
        """Freeze or unfreeze the ViT encoder parameters."""
        for param in self.vit_encoder.parameters():
            param.requires_grad = not freeze
    
    def freeze_ts_decoder(self, freeze: bool = True):
        """Freeze or unfreeze the Transformer decoder parameters."""
        for param in self.ts_decoder.parameters():
            param.requires_grad = not freeze


def create_model(
    prediction_length: int = 96,
    context_length: int = 96,
    **kwargs
) -> ViTToTimeSeriesModel:
    """
    Factory function to create ViTToTimeSeriesModel with common configurations.
    
    Args:
        prediction_length: Length of predictions
        context_length: Length of context window
        **kwargs: Additional model arguments
        
    Returns:
        Initialized model
    """
    return ViTToTimeSeriesModel(
        prediction_length=prediction_length,
        context_length=context_length,
        **kwargs
    )