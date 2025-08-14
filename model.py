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
from get_stft_spectra import get_STFT_spectra_rectangular
from bridge import CorrelationAlignment


class PositionalEncoding(nn.Module):
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
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
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
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target sequence from decoder (batch_size, tgt_len, d_model)
            memory: Encoder output (batch_size, src_len, d_model)
            tgt_mask: Causal mask for target sequence
            memory_mask: Mask for encoder output (optional)
            
        Returns:
            Output tensor (batch_size, tgt_len, d_model)
        """
        # Self-attention with residual connection
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with residual connection
        # Q from decoder (tgt), K and V from encoder (memory)
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward with residual connection
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


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
        time_series_dim: int = 1,
        encoder_dim: int = 768,  # ViT encoder output dimension
    ):
        super().__init__()
        
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.time_series_dim = time_series_dim
        
        # Embedding for time series values
        self.value_embedding = nn.Linear(time_series_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=prediction_length + 1, dropout=dropout)
        
        # Project encoder output to decoder dimension for cross-attention
        self.encoder_projection = nn.Linear(encoder_dim, d_model)
        
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
            nn.Linear(d_model // 2, time_series_dim)
        )
        
        # Learnable start token
        self.start_token = nn.Parameter(torch.randn(1, 1, time_series_dim))
        
        # Initialize parameters with better initialization schemes
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """
        Initialize decoder parameters with better schemes for gradient sensitivity.
        """
        # Initialize start token with small values close to expected range
        nn.init.normal_(self.start_token, mean=0.0, std=0.02)
        
        # Initialize value embedding layer
        nn.init.xavier_uniform_(self.value_embedding.weight)
        nn.init.constant_(self.value_embedding.bias, 0.0)
        
        # Initialize encoder projection
        nn.init.xavier_uniform_(self.encoder_projection.weight)
        nn.init.constant_(self.encoder_projection.bias, 0.0)
        
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
                    nn.init.constant_(layer.bias, 0.0)
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
    
    def forward(
        self, 
        encoder_output: torch.Tensor,
        context_condition: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing.
        
        Args:
            encoder_output: Output from ViT encoder (batch_size, num_patches+1, encoder_dim)
            context_condition: Not used in conventional approach, kept for compatibility
            target: Ground truth for teacher forcing (batch_size, prediction_length, time_series_dim)
            use_teacher_forcing: Whether to use teacher forcing (True during training)
            
        Returns:
            Predictions (batch_size, prediction_length, time_series_dim)
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Project encoder output for cross-attention K, V (conventional approach)
        memory = self.encoder_projection(encoder_output)  # (batch_size, num_patches+1, d_model)
        # Memory sequence: [CLS, patch1, patch2, ..., patchN] - conventional ViT token order
        
        if use_teacher_forcing and target is not None:
            # Teacher forcing: use ground truth as input
            # Prepend start token to target
            start_tokens = self.start_token.expand(batch_size, 1, self.time_series_dim)
            decoder_input = torch.cat([start_tokens, target], dim=1)  # (batch_size, pred_len+1, ts_dim)
            
            # Embed and add positional encoding
            decoder_input = self.value_embedding(decoder_input)  # (batch_size, pred_len+1, d_model)
            decoder_input = self.pos_encoding(decoder_input)
            
            # Create causal mask
            tgt_len = decoder_input.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)
            
            # Pass through decoder layers
            output = decoder_input
            for layer in self.decoder_layers:
                output = layer(output, memory, tgt_mask=tgt_mask)
            
            # Project to output dimension and remove start token
            output = self.output_projection(output)  # (batch_size, pred_len+1, ts_dim)
            
            output = output[:, 1:, :]  # Remove start token: (batch_size, pred_len, ts_dim)
            
        else:
            # Inference mode: autoregressive generation
            predictions = []
            
            # Start with start token
            current_input = self.start_token.expand(batch_size, 1, self.time_series_dim)
            
            for step in range(self.prediction_length):
                # Embed current sequence
                embedded = self.value_embedding(current_input)  # (batch_size, step+1, d_model)
                embedded = self.pos_encoding(embedded)
                
                # Create causal mask
                tgt_len = embedded.size(1)
                tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)
                
                # Pass through decoder layers
                output = embedded
                for layer in self.decoder_layers:
                    output = layer(output, memory, tgt_mask=tgt_mask)
                
                # Get prediction for next time step
                next_pred = self.output_projection(output[:, -1:, :])  # (batch_size, 1, ts_dim)
                predictions.append(next_pred)
                
                # Append prediction to input for next iteration
                current_input = torch.cat([current_input, next_pred], dim=1)
            
            # Concatenate predictions
            output = torch.cat(predictions, dim=1)  # (batch_size, pred_len, ts_dim)
        
        return output


class ViTToTimeSeriesModel(nn.Module):
    """
    Architecture combining rectangular ViT encoder with Transformer decoder.
    
    Uses CORAL for domain bridging and proper cross-attention mechanism.
    Supports teacher forcing during training and autoregressive generation during inference.
    """
    
    def __init__(
        self,
        vit_model_name: str = "google/vit-base-patch16-224",  # Kept for compatibility
        image_size: int = 64,  # Height is always 64
        num_channels: int = 3,
        prediction_length: int = 96,
        context_length: int = 96,
        feature_projection_dim: int = 128,
        time_series_dim: int = 1,
        ts_model_dim: int = 64,
        ts_num_heads: int = 8,
        ts_num_layers: int = 4,
        ts_dim_feedforward: int = 1024,
        ts_dropout: float = 0.1,
        image_mean: list = [0.485, 0.456, 0.406],
        image_std: list = [0.229, 0.224, 0.225],
        use_lstm_decoder: bool = False,  # Kept for compatibility but not used
    ):
        """
        Initialize the model.
        
        Args:
            vit_model_name: Not used, kept for compatibility
            image_size: Height of spectrogram (always 64)
            num_channels: Number of channels in spectrogram
            prediction_length: Length of time series to predict
            context_length: Length of context window
            feature_projection_dim: Dimension for CORAL projection
            time_series_dim: Dimension of time series (usually 1 for univariate)
            ts_model_dim: Hidden dimension for transformer decoder
            ts_num_heads: Number of attention heads
            ts_num_layers: Number of decoder layers
            ts_dim_feedforward: Feed-forward dimension
            ts_dropout: Dropout rate
            image_mean: Mean values for image normalization
            image_std: Std values for image normalization
            use_lstm_decoder: Not used, always uses Transformer decoder
        """
        super().__init__()
        
        # Store configuration
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.time_series_dim = time_series_dim
        self.feature_projection_dim = feature_projection_dim
        self.ts_model_dim = ts_model_dim
        self.num_channels = num_channels
        
        # Image normalization parameters
        self.register_buffer('image_mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).view(1, 3, 1, 1))
        
        # Rectangular ViT Encoder (width = context_length)
        self.vit_encoder = create_rectangular_vit(
            image_height=64,  # Fixed height for spectrograms
            image_width=context_length,  # Width matches context length
            in_channels=num_channels,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            dropout=0.1
        )
        
        # CORAL Domain Bridge
        vit_hidden_size = 768  # Default ViT hidden size
        self.domain_bridge = CorrelationAlignment(
            input_dim=vit_hidden_size,
            output_dim=feature_projection_dim,
            use_bias=False
        )
        
        # Transformer Decoder with Cross-Attention
        self.ts_decoder = TransformerDecoderWithCrossAttention(
            d_model=ts_model_dim,
            nhead=ts_num_heads,
            num_layers=ts_num_layers,
            dim_feedforward=ts_dim_feedforward,
            dropout=ts_dropout,
            prediction_length=prediction_length,
            context_length=context_length,
            time_series_dim=time_series_dim,
            encoder_dim=768,  # ViT encoder output dimension (CORAL skipped)
        )
    
    def encode_spectrogram_to_features(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Encode spectrogram to feature representation using ViT (CORAL bridge skipped).
        
        Args:
            spectrogram: Spectrogram tensor (batch_size, channels, height=64, width=context_length)
            
        Returns:
            Encoded features (batch_size, num_patches+1, 768)
        """
        # Get all token features from ViT
        vit_features = self.vit_encoder.get_last_hidden_state(spectrogram)  # (batch, num_patches+1, 768)
        
        # Skip CORAL domain bridge - return ViT features directly
        return vit_features
    
    def forward(self, context: torch.Tensor, tf_target: torch.Tensor = None, mode: str = 'train') -> torch.Tensor:
        """
        Forward pass with teacher forcing for training or autoregressive for inference.
        
        Args:
            context: Input context time series (batch_size, context_length, features)
            tf_target: Target time series for teacher forcing (batch_size, prediction_length, features) - only used in training
            mode: 'train' for teacher forcing, 'inference' for autoregressive generation
            
        Returns:
            Predicted time series (batch_size, prediction_length, time_series_dim)
        """
        device = next(self.parameters()).device
        batch_size = context.size(0)
        
        # Get last feature column of context as condition
        context_condition = context[:, :, -1]  # (batch, context_len)
        
        # Generate spectrograms from context
        spectra_list = []
        for item in context.cpu().numpy():
            spectra = torch.from_numpy(get_STFT_spectra_rectangular(item, self.context_length))
            spectra = spectra.float().to(device)
            spectra_list.append(spectra)
        
        # Stack into batch tensor
        spectra_tensor = torch.stack(spectra_list, dim=0)  # (batch, channels, 64, context_length)
        
        # Encode spectrograms to features
        encoder_features = self.encode_spectrogram_to_features(spectra_tensor)
        
        if mode == 'train':
            # Teacher forcing: use ground truth target (based on time_series_dim)
            if tf_target is not None:
                if self.time_series_dim == 1:
                    # Single variable: use last feature
                    target_features = tf_target[:, :, -1:]  # (batch, pred_len, 1)
                else:
                    # Multi-variable: use last time_series_dim features
                    target_features = tf_target[:, :, -self.time_series_dim:]  # (batch, pred_len, time_series_dim)
            else:
                raise ValueError("tf_target must be provided in training mode")
            
            
            predictions = self.ts_decoder(
                encoder_output=encoder_features,
                context_condition=context_condition,
                target=target_features,
                use_teacher_forcing=True
            )
        else:
            # Inference: autoregressive generation
            predictions = self.ts_decoder(
                encoder_output=encoder_features,
                context_condition=context_condition,
                target=None,
                use_teacher_forcing=False
            )
        
        return predictions
    
    def inference(self, context: torch.Tensor) -> torch.Tensor:
        """
        Inference mode without teacher forcing.
        
        Args:
            context: Input context time series (batch_size, context_length, features)
            
        Returns:
            Predicted time series (batch_size, prediction_length, time_series_dim)
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
    vit_model: str = "google/vit-base-patch16-224",
    prediction_length: int = 96,
    context_length: int = 96,
    use_lstm_decoder: bool = False,  # Ignored, always uses Transformer
    **kwargs
) -> ViTToTimeSeriesModel:
    """
    Factory function to create ViTToTimeSeriesModel with common configurations.
    
    Args:
        vit_model: Not used, kept for compatibility
        prediction_length: Length of predictions
        context_length: Length of context window
        use_lstm_decoder: Ignored, always uses Transformer decoder
        **kwargs: Additional model arguments
        
    Returns:
        Initialized model
    """
    return ViTToTimeSeriesModel(
        vit_model_name=vit_model,
        prediction_length=prediction_length,
        context_length=context_length,
        use_lstm_decoder=False,  # Always use Transformer
        **kwargs
    )