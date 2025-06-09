"""
ViT to TimeSeries Model

Combines Vision Transformer encoder with custom Transformer decoder
using CORAL domain bridging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
from transformers import ViTConfig
from typing import Optional, Dict, Any, Tuple
import numpy as np
import math
from get_stft_spectra import get_STFT_spectra
from util import pad_to_multiple_of_4_center_bottom, pad_to_64_center_bottom

from bridge import CorrelationAlignment


class PositionalEncoding(nn.Module):
    """
    Positional encoding for time series generation.
    Similar to transformer positional encoding but adapted for time series.
    """
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (context_length, batch_size, d_model)
        """
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformerDecoder(nn.Module):
    """
    Custom Transformer decoder for time series generation from context vector.
    
    Takes a single context vector and generates a fixed-length time series
    using autoregressive generation with cross-attention.
    """
    
    def __init__(
        self,
        context_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        prediction_length: int = 24,
        time_series_dim: int = 1,
    ):
        super().__init__()
        
        self.context_dim = context_dim
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.time_series_dim = time_series_dim
        
        # Project context vector to d_model dimension for cross-attention
        self.context_projection = nn.Linear(context_dim, d_model)
        
        # Embedding layer for time series values
        self.value_embedding = nn.Linear(time_series_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=prediction_length)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # Use (context_length, batch, features) format
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection to time series dimension
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, time_series_dim)
        )
        
        # Start token embedding (learnable)
        self.start_token = nn.Parameter(torch.randn(1, 1, time_series_dim))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(sz, sz, device=device)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(
        self, 
        context: torch.Tensor, 
        target_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training with teacher forcing.
        
        Args:
            context: Context vector from image encoding (batch_size, context_dim)
            target_sequence: Target time series for training (batch_size, context_length, time_series_dim)
            
        Returns:
            Predicted time series (batch_size, prediction_length, time_series_dim)
        """
        batch_size = context.size(0)
        device = context.device
        
        # Project context to d_model
        projected_context = self.context_projection(context)  # (batch_size, d_model)
        
        # Create memory for cross-attention (context needs to be in sequence format)
        # We'll use the context as a single "token" in memory
        memory = projected_context.unsqueeze(0)  # (1, batch_size, d_model)
        
        if self.training and target_sequence is not None:
            # Training mode with teacher forcing
            context_length = target_sequence.size(1)
            
            # Create input sequence by prepending start token
            start_tokens = self.start_token.expand(batch_size, 1, self.time_series_dim)
            if context_length > 1:
                # Use shifted target sequence (exclude last token for input)
                decoder_input = torch.cat([start_tokens, target_sequence[:, :-1, :]], dim=1)
            else:
                decoder_input = start_tokens
            
            # Embed decoder input
            embedded_input = self.value_embedding(decoder_input)  # (batch_size, context_length, d_model)
            
            # Transpose for transformer: (context_length, batch_size, d_model)
            embedded_input = embedded_input.transpose(0, 1)
            
            # Add positional encoding
            embedded_input = self.pos_encoding(embedded_input)
            
            # Create causal mask
            tgt_mask = self._generate_square_subsequent_mask(embedded_input.size(0), device)
            
            # Pass through transformer decoder
            decoder_output = self.transformer_decoder(
                tgt=embedded_input,
                memory=memory,
                tgt_mask=tgt_mask
            )  # (context_length, batch_size, d_model)
            
            # Project to output dimension
            predictions = self.output_projection(decoder_output)  # (context_length, batch_size, time_series_dim)
            
            # Transpose back to batch first: (batch_size, context_length, time_series_dim)
            predictions = predictions.transpose(0, 1)
            
            return predictions
        else:
            # Inference mode - autoregressive generation
            return self.generate(context)
    
    def generate(self, context: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate time series autoregressively from context.
        
        Args:
            context: Context vector (batch_size, context_dim)
            temperature: Temperature for generation (not used for deterministic output)
            
        Returns:
            Generated time series (batch_size, prediction_length, time_series_dim)
        """
        batch_size = context.size(0)
        device = context.device
        
        # Project context
        projected_context = self.context_projection(context)
        memory = projected_context.unsqueeze(0)  # (1, batch_size, d_model)
        
        # Initialize with start token
        generated_sequence = []
        current_input = self.start_token.expand(batch_size, 1, self.time_series_dim)
        
        for step in range(self.prediction_length):
            # Embed current input sequence
            embedded_input = self.value_embedding(current_input)  # (batch_size, step+1, d_model)
            embedded_input = embedded_input.transpose(0, 1)  # (step+1, batch_size, d_model)
            
            # Add positional encoding
            embedded_input = self.pos_encoding(embedded_input)
            
            # Create causal mask
            context_length = embedded_input.size(0)
            tgt_mask = self._generate_square_subsequent_mask(context_length, device)
            
            # Pass through transformer decoder
            decoder_output = self.transformer_decoder(
                tgt=embedded_input,
                memory=memory,
                tgt_mask=tgt_mask
            )  # (step+1, batch_size, d_model)
            
            # Get the last timestep output
            last_output = decoder_output[-1:, :, :]  # (1, batch_size, d_model)
            
            # Project to time series dimension
            next_value = self.output_projection(last_output)  # (1, batch_size, time_series_dim)
            next_value = next_value.transpose(0, 1)  # (batch_size, 1, time_series_dim)
            
            # Add to generated sequence
            generated_sequence.append(next_value)
            
            # Update current input for next iteration
            current_input = torch.cat([current_input, next_value], dim=1)
        
        # Concatenate all generated values
        full_sequence = torch.cat(generated_sequence, dim=1)  # (batch_size, prediction_length, time_series_dim)
        
        return full_sequence


class ViTToTimeSeriesModel(nn.Module):
    """
    Custom architecture combining ViT encoder with Transformer decoder.
    
    Uses CORAL (Correlation Alignment) for domain bridging between
    computer vision and time series domains.
    """
    
    def __init__(
        self,
        vit_model_name: str = "google/vit-base-patch16-224",
        image_size: int = 64,
        num_channels: int = 3,
        prediction_length: int = 24,
        context_length: int = 48,  # Not used but kept for compatibility
        feature_projection_dim: int = 256,
        time_series_dim: int = 1,
        ts_model_dim: int = 256,
        ts_num_heads: int = 8,
        ts_num_layers: int = 4,
        ts_dim_feedforward: int = 1024,
        ts_dropout: float = 0.1,
        image_mean: list = [0.485, 0.456, 0.406],
        image_std: list = [0.229, 0.224, 0.225],
    ):
        """
        Initialize the model.
        
        Args:
            vit_model_name: Pretrained ViT model name
            prediction_length: Length of time series to predict
            context_length: Not used but kept for compatibility
            feature_projection_dim: Dimension for CORAL projection (context vector size)
            time_series_dim: Dimension of time series (usually 1 for univariate)
            ts_model_dim: Hidden dimension for time series transformer
            ts_num_heads: Number of attention heads
            ts_num_layers: Number of decoder layers
            ts_dim_feedforward: Feed-forward dimension
            ts_dropout: Dropout rate
            image_mean: Mean values for image normalization
            image_std: Std values for image normalization
        """
        super().__init__()
        
        # Store configuration
        self.vit_model_name = vit_model_name
        self.prediction_length = prediction_length
        self.context_length = context_length  # Kept for compatibility
        self.time_series_dim = time_series_dim
        self.feature_projection_dim = feature_projection_dim
        self.ts_model_dim = ts_model_dim
        
        # Image normalization parameters
        self.register_buffer('image_mean', torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).view(1, 3, 1, 1))
        
        # ViT Encoder
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        config.image_size = image_size
        config.patch_size = 4
        config.num_channels = num_channels
        self.vit_encoder = ViTModel(config=config)
        
        # CORAL Domain Bridge
        vit_hidden_size = self.vit_encoder.config.hidden_size
        self.domain_bridge = CorrelationAlignment(
            input_dim=vit_hidden_size,
            output_dim=feature_projection_dim,
            use_bias=False
        )
        
        # Custom Transformer Decoder for Time Series Generation
        self.ts_decoder = TimeSeriesTransformerDecoder(
            context_dim=feature_projection_dim,
            d_model=ts_model_dim,
            nhead=ts_num_heads,
            num_layers=ts_num_layers,
            dim_feedforward=ts_dim_feedforward,
            dropout=ts_dropout,
            prediction_length=prediction_length,
            time_series_dim=time_series_dim,
        )
    
    def encode_image_to_context(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Convert image to context vector using ViT encoder and CORAL bridging.
        
        Args:
            pixel_values: Image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Context vector of shape (batch_size, feature_projection_dim)
        """
        # ViT encoding
        vit_outputs = self.vit_encoder(pixel_values=pixel_values)
        image_features = vit_outputs.last_hidden_state[:, 0]  # Use CLS token
        
        # CORAL domain bridging
        context = self.domain_bridge(image_features)
        
        return context
    
    def forward(
        self,
        context_values: torch.Tensor,
        target_sequences: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        device = next(self.parameters()).device
        
        # Compute Spectra of items by items in the batch
        # They do not have gradient flowing through
        spectra_list = []
        for item in context_values.numpy():
            spectra = torch.from_numpy(get_STFT_spectra(item))
            spectra = pad_to_64_center_bottom(spectra).to(device)
            spectra_list.append(spectra)

        # Stack back into batch tensor
        spectra_tensor = torch.stack(spectra_list, dim=0)
        
        # Generate context vector from spectrograms
        context = self.encode_image_to_context(spectra_tensor)
        
        # Generate time series from context using transformer decoder
        predictions = self.ts_decoder(context, target_sequences)
        
        return predictions
    
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
    **kwargs
) -> ViTToTimeSeriesModel:
    """
    Factory function to create ViTToTimeSeriesModel with common configurations.
    
    Args:
        vit_model: ViT model variant to use
        prediction_length: Length of predictions
        context_length: Not used but kept for compatibility
        **kwargs: Additional model arguments
        
    Returns:
        Initialized model
    """
    return ViTToTimeSeriesModel(
        vit_model_name=vit_model,
        prediction_length=prediction_length,
        context_length=context_length,
        **kwargs
    )