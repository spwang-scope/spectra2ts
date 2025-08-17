# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np

def get_STFT_spectra(tensor_data, target_width=None, device='cuda') -> torch.Tensor:
    """
    Generate STFT spectrograms with configurable width using PyTorch.
    
    Args:
        tensor_data: Input time series data tensor of shape (time_steps, features) or numpy array
        target_width: Desired width of spectrogram (should match context_length)
                     If None, uses automatic calculation
        device: Device to run computations on ('cuda' or 'cpu')
    
    Returns:
        Spectrogram tensor of shape (num_features, 128, 128)
    """
    
    # Convert to tensor if numpy array
    if isinstance(tensor_data, np.ndarray):
        tensor_data = torch.from_numpy(tensor_data).float()
    
    # Move to specified device
    tensor_data = tensor_data.to(device)
    
    # Transpose to get shape (features, time_steps)
    tensor_data = tensor_data.T
    
    # Get time series length
    time_length = tensor_data.shape[1]
    
    # If target_width is specified, use it; otherwise use time_length
    if target_width is None:
        target_width = time_length
    
    # Calculate STFT parameters
    n_fft = 64  # This gives us 33 frequency bins (we'll use 32)
    
    # Calculate nperseg and noverlap to achieve target_width time frames (same logic as original)
    if target_width >= time_length:
        # If target width is larger than signal, use small window
        nperseg = min(16, time_length)
        noverlap = 0
    else:
        # Calculate appropriate window size
        nperseg = min(time_length // 4, 64)  # Don't make window too large
        # Calculate noverlap to get approximately target_width frames
        # Rearranging the formula: noverlap = nperseg - (signal_length - nperseg) / (target_width - 1)
        if target_width > 1:
            noverlap = int(nperseg - (time_length - nperseg) / (target_width - 1))
            noverlap = max(0, min(noverlap, nperseg - 1))
        else:
            noverlap = 0
    
    # Convert to PyTorch parameters
    win_length = nperseg
    hop_length = nperseg - noverlap
    
    # Create window tensor on the same device
    window = torch.hann_window(win_length, device=device)
    
    # Initialize list to store spectra
    spectra_list = []
    
    # Process each feature (row in tensor_data)
    for i in range(tensor_data.shape[0]):
        signal_tensor = tensor_data[i]
        
        # Apply STFT
        stft_result = torch.stft(
            signal_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            pad_mode='constant',
            return_complex=True
        )
        
        # Get magnitude spectrum
        spec = torch.abs(stft_result)
        
        # Take only first 32 frequency bins
        spec = spec[:32, :]
        
        # Resize time dimension to exactly target_width using interpolation
        if spec.shape[1] != target_width:
            # Use PyTorch's interpolate function
            spec_expanded = spec.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 32, time)
            
            # Interpolate to target width
            spec_resized = F.interpolate(
                spec_expanded,
                size=(32, target_width),
                mode='bilinear',
                align_corners=False
            )
            
            spec = spec_resized[0, 0]  # Direct indexing instead of squeeze
        
        # Ensure the shape is exactly (32, target_width)
        if spec.shape[0] > 32:
            spec = spec[:32, :]
        elif spec.shape[0] < 32:
            # Pad with zeros if needed
            padding_size = 32 - spec.shape[0]
            padding = torch.zeros(padding_size, spec.shape[1], device=device)
            spec = torch.cat([spec, padding], dim=0)
        
        # Resize spectrum to 128x128 using bilinear interpolation
        spec_expanded = spec.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 32, target_width)
        spec_final = F.interpolate(
            spec_expanded,
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        )
        spec_final = spec_final[0, 0]  # Direct indexing instead of squeeze
        
        spectra_list.append(spec_final)
    
    # Stack all spectra along first dimension
    spectra = torch.stack(spectra_list, dim=0)
    
    return spectra


def get_STFT_spectra_rectangular(tensor_data, context_length, device='cuda') -> torch.Tensor:
    """
    Generate rectangular STFT spectrograms optimized for specific context lengths using PyTorch.
    
    Args:
        tensor_data: Input time series data tensor of shape (time_steps, features) or numpy array
        context_length: Desired width of spectrogram (96, 192, 336, or 720)
        device: Device to run computations on ('cuda' or 'cpu')
    
    Returns:
        Spectrogram tensor of shape (num_features, 128, 128)
    """
    return get_STFT_spectra(tensor_data, target_width=context_length, device=device)
