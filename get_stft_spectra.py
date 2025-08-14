# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize

def get_STFT_spectra(np_array_data, target_width=None) -> np.array:
    """
    Generate STFT spectrograms with configurable width.
    
    Args:
        np_array_data: Input time series data of shape (time_steps, features)
        target_width: Desired width of spectrogram (should match context_length)
                     If None, uses automatic calculation
    
    Returns:
        Spectrogram of shape (num_features, 128, 128)
    """
    
    # Since we want to iterate column by column, we could only firstly transpose it,
    # then iterate row by row
    np_array_data = np_array_data.T 
    
    # Get time series length
    time_length = np_array_data.shape[1]
    
    # If target_width is specified, use it; otherwise use time_length
    if target_width is None:
        target_width = time_length
    
    # Calculate STFT parameters to get desired width
    # We want frequency dimension to be 32 (using nfft=64 gives us 33 freq bins, we'll take 32)
    # For the time dimension, we need to calculate nperseg and noverlap
    nfft = 64  # This gives us 33 frequency bins (we'll use 32)
    
    # Calculate nperseg and noverlap to achieve target_width time frames
    # Formula: num_frames = (signal_length - nperseg) / (nperseg - noverlap) + 1
    # We'll use a heuristic approach
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
    
    # Initialize list to store spectra
    spectra_list = []
    
    # Process each feature (row in transposed data)
    for i, column in enumerate(np_array_data):
        # Apply STFT to the signal (removed double normalization)
        f, t, Zxx = signal.stft(
            column, 
            fs=1.0, 
            nperseg=nperseg, 
            noverlap=noverlap, 
            nfft=nfft,
            boundary='zeros',
            padded=True
        )
        
        # Get magnitude spectrum
        spec = np.abs(Zxx)
        
        # Take only first 32 frequency bins (removing DC component if needed)
        spec = spec[:32, :]
        
        # Resize time dimension to exactly target_width using interpolation
        if spec.shape[1] != target_width:
            # Use linear interpolation to resize
            from scipy.interpolate import interp2d
            x_old = np.arange(spec.shape[1])
            y_old = np.arange(spec.shape[0])
            x_new = np.linspace(0, spec.shape[1]-1, target_width)
            y_new = y_old
            
            # Create interpolation function
            f_interp = interp2d(x_old, y_old, spec, kind='linear')
            spec = f_interp(x_new, y_new)
        
        # Ensure the shape is exactly (32, target_width)
        if spec.shape[0] > 32:
            spec = spec[:32, :]
        elif spec.shape[0] < 32:
            # Pad with zeros if needed
            padding = np.zeros((32 - spec.shape[0], spec.shape[1]))
            spec = np.vstack([spec, padding])
        
        # Resize spectrum to 128x128 using bilinear interpolation
        spec = resize(spec, (128, 128), order=1, preserve_range=True)
            
        spectra_list.append(spec)
    
    # Stack all spectra along first dimension
    spectra = np.stack(spectra_list, axis=0)
    
    return spectra


def get_STFT_spectra_rectangular(np_array_data, context_length) -> np.array:
    """
    Generate rectangular STFT spectrograms optimized for specific context lengths.
    
    Args:
        np_array_data: Input time series data of shape (time_steps, features)
        context_length: Desired width of spectrogram (96, 192, 336, or 720)
    
    Returns:
        Spectrogram of shape (num_features, 128, 128)
    """
    return get_STFT_spectra(np_array_data, target_width=context_length)
