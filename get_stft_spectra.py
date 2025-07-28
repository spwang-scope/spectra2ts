# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler

def get_STFT_spectra(np_array_data) -> np.array:

    # Optimized version for scipy targeting 64 frames width
    def get_stft_params(length):
        """
        Calculate nperseg and noverlap for STFT based on input sequence length.
        
        This function follows the balanced approach (Option 1):
        - nperseg ≈ length/3 (rounded to reasonable values)
        - noverlap = 75% of nperseg (for smooth time evolution)
        - Ensures good balance between time and frequency resolution
        
        Args:
            length (int): Length of the input sequence
            
        Returns:
            tuple: (nperseg, noverlap) parameters for scipy.signal.stft
        """
        # Target nperseg as approximately 1/3 of length (based on Option 1: 32/96 ≈ 1/3)
        target_nperseg = length // 3
        
        # Ensure minimum reasonable window size
        min_nperseg = 8
        max_nperseg = length // 2  # Don't exceed half the signal length
        
        # Round to nearest power of 2 for computational efficiency, with some flexibility
        if target_nperseg < min_nperseg:
            nperseg = min_nperseg
        elif target_nperseg > max_nperseg:
            nperseg = max_nperseg
        else:
            # Find nearest power of 2, but allow some flexibility for better fit
            power_of_2 = 2 ** round(np.log2(target_nperseg))
            
            # If the power of 2 is too far from target, use a more flexible approach
            if abs(power_of_2 - target_nperseg) > target_nperseg * 0.3:
                # Use multiples of 8 for reasonable values
                nperseg = max(min_nperseg, (target_nperseg // 8) * 8)
                if nperseg == 0:
                    nperseg = min_nperseg
            else:
                nperseg = power_of_2
        
        # Ensure nperseg is within bounds
        nperseg = max(min_nperseg, min(nperseg, max_nperseg))
        
        # Set noverlap to 75% of nperseg (Option 1 approach)
        noverlap = int(nperseg * 0.75)
        
        # Ensure noverlap is valid (must be less than nperseg)
        noverlap = min(noverlap, nperseg - 1)
        
        return nperseg, noverlap
    
    # Since we want to iterate column by column, we could only firstly transpose it,
    # then iterate row by row
    np_array_data = np_array_data.T 
    
    # compute the STFT parameters based on the first column's length
    nperseg, noverlap = get_stft_params(np_array_data[0].shape[0])

    # Parameters for STFT
    # nperseg: Window length for STFT
    # noverlap: Overlap between windows

    # Initialize Numpy array to store stacking spectra
    spectra_list = list()

    # Process each column
    for i, column in enumerate(np_array_data):  # The code iterates a row, but it's actually a column in original data

        # Standardize the data before STFT
        data_scaler = StandardScaler()
        standardized_values = data_scaler.fit_transform(column.reshape(-1, 1)).flatten()

        # Apply STFT to the standardized signal
        _, _, Zxx_orig = signal.stft(standardized_values, fs=1.0, nperseg=nperseg, noverlap=noverlap)
        spec_orig = np.abs(Zxx_orig)  # Get magnitude

        spectra_list.append(spec_orig)

    # Convert the list of spectra to a NumPy array, which is the desired output
    spectra = np.stack(spectra_list, axis=0)

    return spectra
