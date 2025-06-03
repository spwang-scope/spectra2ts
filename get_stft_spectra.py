# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler

def get_STFT_spectra(np_array_data, nperseg = 32, noverlap = 28) -> np.array:
    
    # Since we want to iterate column by column, we could only firstly transpose it,
    # then iterate row by row
    np_array_data = np_array_data.T 

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
