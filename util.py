import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

def pad_to_multiple_of_4_center_bottom(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Pads each (x, y) channel in a 3D tensor of shape (c, x, y) to shape (c, a, b),
    where a and b are the smallest multiples of 4 ≥ x and y, respectively.
    The original (x, y) slices are aligned to the center-bottom of each (a, b) channel.

    Args:
        input_tensor (torch.Tensor): A 3D tensor of shape (c, x, y)

    Returns:
        torch.Tensor: Zero-padded tensor of shape (c, a, b)
    """
    c, x, y = input_tensor.shape

    # Target dimensions (a, b) as nearest greater or equal multiples of 4
    a = ((x + 3) // 4) * 4
    b = ((y + 3) // 4) * 4

    pad_bottom = 0  # bottom-aligned
    pad_top = a - x

    total_pad_w = b - y
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    # Padding format for 3D tensor is (last_dim_left, last_dim_right, second_last_dim_top, second_last_dim_bottom)
    # So for shape (c, x, y), we pad (left, right, top, bottom) on dims 2 and 1 respectively
    padded_tensor = F.pad(input_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return padded_tensor

def pad_to_64_center_bottom(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Pads a (c, x, y) tensor to shape (c, 64, 64) with zero-padding.
    The original tensor is aligned to the center-bottom of the output.

    Args:
        input_tensor (torch.Tensor): A 3D tensor of shape (c, x, y)

    Returns:
        torch.Tensor: Zero-padded tensor of shape (c, 64, 64)
    """
    c, x, y = input_tensor.shape

    if x > 64 or y > 64:
        raise ValueError(f"Input tensor dimensions (x={x}, y={y}) must be ≤ 64.")

    pad_top = 64 - x  # push original tensor to the bottom
    pad_bottom = 0

    pad_total_width = 64 - y
    pad_left = pad_total_width // 2
    pad_right = pad_total_width - pad_left

    padded_tensor = F.pad(input_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return padded_tensor

def visual(batch_x, batch_y, pred, save_path):
    """
    Visualizes the context, ground truth, and prediction for a time series batch.
    
    Args:
        batch_x (np.ndarray): Context sequence of shape (seq_len,)
        batch_y (np.ndarray): Ground truth of prediction horizon (same length as pred)
        pred (np.ndarray): Model prediction (same shape as batch_y)
        save_path (str): Output file path for saving the figure
    """
    # Ensure all are 1D arrays
    batch_x = batch_x.flatten()
    batch_y = batch_y.flatten()
    pred = pred.flatten()

    context_len = len(batch_x)
    horizon_len = len(batch_y)

    # Construct x-axis timeline
    x_context = np.arange(context_len)
    x_future = np.arange(context_len, context_len + horizon_len)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(x_context, batch_x, color='blue', label='Context')
    plt.plot(x_future, batch_y, color='blue', linestyle='--', label='Ground Truth')
    plt.plot(x_future, pred, color='red', linestyle='--', label='Prediction')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Time Series Forecasting')
    plt.legend()
    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()