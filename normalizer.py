import torch
import torch.nn.functional as F

class TensorNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, tensor):
        """Calculate mean and std from the tensor"""
        self.mean = tensor.mean(dim=0, keepdim=True)
        self.std = tensor.std(dim=0, keepdim=True)
        return self
    
    def normalize(self, tensor):
        """Apply forward transform (normalize)"""
        if self.mean is None or self.std is None:
            raise ValueError("Must call fit() first")
        return (tensor - self.mean) / self.std
    
    def denormalize(self, tensor):
        """Apply inverse transform (denormalize)"""
        if self.mean is None or self.std is None:
            raise ValueError("Must call fit() first")
        return tensor * self.std + self.mean