"""
Dataset Module for ViT-to-TimeSeries Model

Handles image preprocessing, time series data loading, and batch collation.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import os
import json


class ImageTimeSeriesDataset(Dataset):
    """
    Dataset for image-to-time-series prediction tasks.
    
    Each sample contains:
    - An image that encodes information about time series
    - A corresponding target time series sequence
    """
    
    def __init__(
        self,
        image_paths: List[str],
        time_series_data: List[np.ndarray],
        image_size: int = 224,
        prediction_length: int = 24,
        context_length: int = 48,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        augment: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of paths to images
            time_series_data: List of time series arrays
            image_size: Size to resize images to
            prediction_length: Length of time series to predict
            context_length: Length of context (not used for images but kept for compatibility)
            mean: ImageNet normalization mean
            std: ImageNet normalization std
            augment: Whether to apply data augmentation
        """
        assert len(image_paths) == len(time_series_data), "Images and time series must have same length"
        
        self.image_paths = image_paths
        self.time_series_data = time_series_data
        self.prediction_length = prediction_length
        self.context_length = context_length
        
        # Image preprocessing transforms
        self.transform = self._build_transforms(image_size, mean, std, augment)
        
        # Validate and preprocess time series data
        self._validate_time_series()
    
    def _build_transforms(self, image_size: int, mean: List[float], std: List[float], augment: bool):
        """Build image preprocessing transforms."""
        transform_list = []
        
        # Resize and convert to tensor
        transform_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        # Data augmentation for training
        if augment:
            transform_list.insert(-1, transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ))
            transform_list.insert(-1, transforms.RandomHorizontalFlip(p=0.5))
        
        # Normalization (ImageNet statistics by default)
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)
    
    def _validate_time_series(self):
        """Validate and preprocess time series data."""
        processed_data = []
        
        for i, ts in enumerate(self.time_series_data):
            # Convert to numpy array if not already
            if not isinstance(ts, np.ndarray):
                ts = np.array(ts)
            
            # Ensure minimum length
            if len(ts) < self.prediction_length:
                # Pad with zeros if too short
                padded_ts = np.zeros(self.prediction_length)
                padded_ts[:len(ts)] = ts
                ts = padded_ts
            
            # Take the last prediction_length points as target
            target_sequence = ts[-self.prediction_length:]
            
            # Ensure it's 2D (sequence_length, features)
            if target_sequence.ndim == 1:
                target_sequence = target_sequence.reshape(-1, 1)
            
            processed_data.append(target_sequence.astype(np.float32))
        
        self.time_series_data = processed_data
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - 'pixel_values': Preprocessed image tensor
            - 'target_sequence': Target time series sequence
            - 'image_path': Original image path (for debugging)
        """
        try:
            # Load and preprocess image
            image_path = self.image_paths[idx]
            image = self._load_image(image_path)
            pixel_values = self.transform(image)
            
            # Get target time series
            target_sequence = torch.from_numpy(self.time_series_data[idx])
            
            return {
                'pixel_values': pixel_values,
                'target_sequence': target_sequence,
                'image_path': image_path
            }
        except Exception as e:
            print(f"Error loading sample {idx} (image: {self.image_paths[idx]}): {str(e)}")
            # Return a dummy sample in case of error
            dummy_pixel_values = torch.zeros(3, 224, 224)
            dummy_target = torch.zeros(self.prediction_length, 1)
            return {
                'pixel_values': dummy_pixel_values,
                'target_sequence': dummy_target,
                'image_path': f"error_{idx}"
            }
    
    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load and ensure image has correct color channels.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image in RGB format
        """
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # Handle transparency by creating white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                    image = background
                else:
                    # Convert grayscale or other modes to RGB
                    image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {str(e)}")


class TimeSeriesCollator:
    """
    Custom collator for batching image-time series data.
    
    Handles variable-length sequences and ensures proper tensor shapes.
    """
    
    def __init__(self, pad_value: float = 0.0):
        """
        Initialize collator.
        
        Args:
            pad_value: Value to use for padding sequences
        """
        self.pad_value = pad_value
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched data dictionary
        """
        if not batch:
            raise ValueError("Empty batch provided to collator")
            
        batch_size = len(batch)
        
        # Stack pixel values (all should be same size after preprocessing)
        pixel_values = torch.stack([sample['pixel_values'] for sample in batch])
        
        # Handle time series sequences
        target_sequences = [sample['target_sequence'] for sample in batch]
        
        # Ensure all sequences have valid shapes
        valid_sequences = []
        for i, seq in enumerate(target_sequences):
            if seq.numel() == 0:  # Empty tensor
                print(f"Warning: Empty sequence at index {i}, creating dummy sequence")
                # Create a dummy sequence with proper shape
                seq = torch.zeros(24, 1)  # Default prediction_length=24, features=1
            valid_sequences.append(seq)
        
        target_sequences = self._pad_sequences(valid_sequences)
        
        # Collect metadata
        image_paths = [sample['image_path'] for sample in batch]
        
        return {
            'pixel_values': pixel_values,
            'target_sequences': target_sequences,
            'image_paths': image_paths,
            'batch_size': batch_size
        }
    
    def _pad_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """
        Pad sequences to same length.
        
        Args:
            sequences: List of sequence tensors
            
        Returns:
            Padded sequence tensor of shape (batch_size, max_length, features)
        """
        if not sequences:
            raise ValueError("Empty sequence list provided for padding")
        
        # Ensure all sequences have at least 2 dimensions
        processed_sequences = []
        for seq in sequences:
            if seq.dim() == 1:
                seq = seq.unsqueeze(-1)  # Add feature dimension
            elif seq.dim() == 0:
                seq = seq.unsqueeze(0).unsqueeze(-1)  # Add both dimensions
            processed_sequences.append(seq)
        
        # Find maximum length and feature dimension
        max_length = max(seq.size(0) for seq in processed_sequences)
        feature_dim = processed_sequences[0].size(-1)
        batch_size = len(processed_sequences)
        
        # Create padded tensor
        padded = torch.full(
            (batch_size, max_length, feature_dim),
            self.pad_value,
            dtype=processed_sequences[0].dtype
        )
        
        # Fill in actual sequences
        for i, seq in enumerate(processed_sequences):
            seq_len = seq.size(0)
            padded[i, :seq_len] = seq
        
        return padded


def create_dummy_dataset(
    num_samples: int = 100,
    image_size: int = 224,
    prediction_length: int = 24,
    save_dir: Optional[str] = None
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Create dummy dataset for testing.
    
    Args:
        num_samples: Number of samples to create
        image_size: Size of dummy images
        prediction_length: Length of time series
        save_dir: Directory to save dummy images (if None, creates in-memory data)
        
    Returns:
        Tuple of (image_paths, time_series_data)
    """
    image_paths = []
    time_series_data = []
    
    for i in range(num_samples):
        # Create dummy image data
        if save_dir:
            # Save actual dummy images
            os.makedirs(save_dir, exist_ok=True)
            dummy_image = Image.fromarray(
                np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
            )
            image_path = os.path.join(save_dir, f"dummy_image_{i:04d}.png")
            dummy_image.save(image_path)
            image_paths.append(image_path)
        else:
            # Use placeholder paths (will need actual images for real use)
            image_paths.append(f"dummy_path_{i}")
        
        # Create dummy time series (trending upward with noise)
        trend = np.linspace(0, 10, prediction_length)
        noise = np.random.normal(0, 0.5, prediction_length)
        ts = trend + noise + np.random.uniform(10, 50)  # Add random offset
        
        time_series_data.append(ts)
    
    return image_paths, time_series_data


def create_dataloader(
    image_paths: List[str],
    time_series_data: List[np.ndarray],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = False,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for the image-time series dataset.
    
    Args:
        image_paths: List of image file paths
        time_series_data: List of time series arrays
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        augment: Whether to apply data augmentation
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        DataLoader instance
    """
    try:
        dataset = ImageTimeSeriesDataset(
            image_paths=image_paths,
            time_series_data=time_series_data,
            augment=augment,
            **dataset_kwargs
        )
        
        collator = TimeSeriesCollator()
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
            drop_last=False  # Don't drop incomplete batches
        )
        
        return dataloader
        
    except Exception as e:
        print(f"Error creating dataloader: {str(e)}")
        raise


if __name__ == "__main__":
    # Test dataset creation and loading
    print("Creating dummy dataset...")
    image_paths, time_series_data = create_dummy_dataset(
        num_samples=50,
        save_dir="./dummy_images"
    )
    
    print("Creating dataloader...")
    dataloader = create_dataloader(
        image_paths=image_paths,
        time_series_data=time_series_data,
        batch_size=4,
        augment=True
    )
    
    print("Testing dataloader...")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Pixel values shape: {batch['pixel_values'].shape}")
        print(f"  Target sequences shape: {batch['target_sequences'].shape}")
        print(f"  Batch size: {batch['batch_size']}")
        
        if i == 2:  # Test first few batches
            break
    
    print("Dataset test completed successfully!")
