"""
Main Training and Testing Script for ViT-to-TimeSeries Model

Handles training, evaluation, and inference with configurable arguments.
"""

import argparse
import os
import json
import time
from typing import Dict, Any, Optional
import logging
import glob
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Check transformers version compatibility
try:
    import transformers
    transformers_version = transformers.__version__
    print(f"Using transformers version: {transformers_version}")
except ImportError:
    raise ImportError("transformers library is required. Install with: pip install transformers")

from model import ViTToTimeSeriesModel, create_model
from dataset import create_dataloader, create_dummy_dataset, ImageTimeSeriesDataset
from bridge import CorrelationAlignment


def setup_logging(log_dir: str = "./logs", log_level: str = "INFO"):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ViT-to-TimeSeries Model Training")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Model arguments
    parser.add_argument("--vit_model", type=str, default="google/vit-base-patch16-224",
                       help="ViT model name from HuggingFace")
    parser.add_argument("--prediction_length", type=int, default=24,
                       help="Length of time series to predict")
    parser.add_argument("--context_length", type=int, default=48,
                       help="Not used but kept for compatibility")
    parser.add_argument("--feature_projection_dim", type=int, default=256,
                       help="Dimension for CORAL feature projection")
    parser.add_argument("--time_series_dim", type=int, default=1,
                       help="Dimension of time series (1 for univariate)")
    parser.add_argument("--ts_model_dim", type=int, default=256,
                       help="Hidden dimension for transformer decoder")
    parser.add_argument("--ts_num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--ts_num_layers", type=int, default=4,
                       help="Number of decoder layers")
    parser.add_argument("--ts_dim_feedforward", type=int, default=1024,
                       help="Feed-forward dimension")
    parser.add_argument("--ts_dropout", type=float, default=0.1,
                       help="Dropout rate for transformer")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay for optimizer")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                       help="Number of warmup epochs")
    parser.add_argument("--scheduler", type=str, default="cosine",
                       choices=["cosine", "step", "linear"],
                       help="Learning rate scheduler")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="../dataset/ETT-small",
                       help="Directory containing dataset")
    parser.add_argument("--data_filename", type=str, default="ETTh1.csv",
                       help="Directory containing dataset")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Size to resize images")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--augment", action="store_true",
                       help="Apply data augmentation")
    parser.add_argument("--use_dummy_data", action="store_true",
                       help="Use dummy data for testing")
    parser.add_argument("--num_dummy_samples", type=int, default=500,
                       help="Number of dummy samples to create")
    
    # Experiment arguments
    parser.add_argument("--experiment_name", type=str, default="vit_timeseries",
                       help="Name of the experiment")
    parser.add_argument("--output_dir", type=str, default=f"./outputs_{timestamp}",
                       help="Output directory for models and logs")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save model every N epochs")
    parser.add_argument("--eval_interval", type=int, default=5,
                       help="Evaluate model every N epochs")
    
    # Mode arguments
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "test", "inference"],
                       help="Mode to run the script")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to model checkpoint for testing/inference")
    parser.add_argument("--freeze_vit", action="store_true",
                       help="Freeze ViT encoder during training")
    parser.add_argument("--freeze_ts", action="store_true",
                       help="Freeze TimeSeries decoder during training")
    parser.add_argument("--no_checkpoint", action="store_true",
                       help="Run inference without loading checkpoint (uses random weights)")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--mixed_precision", action="store_true",
                       help="Use mixed precision training")
    
    # Logging arguments
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--tensorboard", action="store_true",
                       help="Use tensorboard logging")
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_arg)


def get_image_paths_from_dir(data_dir: str) -> list:
    """Get all image paths from data directory."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_paths = []
    
    for extension in image_extensions:
        pattern = os.path.join(data_dir, extension)
        image_paths.extend(glob.glob(pattern))
        # Also check for uppercase extensions
        pattern = os.path.join(data_dir, extension.upper())
        image_paths.extend(glob.glob(pattern))
    
    return sorted(image_paths)


def create_data_loaders(args, logger) -> tuple:
    """Create training and validation data loaders."""
    if args.use_dummy_data:
        # Create dummy dataset
        logger.info("Creating dummy dataset...")
        image_paths, time_series_data = create_dummy_dataset(
            num_samples=args.num_dummy_samples,
            prediction_length=args.prediction_length,
            save_dir=os.path.join(args.output_dir, "dummy_images")
        )
        
        # Split into train/val
        split_idx = int(0.8 * len(image_paths))
        train_images, val_images = image_paths[:split_idx], image_paths[split_idx:]
        train_ts, val_ts = time_series_data[:split_idx], time_series_data[split_idx:]
        
    else:
        # Load real dataset (implement your data loading logic here)
        if args.data_dir is None:
            raise ValueError("data_dir must be specified when not using dummy data")
        
        # TODO: Implement real data loading
        raise NotImplementedError("Real data loading not implemented yet")
    
    # Create data loaders
    logger.info("Creating training dataloader...")
    train_loader = create_dataloader(
        image_paths=train_images,
        time_series_data=train_ts,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        augment=args.augment,
        prediction_length=args.prediction_length,
        image_size=args.image_size,
    )
    
    logger.info("Creating validation dataloader...")
    val_loader = create_dataloader(
        image_paths=val_images,
        time_series_data=val_ts,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        augment=False,
        prediction_length=args.prediction_length,
        image_size=args.image_size,
    )
    
    return train_loader, val_loader


def create_optimizer_and_scheduler(model: nn.Module, args):
    """Create optimizer and learning rate scheduler."""
    # Different learning rates for different components
    param_groups = [
        {"params": model.vit_encoder.parameters(), "lr": args.learning_rate * 0.1},
        {"params": model.domain_bridge.parameters(), "lr": args.learning_rate},
        {"params": model.ts_decoder.parameters(), "lr": args.learning_rate},
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=1e-6
        )
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.num_epochs // 3, gamma=0.1
        )
    elif args.scheduler == "linear":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.num_epochs
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def train_epoch(
    model: ViTToTimeSeriesModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    logger,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        pixel_values = batch['pixel_values'].to(device)
        target_sequences = batch['target_sequences'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        logger.debug(f"Computing forward pass for batch {batch_idx}")
        predictions = model(pixel_values, target_sequences)
        
        # Ensure predictions and targets have same shape
        if predictions.shape != target_sequences.shape:
            logger.warning(f"Shape mismatch: predictions {predictions.shape} vs targets {target_sequences.shape}")
            # Try to align shapes
            if target_sequences.dim() == 2 and predictions.dim() == 3:
                target_sequences = target_sequences.unsqueeze(-1)
            elif target_sequences.dim() == 3 and predictions.dim() == 2:
                predictions = predictions.unsqueeze(-1)
        
        # Compute loss
        loss = nn.functional.mse_loss(predictions, target_sequences)
        
        # Check for NaN or inf losses
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss detected: {loss.item()}, skipping batch")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        batch_size = pixel_values.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Log batch metrics
        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        # Print progress
        if batch_idx % 50 == 0:
            logger.info(
                f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                f'Loss: {loss.item():.6f}'
            )
    
    if total_samples == 0:
        logger.error("No valid batches processed in this epoch!")
        return {'train_loss': float('inf')}
    
    avg_loss = total_loss / total_samples
    
    return {
        'train_loss': avg_loss
    }


def evaluate(
    model: ViTToTimeSeriesModel,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    logger,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    
    logger.info("Starting model evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            pixel_values = batch['pixel_values'].to(device)
            target_sequences = batch['target_sequences'].to(device)
            
            # Forward pass (inference mode)
            logger.debug(f"Processing validation batch {batch_idx}")
            predictions = model(pixel_values)
            
            # Ensure predictions and targets have same shape
            if predictions.shape != target_sequences.shape:
                if target_sequences.dim() == 2 and predictions.dim() == 3:
                    target_sequences = target_sequences.unsqueeze(-1)
                elif target_sequences.dim() == 3 and predictions.dim() == 2:
                    predictions = predictions.unsqueeze(-1)
            
            # Calculate metrics
            loss = nn.functional.mse_loss(predictions, target_sequences)
            mae = torch.mean(torch.abs(predictions - target_sequences))
            
            # Check for valid metrics
            if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(mae) or torch.isinf(mae):
                logger.warning(f"Invalid metrics in validation batch {batch_idx}, skipping")
                continue
            
            batch_size = pixel_values.size(0)
            total_loss += loss.item() * batch_size
            total_mae += mae.item() * batch_size
            total_samples += batch_size
    
    if total_samples == 0:
        logger.error("No valid validation batches processed!")
        return {'val_loss': float('inf'), 'val_mae': float('inf')}
    
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/MAE', avg_mae, epoch)
    
    logger.info(f'Validation - Loss: {avg_loss:.6f}, MAE: {avg_mae:.6f}')
    
    return {
        'val_loss': avg_loss,
        'val_mae': avg_mae
    }


def save_checkpoint(model: ViTToTimeSeriesModel, optimizer: optim.Optimizer, epoch: int, 
                   metrics: Dict[str, float], filepath: str, logger):
    """Save model checkpoint."""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f'Checkpoint saved: {filepath}')
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {filepath}: {str(e)}")


def load_checkpoint(filepath: str, model: ViTToTimeSeriesModel, 
                   optimizer: Optional[optim.Optimizer] = None, logger = None) -> int:
    """Load model checkpoint."""
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        if logger:
            logger.info(f'Checkpoint loaded: {filepath}, epoch: {epoch}')
        
        return epoch
    except Exception as e:
        if logger:
            logger.error(f"Failed to load checkpoint from {filepath}: {str(e)}")
        raise


def train(args):
    """Main training function."""
    # Setup logging
    logger = setup_logging(args.output_dir, args.log_level)
    
    # Setup
    device = get_device(args.device)
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args, logger)
    logger.info(f"Created data loaders: {len(train_loader)} train, {len(val_loader)} val batches")
    
    # Create model
    logger.info("Creating ViT-to-TimeSeries model...")
    model = create_model(
        vit_model=args.vit_model,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        feature_projection_dim=args.feature_projection_dim,
        time_series_dim=args.time_series_dim,
        ts_model_dim=args.ts_model_dim,
        ts_num_heads=args.ts_num_heads,
        ts_num_layers=args.ts_num_layers,
        ts_dim_feedforward=args.ts_dim_feedforward,
        ts_dropout=args.ts_dropout,
    ).to(device)
    logger.info("Model created successfully")
    
    # Print model info
    model_info = model.get_model_info()
    logger.info(f"Model parameters: {model_info['total_parameters']:,}")
    
    # Apply freezing if requested
    if args.freeze_vit:
        logger.info("Freezing ViT encoder...")
        model.freeze_vit_encoder(freeze=True)
        logger.info("ViT encoder frozen")
    
    if args.freeze_ts:
        logger.info("Freezing TimeSeries decoder...")
        model.freeze_ts_decoder(freeze=True)
        logger.info("TimeSeries decoder frozen")
    
    # Create optimizer and scheduler
    logger.info("Creating optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_and_scheduler(model, args)
    
    # Setup tensorboard
    writer = None
    if args.tensorboard:
        logger.info("Setting up TensorBoard logging...")
        writer = SummaryWriter(os.path.join(experiment_dir, 'tensorboard'))
    
    # Training loop
    best_val_loss = float('inf')
    start_epoch = 0
    
    if args.checkpoint_path:
        logger.info(f"Loading checkpoint from {args.checkpoint_path}...")
        start_epoch = load_checkpoint(args.checkpoint_path, model, optimizer, logger)
    
    logger.info(f"Starting training from epoch {start_epoch}")
    logger.info(f"Model info: {model_info}")
    logger.info(f"Device: {device}")
    logger.info(f"Using custom Transformer decoder with transformers v{transformers_version}")
    
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"Starting epoch {epoch}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, args, logger, writer)
        
        # Evaluate
        val_metrics = {}
        if epoch % args.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, device, epoch, logger, writer)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log metrics
        if writer:
            writer.add_scalar('Train/Loss', train_metrics['train_loss'], epoch)
            
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            logger.info(f"Saving checkpoint for epoch {epoch}...")
            metrics = {**train_metrics, **val_metrics}
            checkpoint_path = os.path.join(experiment_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path, logger)
            
            # Save best model
            if val_metrics and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_path = os.path.join(experiment_dir, 'best_model.pt')
                logger.info(f"New best model with validation loss: {best_val_loss:.6f}")
                save_checkpoint(model, optimizer, epoch, metrics, best_path, logger)
    
    # Final save
    logger.info("Saving final model...")
    final_path = os.path.join(experiment_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, args.num_epochs - 1, {}, final_path, logger)
    
    if writer:
        writer.close()
    
    logger.info("Training completed!")


def test(args):
    """Test the model."""
    # Setup logging
    logger = setup_logging(args.output_dir, args.log_level)
    
    device = get_device(args.device)
    
    if not args.checkpoint_path:
        raise ValueError("checkpoint_path must be specified for testing")
    
    # Create data loaders
    _, val_loader = create_data_loaders(args, logger)
    
    # Create and load model
    logger.info("Creating model for testing...")
    model = create_model(
        vit_model=args.vit_model,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        feature_projection_dim=args.feature_projection_dim,
        time_series_dim=args.time_series_dim,
        ts_model_dim=args.ts_model_dim,
        ts_num_heads=args.ts_num_heads,
        ts_num_layers=args.ts_num_layers,
        ts_dim_feedforward=args.ts_dim_feedforward,
        ts_dropout=args.ts_dropout,
    ).to(device)
    logger.info("Model created successfully")
    
    load_checkpoint(args.checkpoint_path, model, logger=logger)
    
    # Evaluate
    logger.info("Starting evaluation...")
    metrics = evaluate(model, val_loader, device, 0, logger)
    
    logger.info("Test Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.6f}")


def inference(args):
    '''
    """Run inference on new data."""
    # Setup logging
    logger = setup_logging(args.output_dir, args.log_level)
    
    device = get_device(args.device)
    
    if not args.no_checkpoint and not args.checkpoint_path:
        raise ValueError("checkpoint_path must be specified for inference, or use --no_checkpoint for trial run")
    
    # Create model
    logger.info("Creating model for inference...")
    model = create_model(
        vit_model=args.vit_model,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        feature_projection_dim=args.feature_projection_dim,
        time_series_dim=args.time_series_dim,
        ts_model_dim=args.ts_model_dim,
        ts_num_heads=args.ts_num_heads,
        ts_num_layers=args.ts_num_layers,
        ts_dim_feedforward=args.ts_dim_feedforward,
        ts_dropout=args.ts_dropout,
    ).to(device)
    logger.info("Model created successfully")
    
    # Load checkpoint if provided
    if args.checkpoint_path and not args.no_checkpoint:
        try:
            load_checkpoint(args.checkpoint_path, model, logger=logger)
            logger.info("Using pretrained checkpoint")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            logger.info("Continuing with randomly initialized model")
    else:
        logger.info("Using randomly initialized model (trial run)")
    
    model.eval()
    
    # Create output directory for predictions
    inference_dir = os.path.join(args.output_dir, "inference_results")
    os.makedirs(inference_dir, exist_ok=True)
    
    # Get image paths from data directory
    if os.path.exists(args.data_dir):
        logger.info(f"Searching for images in {args.data_dir}...")
        image_paths = get_image_paths_from_dir(args.data_dir)
        logger.info(f"Found {len(image_paths)} images in {args.data_dir}")
        
        if len(image_paths) == 0:
            logger.warning(f"No images found in {args.data_dir}. Creating dummy data for testing.")
            # Create dummy data for testing
            image_paths, _ = create_dummy_dataset(
                num_samples=10,
                prediction_length=args.prediction_length,
                save_dir=os.path.join(args.output_dir, "inference_images")
            )
    else:
        logger.warning(f"Data directory {args.data_dir} not found. Creating dummy data for testing.")
        # Create dummy data for testing
        image_paths, _ = create_dummy_dataset(
            num_samples=10,
            prediction_length=args.prediction_length,
            save_dir=os.path.join(args.output_dir, "inference_images")
        )
    
    # Run inference on images
    predictions = []
    image_names = []
    failed_images = []
    
    logger.info("Starting inference...")
    
    # Process images in batches
    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i+args.batch_size]
        batch_num = i // args.batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{(len(image_paths) + args.batch_size - 1) // args.batch_size}")
        
        # Create dummy time series data for batch (required by dataset)
        dummy_ts = [np.zeros(args.prediction_length) for _ in batch_paths]
        
        # Create dataloader for this batch
        try:
            logger.debug(f"Creating dataloader for batch {batch_num} with {len(batch_paths)} images")
            batch_loader = create_dataloader(
                image_paths=batch_paths,
                time_series_data=dummy_ts,
                batch_size=len(batch_paths),
                shuffle=False,
                num_workers=0,  # Use 0 workers for small batches
                augment=False,
                prediction_length=args.prediction_length,
                image_size=args.image_size,
            )
            logger.debug(f"Dataloader created successfully for batch {batch_num}")
        except Exception as e:
            logger.error(f"Failed to create dataloader for batch {batch_num}: {str(e)}")
            failed_batch_names = [os.path.basename(path) for path in batch_paths]
            failed_images.extend(failed_batch_names)
            continue
        
        # Process batch
        with torch.no_grad():
            for batch in batch_loader:
                logger.debug(f"Processing batch {batch_num} - pixel_values shape: {batch['pixel_values'].shape}")
                pixel_values = batch['pixel_values'].to(device)
                
                # Generate predictions
                logger.debug(f"Generating predictions for batch {batch_num}...")
                pred = model.generate(pixel_values, num_samples=1)
                logger.debug(f"Generated predictions shape: {pred.shape}")
                
                # Validate predictions
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    logger.warning(f"Invalid predictions in batch {batch_num}, using zeros")
                    pred = torch.zeros_like(pred)
                
                predictions.append(pred.cpu().numpy())
                
                # Store image names for reference
                batch_names = [os.path.basename(path) for path in batch['image_paths']]
                image_names.extend(batch_names)
                
                logger.info(f"Successfully processed batch {batch_num}: {len(batch_names)} images")
                
                break  # Only one batch per dataloader
    
    # Concatenate all predictions
    if predictions:
        logger.info("Concatenating all predictions...")
        all_predictions = np.concatenate(predictions, axis=0)
        
        # Save predictions
        try:
            logger.info("Saving predictions...")
            np.save(os.path.join(inference_dir, "predictions.npy"), all_predictions)
            
            # Save image names for reference
            with open(os.path.join(inference_dir, "image_names.txt"), 'w') as f:
                for name in image_names:
                    f.write(f"{name}\n")
            
            # Save failed images list
            if failed_images:
                with open(os.path.join(inference_dir, "failed_images.txt"), 'w') as f:
                    for name in failed_images:
                        f.write(f"{name}\n")
                logger.warning(f"{len(failed_images)} images failed to process")
            
            # Save summary
            summary = {
                "num_images_processed": len(image_names),
                "num_images_failed": len(failed_images),
                "total_images_attempted": len(image_paths),
                "prediction_length": args.prediction_length,
                "predictions_shape": all_predictions.shape,
                "predictions_stats": {
                    "mean": float(all_predictions.mean()),
                    "std": float(all_predictions.std()),
                    "min": float(all_predictions.min()),
                    "max": float(all_predictions.max())
                },
                "model_config": {
                    "vit_model": args.vit_model,
                    "prediction_length": args.prediction_length,
                    "context_length": args.context_length,
                    "feature_projection_dim": args.feature_projection_dim,
                    "used_checkpoint": not args.no_checkpoint and args.checkpoint_path is not None
                }
            }
            
            with open(os.path.join(inference_dir, "summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Inference completed! Results saved to {inference_dir}")
            logger.info(f"Successfully processed {len(image_names)} images")
            logger.info(f"Failed to process {len(failed_images)} images")
            logger.info(f"Predictions shape: {all_predictions.shape}")
            logger.info(f"Sample prediction stats: mean={all_predictions.mean():.4f}, std={all_predictions.std():.4f}")
            
        except Exception as e:
            logger.error(f"Failed to save inference results: {str(e)}")
            
    else:
        logger.error("No predictions generated! All batches failed.")
        # Still save a summary of the failure
        try:
            summary = {
                "num_images_processed": 0,
                "num_images_failed": len(image_paths),
                "total_images_attempted": len(image_paths),
                "error": "All inference attempts failed"
            }
            with open(os.path.join(inference_dir, "summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save failure summary: {str(e)}")
    '''
    print("pass")
    pass

def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "inference":
        inference(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
