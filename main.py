"""
Main Training and Testing Script for ViT-to-TimeSeries Model

Handles training with teacher forcing, evaluation, and inference with autoregressive generation.
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
import numpy as np
import pandas as pd
from data_factory import data_provider
from metrics import metric
from util import visual

from model import ViTToTimeSeriesModel, create_model

logger = logging.getLogger(__name__)


def setup_logging(args):
    """Setup logging configuration."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ViT-to-TimeSeries Model Training")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Model arguments
    parser.add_argument("--prediction_length", type=int, default=96,
                       help="Length of time series to predict")
    parser.add_argument("--context_length", type=int, default=96,
                       help="Length of context window")
    parser.add_argument("--feature_projection_dim", type=int, default=128,
                       help="Dimension for CORAL feature projection")
    parser.add_argument("--time_series_dim", type=int, default=1,
                       help="Dimension of time series (1 for univariate)")
    parser.add_argument("--ts_model_dim", type=int, default=768,
                       help="Hidden dimension for transformer decoder")
    parser.add_argument("--ts_num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--ts_num_layers", type=int, default=3,
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
    parser.add_argument("--num_epochs", type=int, default=200,
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
    parser.add_argument("--image_size", type=int, default=64,
                       help="Height of spectrogram (always 64)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--augment", action="store_true",
                       help="Apply data augmentation")
    parser.add_argument("--use_dummy_data", action="store_true",
                       help="Use dummy data for testing")
    parser.add_argument("--num_dummy_samples", type=int, default=50,
                       help="Number of dummy samples to create")
    
    # Experiment arguments
    parser.add_argument("--experiment_name", type=str, default="vit_timeseries",
                       help="Name of the experiment")
    parser.add_argument("--output_dir", type=str, default=f"./outputs_{timestamp}",
                       help="Output directory for models and logs")
    parser.add_argument("--save_interval", type=int, default=50,
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
    parser.add_argument("--cuda_num", type=int, default=7,
                       help="CUDA device number")

    # Logging arguments
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--tensorboard", action="store_true",
                       help="Use tensorboard logging")
    
    args = parser.parse_args()
    args.target = 'OT'
    
    def get_df_channel():
        df = pd.read_csv(os.path.join(args.data_dir,
                                          args.data_filename))
        return df.shape[1]-1  # number of columns, exclude Datetime
    
    # Determine number of data channels since we want to construct model according to it
    args.num_channels = get_df_channel()
    
    return args


def get_device(device_arg: str, cuda_num: int) -> torch.device:
    """Get the appropriate device."""
    if device_arg == "auto":
        return torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_arg)


def create_optimizer_and_scheduler(model: nn.Module, args):
    """Create optimizer and learning rate scheduler."""
    # Different learning rates for different components
    param_groups = [
        {"params": model.vit_encoder.parameters(), "lr": args.learning_rate},
        # {"params": model.domain_bridge.parameters(), "lr": args.learning_rate},
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
    """Main training function with teacher forcing."""
    
    # Setup
    device = get_device(args.device, args.cuda_num)
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create data loaders
    train_dataset, train_loader = data_provider(args, flag='train')
    test_dataset, test_loader = data_provider(args, flag='test')
    logger.info(f"Created data loaders: {len(train_loader)} trains {len(test_loader)} tests")
    
    # Create model
    logger.info("Creating ViT-to-TimeSeries model with Transformer decoder...")
    model = create_model(
        vit_model=args.vit_model,
        image_size=args.image_size,
        num_channels=args.num_channels,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        feature_projection_dim=args.feature_projection_dim,
        time_series_dim=args.time_series_dim,
        ts_model_dim=args.ts_model_dim,
        ts_num_heads=args.ts_num_heads,
        ts_num_layers=args.ts_num_layers,
        ts_dim_feedforward=args.ts_dim_feedforward,
        ts_dropout=args.ts_dropout,
        use_lstm_decoder=False,  # Always use Transformer
    ).to(device)
    logger.info("Model created successfully")

    logger.info(f"Model architecture: {model}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
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
    criterion = nn.HuberLoss(delta=0.1)  
    
    # Setup tensorboard
    writer = None
    if args.tensorboard:
        logger.info("Setting up TensorBoard logging...")
        writer = SummaryWriter(os.path.join(experiment_dir, 'tensorboard'))
    
    # Training loop
    start_epoch = 0
    
    if args.checkpoint_path:
        logger.info(f"Loading checkpoint from {args.checkpoint_path}...")
        start_epoch = load_checkpoint(args.checkpoint_path, model, optimizer, logger)
    
    logger.info(f"Starting training from epoch {start_epoch}")
    logger.info(f"Device: {device}")
    logger.info("Using custom Transformer decoder with cross-attention and teacher forcing")
    
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"Starting epoch {epoch}/{args.num_epochs}")
        
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # Forward pass with teacher forcing (training mode)
            outputs = model(context=batch_x, tf_target=batch_y, mode='train')

            # Select target feature for loss calculation based on time_series_dim
            outputs = outputs[:, :, :].to(device)
            if hasattr(args, 'time_series_dim') and args.time_series_dim > 1:
                # Multi-variable prediction: use last time_series_dim features
                batch_y = batch_y[:, :args.prediction_length, -args.time_series_dim:].to(device)
            else:
                # Single variable prediction: use last feature only
                batch_y = batch_y[:, :args.prediction_length, -1:].to(device)
            
            # Check for empty tensors
            if outputs.numel() == 0:
                logger.error(f"Empty outputs tensor! Shape: {outputs.shape}")
                continue
            
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                logger.info(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                iter_count = 0

            loss.backward()
            
            # Gradient clipping during warmup
            if epoch < args.warmup_epochs:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

        logger.info(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
        train_loss = np.average(train_loss)
        
        # Evaluate on test set
        test_loss = test(args, peeking=True, model=model, epoch=epoch)
        
        logger.info(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} Test Loss: {test_loss:.7f}")
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            logger.info(f"Saving checkpoint for epoch {epoch + 1}...")
            checkpoint_path = os.path.join(experiment_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            save_checkpoint(model, optimizer, epoch + 1, 
                          {'train_loss': train_loss, 'test_loss': test_loss}, 
                          checkpoint_path, logger)
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
    
    # Final save
    logger.info("Saving final model...")
    final_path = os.path.join(experiment_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, args.num_epochs, {}, final_path, logger)
    
    if writer:
        writer.close()
    
    logger.info("Training completed!")


def test(args, peeking=False, model=None, epoch=None):
    """Test the model using inference mode (no teacher forcing)."""
    
    device = get_device(args.device, args.cuda_num)
    
    if not args.checkpoint_path and not peeking:
        raise ValueError("checkpoint_path must be specified for testing")
    
    # Create data loaders
    test_data, test_loader = data_provider(args, flag='test')
    
    # Create and load model
    if not peeking:
        logger.info("Creating model for testing...")
        model = create_model(
            image_size=args.image_size,
            num_channels=args.num_channels,
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
        logger.info("Model weights loaded successfully") 
    else:
        assert(model is not None)

    preds = []
    trues = []
    criterion = nn.MSELoss()
    total_loss = []
    
    model.eval()  # Set to evaluation mode
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # Use inference mode (no teacher forcing)
            outputs = model.inference(batch_x[:, :args.context_length, :])

            # Calculate loss
            f_dim = -1
            outputs = outputs[:, :, :].to(device)
            batch_y = batch_y[:, -args.prediction_length:, f_dim:].to(device)
            
            loss = criterion(outputs, batch_y)
            
            # Store predictions and ground truth
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            
            preds.append(pred)
            trues.append(true)


            input = batch_x.detach().cpu().numpy()
            gt = np.concatenate((input[0, :args.context_length, -1], true[0, :, -1]), axis=0)
            pd = np.concatenate((input[0, :args.context_length, -1], pred[0, :, -1]), axis=0)
            if peeking:
                if (i % 20 == 0 and (epoch + 1) % 10 == 0):
                    visual(gt, pd, os.path.join(args.output_dir, f'test_vis_{i}_epoch{epoch}.png'))
            else:
                if (i % 10 == 0):
                    visual(gt, pd, os.path.join(args.output_dir, f'test_vis_{i}.png'))

            total_loss.append(loss.item())
            
    avg_loss = np.average(total_loss)

    # Concatenate all predictions and ground truth
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Avoid divide by zero in metrics
    preds[preds == 0] = 1e-6
    trues[trues == 0] = 1e-6
    
    # Reshape for metric calculation
    #preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    print(f"Preds: {preds.shape}, Trues: {trues.shape}")
    print(f"Pred range: {preds.min():.6f} to {preds.max():.6f}")
    print(f"True range: {trues.min():.6f} to {trues.max():.6f}")

    

    # Calculate metrics
    mae, mse, _, _, _ = metric(preds, trues)
    logger.info(f'MSE: {mse:.7f}, MAE: {mae:.7f}')
    
    if peeking:
        model.train()  # Return to training mode
        return avg_loss

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "test_results.txt"), 'w') as f:
        f.write(f'MSE: {mse:.7f}, MAE: {mae:.7f}\n')

    np.save(os.path.join(args.output_dir, 'test_metrics.npy'), np.array([mae, mse]))
    np.save(os.path.join(args.output_dir, 'test_pred.npy'), preds)
    np.save(os.path.join(args.output_dir, 'test_true.npy'), trues)

    return avg_loss


def inference(args):
    """Run inference on new data using autoregressive generation."""
    
    device = get_device(args.device, args.cuda_num)
    
    if not args.no_checkpoint and not args.checkpoint_path:
        raise ValueError("checkpoint_path must be specified for inference, or use --no_checkpoint for trial run")
    
    # Create model
    logger.info("Creating model for inference...")
    model = create_model(
        vit_model=args.vit_model,
        image_size=args.image_size,
        num_channels=args.num_channels,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        feature_projection_dim=args.feature_projection_dim,
        time_series_dim=args.time_series_dim,
        ts_model_dim=args.ts_model_dim,
        ts_num_heads=args.ts_num_heads,
        ts_num_layers=args.ts_num_layers,
        ts_dim_feedforward=args.ts_dim_feedforward,
        ts_dropout=args.ts_dropout,
        use_lstm_decoder=False,
    ).to(device)
    logger.info("Model created successfully with Transformer decoder")
    
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

    # Load data
    data_set, data_loader = data_provider(args, flag='test')
    
    model.eval()
    
    # Create output directory for predictions
    inference_dir = os.path.join(args.output_dir, "inference_results")
    os.makedirs(inference_dir, exist_ok=True)
    
    # Run inference
    predictions = []
    contexts = []
    
    logger.info("Starting inference with autoregressive generation...")
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.float().to(device)
            
            # Use only context for inference
            context = batch_x[:, :args.context_length, :]
            
            # Generate predictions using inference mode
            logger.debug(f"Generating predictions for batch {i}...")
            pred = model.inference(context)
            logger.debug(f"Generated predictions shape: {pred.shape}")
            
            # Validate predictions
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                logger.warning(f"Invalid predictions in batch {i}, using zeros")
                pred = torch.zeros_like(pred)
            
            predictions.append(pred.cpu().numpy())
            contexts.append(context.cpu().numpy())
            
            logger.info(f"Successfully processed batch {i}: {len(batch_x)} samples")
            
            if i >= 10:  # Process first 10 batches for demo
                break
    
    logger.info(f"Inference complete")
    
    # Save results
    if predictions:
        logger.info("Saving predictions...")
        all_predictions = np.concatenate(predictions, axis=0)
        all_contexts = np.concatenate(contexts, axis=0)
        
        np.save(os.path.join(inference_dir, "predictions.npy"), all_predictions)
        np.save(os.path.join(inference_dir, "contexts.npy"), all_contexts)
        
        logger.info(f"Saved predictions shape: {all_predictions.shape}")
        logger.info(f"Saved contexts shape: {all_contexts.shape}")
    else:
        logger.error("No predictions generated!")

    logger.info("Inference results saved")


def main():
    """Main entry point."""
    args = parse_arguments()

    setup_logging(args)
    logger.info("Starting experiment with Transformer decoder and cross-attention")
    logger.info("Experiment arguments: %s", vars(args))
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
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