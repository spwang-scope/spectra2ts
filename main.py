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
import signal
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

from cuda_device_mapper import CUDADeviceMapper

logger = logging.getLogger(__name__)


def setup_logging(args):
    """Setup logging configuration."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, f'{args.mode}.log')),
            logging.StreamHandler()
        ]
    )
    
    return


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ViT-to-TimeSeries Model Training")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Model arguments
    parser.add_argument("--pred_len", type=int, default=96,
                       help="Length of time series to predict")
    parser.add_argument("--seq_len", type=int, default=96,
                       help="Length of context window")
    parser.add_argument("--feature_projection_dim", type=int, default=256,
                       help="Dimension for QKV vectors in decoder cross-attention")
    parser.add_argument("--pred_dim", type=int, default=1,
                       help="Dimension of time series (1 for univariate)")
    parser.add_argument("--d_model", type=int, default=768,
                       help="Hidden dimension for transformer decoder")
    parser.add_argument("--n_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--d_layers", type=int, default=3,
                       help="Number of decoder layers")
    parser.add_argument("--d_ff", type=int, default=1024,
                       help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate for transformer")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--train_epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay for optimizer")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                       help="Number of warmup epochs")
    parser.add_argument("--scheduler", type=str, default="cosine",
                       choices=["cosine", "step", "linear"],
                       help="Learning rate scheduler")
    
    # Data arguments
    parser.add_argument("--root_path", type=str, default="../dataset/ETT-small",
                       help="Directory containing dataset")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv",
                       help="Dataset filename")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Experiment arguments
    parser.add_argument("--experiment_name", type=str, default="vit_timeseries",
                       help="Name of the experiment")
    
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
    parser.add_argument("--device", type=str, default="cuda",
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
    args.output_dir = f"./outputs_{timestamp}_{args.data_path.split('.')[0]}_{args.mode}_{args.seq_len}_{args.pred_len}"
    
    def get_df_channel():
        df = pd.read_csv(os.path.join(args.root_path,
                                          args.data_path))
        return df.shape[1]-1  # number of columns, exclude Datetime
    
    # Determine number of data channels since we want to construct model according to it
    args.num_channels = get_df_channel()
    
    return args


def get_device(args) -> torch.device:
    """Get the appropriate device."""
    if hasattr(args, 'device') and hasattr(args, 'cuda_num'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    else:
        exit("Device error!")


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
            optimizer, T_max=args.train_epochs, eta_min=1e-6
        )
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.train_epochs // 3, gamma=0.1
        )
    elif args.scheduler == "linear":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.train_epochs
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def save_checkpoint(model: ViTToTimeSeriesModel, optimizer: optim.Optimizer, epoch: int, 
                   metrics: Dict[str, float], filepath: str, logger, scaler=None, args=None):
    """Save model checkpoint."""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'scaler': scaler,  # Save scaler for TSLib standard testing
        }
        
        if args and args.mode == 'train':
            checkpoint['args'] = vars(args)
        
        torch.save(checkpoint, filepath)
        logger.info(f'Checkpoint saved: {filepath}')
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {filepath}: {str(e)}")


def load_checkpoint(filepath: str, model: ViTToTimeSeriesModel, 
                   optimizer: Optional[optim.Optimizer] = None, logger = None) -> tuple:
    """Load model checkpoint with smart handling for decoder positional encoding compatibility."""
    try:
        # Get current model's prediction length
        current_prediction_length = model.prediction_length
        current_pos_encoding_type = type(model.ts_decoder.pos_encoding).__name__
        
        if logger:
            logger.info(f'Current model prediction_length: {current_prediction_length}')
            logger.info(f'Current model uses: {current_pos_encoding_type}')
        
        checkpoint = torch.load(filepath, map_location='cpu')
        model_state = checkpoint['model_state_dict']
        
        # Load the model state dict (strict=False only for positional encoding buffer)
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        
        # Check if missing keys are only expected positional encoding parameters
        unexpected_missing = [k for k in missing_keys if not k.startswith('ts_decoder.pos_encoding.')]
        if unexpected_missing:
            if logger:
                logger.warning(f'Unexpected missing keys (not positional encoding): {unexpected_missing}')
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        scaler = checkpoint.get('scaler', None)
        
        if logger:
            logger.info(f'Checkpoint loaded: {filepath}, epoch: {epoch}')
            if scaler is not None:
                logger.info('Scaler loaded from checkpoint for TSLib standard testing')
            else:
                logger.warning('No scaler found in checkpoint - predictions may be on wrong scale!')
        
        return epoch, scaler
    except Exception as e:
        if logger:
            logger.error(f"Failed to load checkpoint from {filepath}: {str(e)}")
        raise


def train(args):
    """Main training function with teacher forcing."""
    
    # Setup
    device = get_device(args)
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create data loaders
    train_dataset, train_loader = data_provider(args, flag='train')
    logger.info(f"Created data loaders: {len(train_loader)} trains")
    # Create data loaders
    test_data, test_loader = data_provider(args, flag='test')
    logger.info(f"Created data loaders: {len(test_loader)} tests")
    # Create data loaders
    vali_data, val_loader = data_provider(args, flag='val')
    logger.info(f"Created data loaders: {len(val_loader)} vals")
    
    # Create model
    logger.info("Creating ViT-to-TimeSeries model with Transformer decoder...")
    model = create_model(
        num_channels=args.num_channels,
        prediction_length=args.pred_len,
        context_length=args.seq_len,
        feature_projection_dim=args.feature_projection_dim,
        pred_dim=args.pred_dim,
        d_model=args.d_model,
        ts_num_heads=args.n_heads,
        ts_num_layers=args.d_layers,
        ts_dim_feedforward=args.d_ff,
        ts_dropout=args.dropout
    ).to(device)
    logger.info("Model created successfully")

    #logger.info(f"Model architecture: {model}")
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
    criterion = nn.MSELoss() 
    
    # Training loop
    start_epoch = 0
    start_max_vali_loss = float('inf')
    
    if args.checkpoint_path:
        logger.info(f"Loading checkpoint from {args.checkpoint_path}...")
        start_epoch, _ = load_checkpoint(args.checkpoint_path, model, optimizer, logger)
    
    logger.info(f"Starting training from epoch {start_epoch}")
    logger.info(f"Device: {device}")
    
    for epoch in range(start_epoch, args.train_epochs):
        logger.info(f"Starting epoch {epoch}/{args.train_epochs}")
        
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

            # Select target feature for loss calculation based on pred_dim
            outputs = outputs[:, :, :].to(device)
            if hasattr(args, 'pred_dim') and args.pred_dim > 1:
                # Multi-variable prediction: use last pred_dim features
                batch_y = batch_y[:, :args.pred_len, -args.pred_dim:].to(device)
            else:
                # Single variable prediction: use last feature only
                batch_y = batch_y[:, :args.pred_len, -1:].to(device)
            
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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

        logger.info(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
        train_loss = np.average(train_loss)
        
        # Evaluate on test set
        test_loss = test(args, peeking=True, model=model, epoch=epoch, test_loader=test_loader)
        vali_loss = vali(args, model=model, val_loader=val_loader)
        
        logger.info(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} Val Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f} ")
        
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
        
        # Save checkpoint
        if vali_loss < start_max_vali_loss:
            logger.info(f"New best model found at epoch {epoch + 1} with Val Loss: {vali_loss:.7f}.")
            logger.info(f"Saving checkpoint for epoch {epoch + 1}...")
            checkpoint_path = os.path.join(experiment_dir, f'checkpoint_best.pt')
            save_checkpoint(model, optimizer, epoch + 1, 
                          {'train_loss': train_loss, 'test_loss': test_loss}, 
                          checkpoint_path, logger, scaler=args._scaler, args=args)
            args.checkpoint_path = checkpoint_path
            start_max_vali_loss = vali_loss
    
    logger.info("Training completed!")

def vali(args, model, val_loader):
    device = get_device(args)
    criterion = nn.MSELoss()

    preds = []
    trues = []
    total_loss = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(val_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # Use inference mode (no teacher forcing)
            outputs = model.inference(batch_x[:, :args.seq_len, :])

            # Calculate loss
            f_dim = -1
            outputs = outputs[:, :, :].to(device)
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            
            loss = criterion(outputs, batch_y)
            
            # Store predictions and ground truth (normalized for loss calculation)
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            
            preds.append(pred)
            trues.append(true)

            total_loss.append(loss.item())
            
    avg_loss = np.average(total_loss)

    model.train()  # Return to training mode
    return avg_loss

def test(args, peeking=False, model=None, epoch=None, test_loader=None):
    """Test the model using inference mode (no teacher forcing)."""
    
    device = get_device(args)
    
    if not args.checkpoint_path and not peeking and not model:
        raise ValueError("checkpoint_path must be specified for testing")
    
    # Create and load model
    if not peeking:
        logger.info("Creating model for testing...")
        model = create_model(
            num_channels=args.num_channels,
            prediction_length=args.pred_len,
            context_length=args.seq_len,
            feature_projection_dim=args.feature_projection_dim,
            pred_dim=args.pred_dim,
            d_model=args.d_model,
            ts_num_heads=args.n_heads,
            ts_num_layers=args.d_layers,
            ts_dim_feedforward=args.d_ff,
            ts_dropout=args.dropout,
        ).to(device)
        _, scaler = load_checkpoint(args.checkpoint_path, model, logger=logger)
        logger.info("Model created successfully") 
        
        # Set scaler for proper data normalization and denormalization
        if scaler is not None:
            args._scaler = scaler
            assert args._scaler is not None,  "scaler is not loaded correctly!"
            logger.info("Scaler loaded and set for TSLib standard testing")
        else:
            logger.warning("No scaler found in checkpoint - this will cause incorrect results!")

        if test_loader is None:
        # Create data loader
            test_data, test_loader = data_provider(args, flag='test')
            logger.info(f"replaced test loader scaler with loaded scaler")
            logger.info(f"Created data loader: {len(test_loader)} tests")
        
        
        
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
            outputs = model.inference(batch_x[:, :args.seq_len, :])

            # Calculate loss
            f_dim = -1
            outputs = outputs[:, :, :].to(device)
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            
            loss = criterion(outputs, batch_y)
            
            # Store predictions and ground truth (normalized for loss calculation)
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            
            preds.append(pred)
            trues.append(true)

            # TSLib standard: Denormalize for visualization only
            input_np = batch_x.detach().cpu().numpy()
            
            if not peeking and hasattr(args, '_scaler') and args._scaler is not None:
                # Denormalize for visualization (standalone testing only)
                # Reconstruct full feature dimension for inverse transform
                batch_size = pred.shape[0]
                pred_full = np.zeros((batch_size, pred.shape[1], input_np.shape[2]))
                true_full = np.zeros((batch_size, true.shape[1], input_np.shape[2]))
                
                pred_full[:, :, -1] = pred.numpy()[:, :, -1]  # Only last feature (target)
                true_full[:, :, -1] = true.numpy()[:, :, -1]
                
                # Denormalize predictions and ground truth
                pred_denorm = args._scaler.inverse_transform(
                    pred_full.reshape(-1, input_np.shape[2])
                ).reshape(pred_full.shape)
                true_denorm = args._scaler.inverse_transform(
                    true_full.reshape(-1, input_np.shape[2])
                ).reshape(true_full.shape)
                
                # Also denormalize input context for consistent visualization
                input_denorm = args._scaler.inverse_transform(
                    input_np.reshape(-1, input_np.shape[2])
                ).reshape(input_np.shape)
                
                # Create visualization with denormalized data
                gt = np.concatenate((input_denorm[0, :args.seq_len, -1], true_denorm[0, :, -1]), axis=0)
                pd = np.concatenate((input_denorm[0, :args.seq_len, -1], pred_denorm[0, :, -1]), axis=0)
            else:
                # Use normalized data for visualization (during peeking or no scaler)
                gt = np.concatenate((input_np[0, :args.seq_len, -1], true[0, :, -1].numpy()), axis=0)
                pd = np.concatenate((input_np[0, :args.seq_len, -1], pred[0, :, -1].numpy()), axis=0)
            
            # Generate visualization
            #if peeking:
            #    if (i % 20 == 0 and (epoch + 1) % 10 == 0):
            #        visual(gt, pd, os.path.join(args.output_dir, f'test_vis_{i}_epoch{epoch + 1}.png'))
            #else:
            #    if (i % 10 == 0):
            #        visual(gt, pd, os.path.join(args.output_dir, f'test_vis_{i}.png'))

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

    logger.info(f"Preds: {preds.shape}, Trues: {trues.shape}")
    logger.info(f"Pred range: {preds.min():.6f} to {preds.max():.6f}")
    logger.info(f"True range: {trues.min():.6f} to {trues.max():.6f}")

    

    # Calculate metrics on normalized data (TSLib standard)
    mae, mse, _, _, _ = metric(preds, trues)
    logger.info(f'MSE (normalized): {mse:.7f}, MAE (normalized): {mae:.7f}')
    
    if peeking:
        model.train()  # Return to training mode
        return avg_loss

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # TSLib standard: Also calculate denormalized metrics for interpretation
    if hasattr(args, '_scaler') and args._scaler is not None:
        logger.info("Computing denormalized metrics for interpretation...")
        
        # Denormalize all predictions and ground truth for evaluation
        preds_full = np.zeros((preds.shape[0], preds.shape[1], len(args._scaler.mean_)))
        trues_full = np.zeros((trues.shape[0], trues.shape[1], len(args._scaler.mean_)))
        
        preds_full[:, :, -1] = preds[:, :, -1]  # Only target feature
        trues_full[:, :, -1] = trues[:, :, -1]
        
        preds_denorm = args._scaler.inverse_transform(
            preds_full.reshape(-1, preds_full.shape[2])
        ).reshape(preds_full.shape)[:, :, -1:] # Keep only target dimension
        
        trues_denorm = args._scaler.inverse_transform(
            trues_full.reshape(-1, trues_full.shape[2])
        ).reshape(trues_full.shape)[:, :, -1:] # Keep only target dimension
        
        # Calculate denormalized metrics
        mae_denorm, mse_denorm, _, _, _ = metric(preds_denorm, trues_denorm)
        logger.info(f'MSE (denormalized): {mse_denorm:.7f}, MAE (denormalized): {mae_denorm:.7f}')
        
        # Save both normalized and denormalized results
        with open(os.path.join(args.output_dir, "test_results.txt"), 'w') as f:
            f.write(f'Normalized - MSE: {mse:.7f}, MAE: {mae:.7f}\n')
            f.write(f'Denormalized - MSE: {mse_denorm:.7f}, MAE: {mae_denorm:.7f}\n')
        
        np.save(os.path.join(args.output_dir, 'test_pred_denorm.npy'), preds_denorm)
        np.save(os.path.join(args.output_dir, 'test_true_denorm.npy'), trues_denorm)
        np.save(os.path.join(args.output_dir, 'test_metrics_denorm.npy'), np.array([mae_denorm, mse_denorm]))
    else:
        with open(os.path.join(args.output_dir, "test_results.txt"), 'w') as f:
            f.write(f'MSE: {mse:.7f}, MAE: {mae:.7f}\n')

    np.save(os.path.join(args.output_dir, 'test_metrics.npy'), np.array([mae, mse]))
    np.save(os.path.join(args.output_dir, 'test_pred.npy'), preds)
    np.save(os.path.join(args.output_dir, 'test_true.npy'), trues)

    return avg_loss


def inference(args):
    """Run inference on new data using autoregressive generation."""
    
    device = get_device(args)
    
    if not args.no_checkpoint and not args.checkpoint_path:
        raise ValueError("checkpoint_path must be specified for inference, or use --no_checkpoint for trial run")
    
    # Create model
    logger.info("Creating model for inference...")
    model = create_model(
        num_channels=args.num_channels,
        prediction_length=args.pred_len,
        context_length=args.seq_len,
        feature_projection_dim=args.feature_projection_dim,
        pred_dim=args.pred_dim,
        d_model=args.d_model,
        ts_num_heads=args.n_heads,
        ts_num_layers=args.d_layers,
        ts_dim_feedforward=args.d_ff,
        ts_dropout=args.dropout
    ).to(device)
    logger.info("Model created successfully with Transformer decoder")
    
    # Load checkpoint if provided
    if args.checkpoint_path and not args.no_checkpoint:
        try:
            _, scaler = load_checkpoint(args.checkpoint_path, model, logger=logger)
            if scaler is not None:
                args._scaler = scaler
                logger.info("Scaler loaded for denormalized inference outputs")
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
            context = batch_x[:, :args.seq_len, :]
            
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
        
        # Save normalized predictions
        np.save(os.path.join(inference_dir, "predictions.npy"), all_predictions)
        np.save(os.path.join(inference_dir, "contexts.npy"), all_contexts)
        
        # TSLib standard: Also save denormalized predictions if scaler is available
        if hasattr(args, '_scaler') and args._scaler is not None:
            logger.info("Saving denormalized predictions...")
            
            # Denormalize predictions
            pred_full = np.zeros((all_predictions.shape[0], all_predictions.shape[1], len(args._scaler.mean_)))
            pred_full[:, :, -1] = all_predictions[:, :, -1]  # Only target feature
            
            pred_denorm = args._scaler.inverse_transform(
                pred_full.reshape(-1, pred_full.shape[2])
            ).reshape(pred_full.shape)[:, :, -1:]
            
            # Denormalize contexts
            context_denorm = args._scaler.inverse_transform(
                all_contexts.reshape(-1, all_contexts.shape[2])
            ).reshape(all_contexts.shape)
            
            np.save(os.path.join(inference_dir, "predictions_denorm.npy"), pred_denorm)
            np.save(os.path.join(inference_dir, "contexts_denorm.npy"), context_denorm)
            
            logger.info(f"Saved denormalized predictions shape: {pred_denorm.shape}")
            logger.info(f"Saved denormalized contexts shape: {context_denorm.shape}")
        
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
        test(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "inference":
        inference(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()