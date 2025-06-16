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
import pandas as pd
from data_factory import data_provider
from metrics import metric
from util import visual

# Check transformers version compatibility
try:
    import transformers
    transformers_version = transformers.__version__
    print(f"Using transformers version: {transformers_version}")
except ImportError:
    raise ImportError("transformers library is required. Install with: pip install transformers")

from model import ViTToTimeSeriesModel, create_model
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
    parser.add_argument("--ts_model_dim", type=int, default=768,
                       help="Hidden dimension for transformer decoder")
    parser.add_argument("--ts_num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--ts_num_layers", type=int, default=4,
                       help="Number of decoder layers")
    parser.add_argument("--ts_dim_feedforward", type=int, default=1024,
                       help="Feed-forward dimension")
    parser.add_argument("--ts_dropout", type=float, default=0.1,
                       help="Dropout rate for transformer")
    parser.add_argument('--target', type=str, default='OT',
                       help='target feature in S or MS task')
    parser.add_argument("--lstm", action="store_true",
                       help="Use LSTM decoder instead of Transformer decoder")
    
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
                       help="Size of spectrogram")
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
                       choices=["pred", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--tensorboard", action="store_true",
                       help="Use tensorboard logging")
    
    args = parser.parse_args()
    
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
    device = get_device(args.device, args.cuda_num)
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create data loaders
    train_dataset, train_loader = data_provider(args, flag='train')
    test_dataset, test_loader = data_provider(args, flag='test')
    logger.info(f"Created data loaders: {len(train_loader)} trains {len(test_loader)} tests")
    
    # Create model
    logger.info("Creating ViT-to-TimeSeries model...")
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
        use_lstm_decoder=args.lstm,
    ).to(device)
    logger.info("Model created successfully")
    
    
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
    optimizer, _ = create_optimizer_and_scheduler(model, args)
    criterion = nn.MSELoss()
    
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
    logger.info(f"Using custom Transformer decoder with transformers v{transformers_version}")
    
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

            outputs = model(batch_x)

            f_dim = -1 # always select last feature (OT) for calculating loss?
            outputs = outputs[:, -args.prediction_length:, f_dim:].to(device)
            batch_y = batch_y[:, -args.prediction_length:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                iter_count = 0

            loss.backward()
            if epoch < args.num_epochs*0.05: # Warmup phase
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

        logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        test_loss = test(args, peeking=True, model=model)    # Peek testing result

        logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
            epoch + 1, -1, train_loss, test_loss))
        
        if epoch % args.save_interval == 0:
            logger.info(f"Saving checkpoint for epoch {epoch}...")
            checkpoint_path = os.path.join(experiment_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, epoch, {}, checkpoint_path, logger)
    
    # Final save
    logger.info("Saving final model...")
    final_path = os.path.join(experiment_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, args.num_epochs - 1, {}, final_path, logger)
    
    if writer:
        writer.close()
    
    logger.info("Training completed!")


def test(args, peeking=False, model=None):
    """Test the model."""
    # Setup logging
    logger = setup_logging(args.output_dir, args.log_level)
    
    device = get_device(args.device, args.cuda_num)
    
    if not args.checkpoint_path and not peeking:
        raise ValueError("checkpoint_path must be specified for testing")
    
    # Create data loaders
    test_data, test_loader = data_provider(args, flag='test')
    
    # Create and load model
    if not peeking:
        logger.info("Creating model for testing...")
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
            use_lstm_decoder=args.lstm,
        ).to(device)
        logger.info("Model created successfully") 
        load_checkpoint(args.checkpoint_path, model, logger=logger)
        logger.info("Model weights loaded successfully") 
    else:   # peeking
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

            outputs = model(batch_x)

            f_dim = -1 # always select last feature (OT) for calculating loss?
            outputs = outputs[:, -args.prediction_length:, f_dim:].to(device)
            batch_y = batch_y[:, -args.prediction_length:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)
            
            # FIXED: Move both tensors to CPU for loss calculation (following original implementation)
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            # Store predictions and ground truth
            preds.append(pred)
            trues.append(true)
            if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(args.output_dir, str(i) + '.png'))
            
            total_loss.append(loss.item())
            
    avg_loss = np.average(total_loss)

    if peeking:
        model.train()  # Return to training mode if peeking

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
    print('contains NaN:', np.isnan(preds).any(), np.isnan(trues).any())
    print('contains Inf:', np.isinf(preds).any(), np.isinf(trues).any())

    # Calculate metrics
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))
    
    # Save results
    f = open(os.path.join(args.output_dir, "result.txt"), 'a')
    f.write('MSE: {:.7f}, MAE: {:.7f}, RMSE: {:.7f}, MAPE: {:.7f}, MSPE: {:.7f}\n'.format(
        mse, mae, rmse, mape, mspe))
    f.write('\n')
    f.write('\n')
    f.close()

    np.save(os.path.join(args.output_dir, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
    np.save(os.path.join(args.output_dir, 'pred.npy'), preds)
    np.save(os.path.join(args.output_dir, 'true.npy'), trues)

    return avg_loss


def inference(args):
    
    """Run inference on new data."""
    # Setup logging
    logger = setup_logging(args.output_dir, args.log_level)
    
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
        use_lstm_decoder=args.lstm,
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


    data_set, data_loader = data_provider(args, flag='test')
    
    model.eval()
    
    # Create output directory for predictions
    inference_dir = os.path.join(args.output_dir, "inference_results")
    os.makedirs(inference_dir, exist_ok=True)
    
    # Run inference on images
    predictions = []
    image_names = []
    failed_images = []
    
    logger.info("Starting inference...")
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            
            # Generate predictions
            logger.debug(f"Generating predictions for batch {i}...")
            pred = model(batch_x)
            logger.debug(f"Generated predictions shape: {pred.shape}")
            
            # Validate predictions
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                logger.warning(f"Invalid predictions in batch {i}, using zeros")
                pred = torch.zeros_like(pred)
            
            predictions.append(pred.cpu().numpy())
            
            logger.info(f"Successfully processed batch {i}: {len(batch_x)} images")
            
            break  # Only one batch per dataloader
    
    logger.info(f"Inference complete")
    
    # Concatenate all predictions
    if predictions:
        logger.info("Concatenating all predictions...")
        all_predictions = np.concatenate(predictions, axis=0)
        
        # Save predictions
        try:
            logger.info("Saving predictions...")
            np.save(os.path.join(inference_dir, "predictions.npy"), all_predictions)
        except Exception as e:
            logger.error(f"Failed to save inference results: {str(e)}")
            
    else:
        logger.error("No predictions generated! All batches failed.")

    logger.info(f"Inference result saved")
    

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