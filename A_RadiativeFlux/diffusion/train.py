"""
Training script for diffusion model with UNet backbone for radiative flux prediction
"""

import os
import logging
import pickle
import argparse
import glob
from os.path import join, basename, normpath
import re
import random
import warnings

import torch
import lightning as L
from lightning import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

import wandb
from lightning.pytorch.loggers import WandbLogger


from unet import UNetDiffusion
from data_utils_diffusion import DataNormalizer, IconDiffusionDataset
from edm import LightningEDM, EDM

import torch.multiprocessing as mp


def get_normalization_params(stats_file, device):
    """Load normalization parameters from a pickle file"""
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)
        return torch.tensor(stats['mean2d']).to(device), \
            torch.tensor(stats['var2d']).to(device), \
                torch.tensor(stats['mean3d']).to(device), \
                    torch.tensor(stats['var3d']).to(device)


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Clean up any existing wandb run at the very beginning
    if wandb.run is not None:
        wandb.finish()
        
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
        
    # Modify CUDA settings to avoid cudnn issues
    if torch.cuda.is_available():
        # Disable cudnn for Conv1d operations
        torch.backends.cudnn.enabled = False
    
    parser = argparse.ArgumentParser(description='Train diffusion model for radiation flux prediction')
    
    # General parameters
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--save', type=str, required=True, help='Path to save the results')
    parser.add_argument('--percent', type=float, default=1.0, help='Percent of data to train on')
    parser.add_argument('--subsample', type=float, default=None, help='Subsampling rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='Prefetch factor for data loading')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    # parser.add_argument('--max-steps', type=int, default=100000, help='Maximum training steps')
    parser.add_argument('--max-epochs', type=int, default=100, help='Maximum training epochs')
    # Model parameters
    parser.add_argument('--height-in', type=int, default=71, help='Number of height levels')
    parser.add_argument('--channel-out', type=int, default=4, help='Output channels')
    parser.add_argument('--channel-3d', type=int, default=6, help='3D input channels')
    parser.add_argument('--channel-2d', type=int, default=6, help='2D input channels')
    parser.add_argument('--cnn-units', nargs='+', type=int, default=[64, 128, 256, 512], 
                       help='Number of filters in each CNN layer')
    parser.add_argument('--cnn-kernel-sizes', nargs='+', type=int, default=[2, 2, 2, 2], 
                       help='Kernel sizes for max pooling in CNN layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    
    # Diffusion parameters
    parser.add_argument('--num-sampling-steps', type=int, default=25, 
                       help='Number of sampling steps for diffusion model')
    parser.add_argument('--deterministic-sampling', action='store_true', default=True,
                       help='Whether to use deterministic sampling in diffusion model')
    parser.add_argument('--sigma-min', type=float, default=0.002, 
                       help='Minimum noise level for diffusion model')
    parser.add_argument('--sigma-max', type=float, default=80.0, 
                       help='Maximum noise level for diffusion model')
    parser.add_argument('--sigma-data', type=float, default=0.5, 
                       help='Data noise level for diffusion model')
    parser.add_argument('--time-embedding-dim', type=int, default=128, 
                       help='Time embedding dimension for diffusion model')
    
    parser.add_argument('--wandb-mode', type=str, default='disabled', 
                      choices={'online', 'offline', 'disabled'}, help='Operating mode for W&B')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup save directory and ID
    save_id = basename(normpath(args.save))
    log_dir = args.save
    os.makedirs(log_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    # WandB config
    wandb_config = {'seed': seed}
    
    # Load and prepare dataset
    logging.info("Preparing dataset...")
    FILENAMES = glob.glob(join(args.dataset, '*.h5'))
    time_indices = [float(re.search(r'\_time_(.*?)\.h5', f).group(1)) for f in FILENAMES]
    sorted_files = [x for _, x in sorted(zip(time_indices, FILENAMES))]
    
    train_files = sorted_files[200:2000]
    val_files = sorted_files[:160] + sorted_files[2020:2180]
    
    # Sample dataset if needed
    if args.percent < 1:
        random.shuffle(train_files)
        train_files = train_files[:int(args.percent * len(train_files))]
        
        random.shuffle(val_files)
        val_files = val_files[:int(args.percent * len(val_files))]
    
    # Get normalization parameters
    stats_file = join(args.dataset, 'normalizer_stats_per_feat.pickle')
    mean2d, var2d, mean3d, var3d = get_normalization_params(stats_file, device)
    normalizer = DataNormalizer(mean2d, var2d, mean3d, var3d, device=device)
    
    # Create datasets
    train_dataset = IconDiffusionDataset(
        filenames=train_files,
        normalizer=normalizer,
        shuffle=True,
        subsample=args.subsample,
        cache_dir='/tmp'
    )
    
    val_dataset = IconDiffusionDataset(
        filenames=val_files,
        normalizer=normalizer,
        shuffle=False,
        subsample=args.subsample,
        cache_dir='/tmp'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor
    )
    
    # TODO: Configure optimizer params a bit hacky now
    # Calculate approximate steps based on dataset size and epochs
    # Calculate using actual columns in the dataset rather than file count
    # For ICON grid data, each file contains a different number of columns
    total_samples = 81920  # This is the hard-coded total samples in your dataset
    # Adjust for subsampling and percent
    effective_samples = total_samples
    if args.subsample:
        effective_samples = int(effective_samples * args.subsample)
            
    steps_per_epoch = max(1, effective_samples // args.batch_size)
    max_steps = steps_per_epoch * args.max_epochs
    
    # Log these estimates
    logging.info(f"Estimated files: {len(train_files)}")
    logging.info(f"Estimated total samples: {total_samples}")
    logging.info(f"Effective samples after subsampling: {effective_samples}")
    logging.info(f"Steps per epoch: {steps_per_epoch}")
    logging.info(f"Max steps: {max_steps}")
    
    optimizer_params = {
        "learning_rate": args.learning_rate,
        "max_steps": max_steps  # Required by EDM's CosineAnnealingLR scheduler
    }
    
    # Setup EDM
    edm = EDM(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=args.sigma_data
    )
    
    # Build UNet model
    logging.info("Building UNet model...")
    unet = UNetDiffusion(
        height_in=args.height_in,
        channel_3d=args.channel_3d,
        channel_2d=args.channel_2d,
        channel_out=args.channel_out,
        cnn_units=args.cnn_units,
        kernel_sizes=args.cnn_kernel_sizes,
        time_embed_dim=args.time_embedding_dim,
        dropout=args.dropout,
        device=device
    )
    
    # Get parameter count
    num_params = count_parameters(unet)
    logging.info(f"Model has {num_params:,} trainable parameters")
    
    # Initialize wandb if not disabled
    logger = None
    if args.wandb_mode != 'disabled':
        # First finish any existing run
        if wandb.run is not None:
            wandb.finish()
            
        wandb.init(
            project='deepcloud-yves', 
            name=save_id, 
            id=save_id, 
            config={**wandb_config, **args.__dict__}, 
            sync_tensorboard=True, 
            save_code=True, 
            resume=None,  # Don't attempt to resume, always create a new run 
            tags=['icon grid', 'diffusion'],    
            mode=args.wandb_mode
        )
        # Watch the model
        wandb.watch(unet, log_freq=100)
        logging.info("WandB initialized in mode: " + args.wandb_mode)
        logger = WandbLogger(project='deepcloud-yves', log_model=True)
    
    # Build Lightning EDM model
    model = LightningEDM(
        unet=unet,
        optimizer_params=optimizer_params,
        num_sampling_steps=args.num_sampling_steps,
        deterministic_sampling=args.deterministic_sampling,
        edm=edm
    )
    
    # Setup callbacks
    checkpoint_callbacks = [
        ModelCheckpoint(
        dirpath=log_dir,
        filename="last",
        save_on_train_epoch_end=True,
        save_last=True
        ),
        ModelCheckpoint(
        dirpath=log_dir, 
            filename="model_best_val_ckpt-{epoch}-{validation/loss:.4f}", 
        monitor="validation/loss", 
        save_top_k=3, 
        mode='min', 
        save_on_train_epoch_end=False
    )
    ]
    
    # Setup trainer
    logging.info("Setting up trainer...")
    torch.set_float32_matmul_precision("high")
    
    # Check if a checkpoint exists, but don't require it
    checkpoint_file = os.path.join(log_dir, 'last.ckpt')
    last_ckpt = checkpoint_file if os.path.exists(checkpoint_file) else None
    if last_ckpt:
        logging.info(f"Found checkpoint at {last_ckpt}, will resume training")
    else:
        logging.info("No checkpoint found, starting training from scratch")
    
    trainer = Trainer(
        precision=32,
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        num_nodes=1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        log_every_n_steps=max(1, steps_per_epoch // 10),  # Log at most 10 times per epoch
        accumulate_grad_batches=1,
        strategy='auto',
        callbacks=checkpoint_callbacks,
        gradient_clip_val=1.0,
        enable_progress_bar=False,  # Disable progress bar completely
        enable_checkpointing=True,
        logger=logger,
    )   
    
    # Start training
    logging.info("Starting training...")
    try:
        print(f"\n{'='*50}")
        print(f" DIFFUSION TRAINING")
        print(f" - Files: Train {len(train_files)}, Val {len(val_files)}")
        print(f" - Batch size: {args.batch_size}, Steps/epoch: ~{steps_per_epoch}")
        print(f" - Learning rate: {args.learning_rate}")
        print(f"{'='*50}\n")
        
        # Train the model
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=last_ckpt,
        )
        
        logging.info("Training completed successfully!")
        print(f"\n{'='*50}")
        print(f" TRAINING COMPLETE")
        print(f"{'='*50}\n")
        
        # Log final metrics to wandb
        if args.wandb_mode != 'disabled':
            # Get the final metrics from the trainer
            final_metrics = {"num_parameters": num_params}
            wandb.log(final_metrics)
            wandb.finish()
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        if args.wandb_mode != 'disabled':
            wandb.finish()
        raise e


if __name__ == "__main__":
    main()