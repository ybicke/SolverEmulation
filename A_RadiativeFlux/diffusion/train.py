"""
Training script for diffusion model with UNet backbone for radiative flux prediction
"""

import os
import logging
import pickle
import argparse
import glob
from os.path import join
import re
import random

import torch
import lightning as L
from lightning import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from unet import UNet
from data_utils_diffusion import DataNormalizer, IconDiffusionDataset
from edm import LightningEDM, EDM


def get_normalization_params(stats_file, device):
    """Load normalization parameters from a pickle file"""
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)
        return torch.tensor(stats['mean2d']).to(device), \
            torch.tensor(stats['var2d']).to(device), \
                torch.tensor(stats['mean3d']).to(device), \
                    torch.tensor(stats['var3d']).to(device)


def main():
    parser = argparse.ArgumentParser(description='Train diffusion model for radiation flux prediction')
    
    # General parameters
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--save', type=str, required=True, help='Path to save the results')
    parser.add_argument('--percent', type=float, default=1.0, help='Percent of data to train on')
    parser.add_argument('--subsample', type=float, default=None, help='Subsampling rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max-steps', type=int, default=100000, help='Maximum training steps')
    
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
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create save directory
    log_dir = args.save
    os.makedirs(log_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
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
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Configure optimizer params
    optimizer_params = {
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps
    }
    
    # Setup EDM
    edm = EDM(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=args.sigma_data
    )
    
    # Build UNet model
    logging.info("Building UNet model...")
    unet = UNet(
        height_in=args.height_in,
        channel_3d=args.channel_3d,
        channel_2d=args.channel_2d,
        channel_out=args.channel_out,
        cnn_units=args.cnn_units,
        kernel_sizes=args.cnn_kernel_sizes,
        time_embedding_dim=args.time_embedding_dim,
        dropout=args.dropout,
        device=device
    )
    
    # Build Lightning EDM model
    model = LightningEDM(
        unet=unet,
        optimizer_params=optimizer_params,
        num_sampling_steps=args.num_sampling_steps,
        deterministic_sampling=args.deterministic_sampling,
        edm=edm
    )
    
    # Setup checkpointing
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
    
    last_ckpt = os.path.join(log_dir, 'last.ckpt') if os.path.exists(
        os.path.join(log_dir, 'last.ckpt')) else None
    
    trainer = Trainer(
        precision=32,
        max_steps=args.max_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        num_nodes=1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        # log_every_n_steps=1,
        log_every_n_steps=10,
        accumulate_grad_batches=1,
        strategy='auto',
        callbacks=checkpoint_callbacks,
        gradient_clip_val=1.0,
    )   
    
    # Start training
    logging.info("Starting training...")
    try:
        trainer.fit(
        model,
        train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        ckpt_path=last_ckpt,
    )
        logging.info("Training completed successfully!")
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise e


if __name__ == "__main__":
    main()