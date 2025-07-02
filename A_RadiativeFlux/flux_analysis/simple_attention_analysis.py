#!/usr/bin/env python3
"""
Simple Attention Analysis Script

Based on the proven approach from train_column_vit_analysis.py and vit_column_analysis.py
but adapted for the new code setup.

Usage:
    cd SolverEmulation/A_RadiativeFlux
    python simple_attention_analysis.py --model-path /path/to/best_model.pth --dataset /path/to/data --save-dir /path/to/results
"""

import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from os.path import join
from torch.utils.data import DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_attention_weights(model, x3d_norm, x2d_norm, x2d_orig, layer_idx=0):
    """
    Extract attention weights from a specific transformer layer.
    Simple approach that directly accesses the attention computation.
    """
    
    # Ensure model is in eval mode
    model.eval()
    
    with torch.no_grad():
        # Process inputs through normalization if needed
        if hasattr(model, 'normalizer'):
            x3d_norm = model.normalizer.normalize_3d(x3d_norm)
            x2d_norm = model.normalizer.normalize_2d(x2d_norm)
        
        # Concatenate inputs (following the model's forward pass logic)
        x2d_repeated = x2d_norm.unsqueeze(1).repeat(1, x3d_norm.shape[1], 1)
        x_concat = torch.cat((x3d_norm, x2d_repeated), dim=-1)
        
        # Apply patch embedding
        x = model.to_patch_embedding(x_concat)
        
        # Add positional embedding
        x += model.pos_embedding[:, :x.shape[1]]
        x = model.dropout(x)
        
        # Process through transformer layers up to the target layer
        for i, (attn_layer, ff_layer) in enumerate(model.transformer.layers):
            if i == layer_idx:
                # This is our target layer - extract attention weights
                
                # Normalize input
                x_norm = attn_layer.norm(x)
                
                # Get Q, K, V
                qkv = attn_layer.to_qkv(x_norm).chunk(3, dim=-1)
                
                # Import rearrange here
                from einops import rearrange
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=attn_layer.heads), qkv)
                
                # Compute attention weights
                dots = torch.matmul(q, k.transpose(-1, -2)) * attn_layer.scale
                attn_weights = attn_layer.softmax(dots)
                
                return attn_weights  # Shape: [batch, heads, seq_len, seq_len]
            
            # Apply the layer normally
            x = attn_layer(x) + x
            x = ff_layer(x) + x
    
    return None


def visualize_attention(avg_attention_matrix, layer_idx=0, save_path=None):
    """
    Visualize attention matrix with proper atmospheric height mapping.
    Based on the proven visualization from vit_column_analysis.py
    """
    
    # Height mapping: Position 0 = Surface token, Position 1-71 = x3d[0-70]
    # Level 70 = 0 km (surface), Level 0 = 65 km (TOA)
    height_mapping = {
        0: 65, 10: 39, 20: 25, 30: 15, 40: 8, 50: 4, 60: 1, 70: 0
    }
    
    # Create tick positions and labels
    tick_positions = list(height_mapping.keys())
    tick_labels = [f"{level} ({height_mapping[level]} km)" for level in tick_positions]
    
    # Flip both dimensions to have surface at bottom-right, TOA at top-left
    flipped_attn_rows = torch.flip(avg_attention_matrix, dims=[0])
    flipped_attn_cols = torch.flip(flipped_attn_rows, dims=[1])
    flipped_attn = flipped_attn_cols

    # Create publication-ready plot
    plt.figure(figsize=(12, 10), dpi=300)
    
    # Plot the attention matrix
    im = plt.imshow(flipped_attn.numpy(), cmap='viridis', interpolation='nearest', aspect='equal')
    
    # Set ticks and labels
    plt.xticks(tick_positions, tick_labels, fontsize=14, rotation=45)
    plt.yticks(tick_positions, tick_labels, fontsize=14)
    
    # Add labels
    plt.xlabel('Key Position (Information Source)', fontsize=16, labelpad=10)
    plt.ylabel('Query Position (Information Consumer)', fontsize=16, labelpad=10)
    plt.title(f'Attention Matrix - Layer {layer_idx}', fontsize=18, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    
    # Improve layout and save
    plt.tight_layout()
    save_filename = save_path if save_path else f'attention_layer_{layer_idx}_simple.png'
    plt.savefig(save_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Attention matrix saved: {save_filename}")
    logger.info(f"Matrix shape: {avg_attention_matrix.shape}")
    logger.info("Atmospheric visualization: Surface (0 km) at bottom-right, TOA (65 km) at top-left")


def run_analysis(args):
    """Run attention analysis with given arguments"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load normalization parameters
    stats_file = join(args.dataset, 'normalizer_stats_per_feat.pickle')
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)
        mean2d = torch.tensor(stats['mean2d'], dtype=torch.float32).to(device)
        var2d = torch.tensor(stats['var2d'], dtype=torch.float32).to(device)
        mean3d = torch.tensor(stats['mean3d'], dtype=torch.float32).to(device)
        var3d = torch.tensor(stats['var3d'], dtype=torch.float32).to(device)
    
    # Create normalizer
    from utils.data_utils import DataNormalizer
    normalizer = DataNormalizer(mean2d, var2d, mean3d, var3d, device=device)
    
    # Create model (matching the exact constructor from models_1d/vit.py)
    from models_1d.vit import ViT
    model = ViT(
        patch_size=args.patch_size,
        embed_dim=args.hidden_dim,
        depth=args.layers,
        heads=args.heads,
        dropout=0.0,
        emb_dropout=0.0,
        channels_in_3D=6,  # Adjust based on your data
        channels_in_2D=6,  # Adjust based on your data
        channels_out=4,    # Adjust based on your data
        height=70,         # Adjust based on your data (70 or 71)
        mlp_ratio=args.mlp_ratio,
        dim_head=args.dim_head
    ).to(device)
    
    # Load model weights
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test dataset
    filenames = glob.glob(join(args.dataset, '*.h5'))
    time_indices = [float(re.search(r'_time_(.*?)\.h5', f).group(1)) for f in filenames]
    sorted_files = [x for _, x in sorted(zip(time_indices, filenames))]
    test_files = sorted_files[2220:2300]  # Use subset for analysis
    
    # Create data loader (adjust this to match your data loader)
    from flux_data_loader import UnifiedFluxDataset
    dataset = UnifiedFluxDataset(
        filenames=test_files,
        mode='1d',
        dataset_type='triangle',
        triangle_id=0,
        division_factor=1,
        shuffle=False,
        subsample=None,
        cache_dir='/tmp',
        total_cols=81920,
        subsample_seed=42,
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Collect samples for attention analysis
    logger.info('Collecting samples for attention analysis...')
    attention_data = []
    sample_count = 0
    
    for i, data in enumerate(test_loader):
        if sample_count >= args.attention_samples:
            break
            
        x3d, x2d, y = data
        x3d, x2d, y = x3d.to(device), x2d.to(device), y.to(device)
        
        # Normalize data
        x3d_norm, x2d_norm, x2d_orig = normalizer.normalize(x3d, x2d)
        
        # Take only the first sample from each batch to avoid memory issues
        attention_data.append((x3d_norm[:1], x2d_norm[:1], x2d_orig[:1]))
        sample_count += x3d.shape[0]
        
        if i % 50 == 0:
            logger.info(f'Collected {sample_count} samples for attention analysis')
    
    logger.info(f'Collected {len(attention_data)} batches with {sample_count} samples total')
    
    # Analyze attention for each specified layer
    for layer_idx in args.attention_layers:
        logger.info(f'Analyzing attention for layer {layer_idx}...')
        
        all_attention_weights = []
        
        # Process samples in batches
        for x3d_norm, x2d_norm, x2d_orig in attention_data:
            try:
                # Get attention weights for this batch
                attention_weights = get_attention_weights(model, x3d_norm, x2d_norm, x2d_orig, layer_idx=layer_idx)
                
                if attention_weights is not None:
                    # Average over batch and heads dimensions
                    # Shape: (batch_size, heads, seq_len, seq_len) -> (seq_len, seq_len)
                    avg_attention = torch.mean(attention_weights, dim=(0, 1))
                    all_attention_weights.append(avg_attention.cpu())
                    
            except Exception as e:
                logger.warning(f'Error computing attention weights: {e}')
                continue
        
        if all_attention_weights:
            # Average attention weights across all samples
            avg_attention_matrix = torch.mean(torch.stack(all_attention_weights), dim=0)
            
            # Create visualization
            save_path = join(args.save_dir, f'attention_layer_{layer_idx}_simple.png')
            visualize_attention(avg_attention_matrix, layer_idx=layer_idx, save_path=save_path)
            
            # Save attention matrix data
            attention_data_path = join(args.save_dir, f'attention_matrix_layer_{layer_idx}.pickle')
            with open(attention_data_path, 'wb') as handle:
                pickle.dump(avg_attention_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f'Attention analysis for layer {layer_idx} completed and saved')
        else:
            logger.warning(f'No valid attention weights collected for layer {layer_idx}')
    
    logger.info('Attention analysis complete!')


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Simple attention analysis for ViT models')
    
    # Required arguments
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--save-dir', type=str, required=True, help='Directory to save results')
    
    # Model configuration (should match your training config)
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--dim-head', type=int, default=64, help='Dimension per head')
    parser.add_argument('--mlp-ratio', type=float, default=4.0, help='MLP ratio')
    parser.add_argument('--patch-size', type=int, default=1, help='Patch size')
    
    # Analysis configuration
    parser.add_argument('--attention-samples', type=int, default=500, help='Number of samples to analyze')
    parser.add_argument('--attention-layers', nargs='+', type=int, default=[0, 1, 2, 3], help='Layers to analyze')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loading workers')
    
    args = parser.parse_args()
    run_analysis(args)


if __name__ == '__main__':
    main() 