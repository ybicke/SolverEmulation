#!/usr/bin/env python3
"""
Standalone script for attention analysis of trained flux models

Usage:
    cd SolverEmulation/A_RadiativeFlux
    python -m flux_analysis.analyze_attention --model-path /path/to/best_model.pth --dataset /path/to/data --save-dir /path/to/results
"""

import argparse
import pickle
import torch
from os.path import join


from flux_data_loader import UnifiedFluxDataset  
from utils.data_utils import DataNormalizer
from flux_analysis.attention_analysis import analyze_model_attention
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Analyze attention patterns in trained flux models')
    
    # Required arguments
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--save-dir', type=str, required=True, help='Directory to save attention analysis results')
    
    # Model configuration (should match training config)
    parser.add_argument('--model', type=str, default='vit', help='Model type')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--layers', type=int, default=4, help='Number of layers') 
    parser.add_argument('--heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--patch-size', type=int, default=1, help='Patch size')
    parser.add_argument('--dim-head', type=int, default=64, help='Dimension per head')
    parser.add_argument('--mlp-ratio', type=float, default=4.0, help='MLP ratio')
    
    # Data configuration
    parser.add_argument('--channel-3d', type=int, default=6, help='3D input channels')
    parser.add_argument('--channel-2d', type=int, default=6, help='2D input channels') 
    parser.add_argument('--channel-out', type=int, default=4, help='Output channels')
    parser.add_argument('--height-in', type=int, default=70, help='Height levels')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loading workers')
    
    # Analysis configuration
    parser.add_argument('--attention-samples', type=int, default=500, help='Number of samples to analyze')
    parser.add_argument('--dataset-type', type=str, default='triangle', choices=['triangle', 'full'])
    parser.add_argument('--triangle-id', type=int, default=0, help='Triangle ID')
    parser.add_argument('--mode', type=str, default='1d', choices=['1d', '3d'])
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load normalization parameters
    stats_file = join(args.dataset, 'normalizer_stats_per_feat.pickle')
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)
        mean2d = torch.tensor(stats['mean2d'], dtype=torch.float32).to(device)
        var2d = torch.tensor(stats['var2d'], dtype=torch.float32).to(device)
        mean3d = torch.tensor(stats['mean3d'], dtype=torch.float32).to(device)
        var3d = torch.tensor(stats['var3d'], dtype=torch.float32).to(device)
    
    normalizer = DataNormalizer(mean2d, var2d, mean3d, var3d, device=device)
    
    # Create model
    if args.model == 'vit':
        from models_1d.vit import ViT
        model = ViT(
            patch_size=args.patch_size,
            dim=args.hidden_dim,
            mlp_dim=int(args.hidden_dim * args.mlp_ratio),
            depth=args.layers,
            heads=args.heads,
            channel_3d=args.channel_3d,
            channel_2d=args.channel_2d,
            channel_out=args.channel_out,
            height_in=args.height_in,
            dim_head=args.dim_head,
            dropout=0.0,
            emb_dropout=0.0,
            device=device
        ).to(device)
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    # Load model weights
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test dataset (using a subset of files)
    import glob
    import re
    filenames = glob.glob(join(args.dataset, '*.h5'))
    time_indices = [float(re.search(r'_time_(.*?)\.h5', f).group(1)) for f in filenames]
    sorted_files = [x for _, x in sorted(zip(time_indices, filenames))]
    test_files = sorted_files[2220:2300]  # Use subset for analysis
    
    # Create data loader
    dataset = UnifiedFluxDataset(
        filenames=test_files,
        mode=args.mode,
        dataset_type=args.dataset_type,
        triangle_id=args.triangle_id,
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
    
    # Run attention analysis
    print("Starting attention analysis...")
    attention_matrices = analyze_model_attention(
        model=model,
        test_loader=test_loader,
        normalizer=normalizer,
        save_dir=args.save_dir,
        num_samples=args.attention_samples
    )
    
    print(f"Analysis complete! Results saved to {args.save_dir}")
    print(f"Found attention patterns for {len(attention_matrices)} layers")

if __name__ == '__main__':
    main() 