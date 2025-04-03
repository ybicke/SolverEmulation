#!/usr/bin/env python

import os
import re
import glob
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from torch.utils.data import DataLoader

#############################
# 1) Import the same stuff #
#############################
from column_files.vit_column_analysis2 import ViT
from train_column_vit import get_column_data_with_disk_cache, get_normalization_params, parser as train_parser
from data_loaders_new import IconColumnIterableDataset

# Start with the base parser from train_column_vit
parser = train_parser

# Add visualization-specific arguments
parser.add_argument('--max-batches', type=int, default=10, help='Maximum number of batches to process')
parser.add_argument('--dpi', type=int, default=300, help='DPI for saved figures')
parser.add_argument('--layer-idxs', nargs='+', type=int, default=[0, 1, 2, 3], help='Layer indices to visualize')
parser.add_argument('--head-idxs', nargs='+', type=int, default=None, help='Head indices to visualize (defaults to all heads)')
parser.add_argument('--output-dir', type=str, default=None, help='Path to save visualizations (defaults to MODEL_DIR/attention_analysis)')

args = parser.parse_args()

def main():
    ########################
    # 2) Config & Checkpoint
    ########################
    checkpoint_dir = args.save
    best_ckpt = os.path.join(checkpoint_dir, "best_model.pth")
    assert os.path.isfile(best_ckpt), f"No best_model.pth found in {checkpoint_dir}"
    
    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(checkpoint_dir, "attention_analysis")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ########################
    # 3) Load Stats & Model
    ########################
    dataset_dir = args.dataset
    stats_file = os.path.join(dataset_dir, 'normalizer_stats_per_feat.pickle')
    mean2d, var2d, mean3d, var3d = get_normalization_params(stats_file)

    print("Loading model...")
    model = ViT(
        num_cells=args.num_cells,
        patch_size=args.patch_size,
        dim=args.vit_hidden_dim,
        mlp_dim=args.vit_hidden_dim,
        depth=args.vit_layers,
        heads=args.vit_heads,
        channels_in=6,
        channels_out=4,
        height=70,
        dim_head=64,
        dropout=args.vit_dropout,
        emb_dropout=0.0,
        mean2d=mean2d,
        var2d=var2d,
        mean3d=mean3d,
        var3d=var3d,
        device=device
    ).to(device)

    print(f"Loading checkpoint from {best_ckpt}")
    ckpt_data = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()

    ###################################
    # 4) Build Your Test DataLoader   #
    ###################################
    print("Setting up test dataset...")
    FILENAMES = glob.glob(os.path.join(dataset_dir, '*.h5'))
    time_indices = [float(re.search(r'\_time_(.*?)\.h5', f).group(1)) for f in FILENAMES]
    sorted_files = [x for _,x in sorted(zip(time_indices, FILENAMES))]

    test_files = sorted_files[2220:]
    print(f"Found {len(test_files)} test files")

    print("Creating test loader...")
    test_loader = get_column_data_with_disk_cache(test_files, shuffle=False, num_workers=args.num_workers)

    ###################################
    # 5) Extract Attention, All Layers
    ###################################
    layer_idxs = args.layer_idxs
    all_attn = {ly: [] for ly in layer_idxs}
    
    max_batches = args.max_batches
    
    print(f"Processing up to {max_batches} test batches...")
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= max_batches:
            break
            
        x3d, x2d, _ = batch
        x3d, x2d = x3d.to(device), x2d.to(device)
        
        print(f"  Processing batch {batch_idx+1}/{max_batches}")
        for ly in layer_idxs:
            print(f"    Extracting attention for layer {ly}")
            attn = model.get_attention_weights(x3d, x2d, layer_idx=ly)
            all_attn[ly].append(attn.detach().cpu())
    
    for ly in layer_idxs:
        all_attn[ly] = torch.cat(all_attn[ly], dim=0)
        print(f"Layer {ly} attention tensor shape: {all_attn[ly].shape}")

    #################################
    # 6) Analyze or Average Attn    #
    #################################
    print("Generating visualizations...")
    
    num_heads = model.transformer.blocks[0].attention.heads
    head_idxs = args.head_idxs if args.head_idxs is not None else range(num_heads)
    
    for ly in layer_idxs:
        attn_tensor = all_attn[ly]
        mean_attn = attn_tensor.mean(dim=0)
        
        for head_idx in head_idxs:
            attn_map = mean_attn[head_idx].numpy()
            
            print(f"Layer {ly}, Head {head_idx}, attention map shape: {attn_map.shape}")
            
            plt.figure(figsize=(10, 8))
            plt.imshow(attn_map, cmap="viridis")
            plt.title(f"Average Attention, Layer {ly}, Head {head_idx}")
            plt.colorbar(label="Attention Weight")
            plt.xlabel("Key Position (Height Level)")
            plt.ylabel("Query Position (Height Level)")
            plt.grid(False)
            
            save_path = os.path.join(output_dir, f"attn_layer{ly}_head{head_idx}.png")
            plt.savefig(save_path, dpi=args.dpi, bbox_inches='tight')
            plt.close()
            print(f"  Saved to {save_path}")
    
    save_attn_path = os.path.join(output_dir, "attention_data.pt")
    torch.save(all_attn, save_attn_path)
    print(f"Saved raw attention data to {save_attn_path}")
    
    print(f"Analysis complete! Check results in: {output_dir}")

if __name__ == "__main__":
    main()