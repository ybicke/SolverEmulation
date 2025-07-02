#!/usr/bin/env python3
"""
Simple runner script for attention analysis with your specific model configuration

Usage:
    python run_simple_attention.py (from anywhere)
"""

import sys
import os
from os.path import join, dirname, abspath

# Add the current directory to the path so we can import the analysis script
script_dir = dirname(abspath(__file__))
sys.path.insert(0, script_dir)

def main():
    # Your specific model configuration
    model_path = "/mydata/deepcloud/yves/results_A_RadiativeFlux/results/vit_128_hrlu_0005_new/best_model.pth"
    dataset_path = "/mydata/deepcloud/salman/dataset/h5_data_all_chuncked"
    save_dir = "/mydata/deepcloud/yves/results_A_RadiativeFlux/results/vit_128_hrlu_0005_new/attention_analysis_simple"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please update the model_path in this script to point to your trained model.")
        sys.exit(1)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Please update the dataset_path in this script to point to your dataset.")
        sys.exit(1)
    
    print("Running simple attention analysis with configuration:")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Save dir: {save_dir}")
    print(f"Config: 128 dim, 4 layers, 6 heads, 64 dim_head")
    print()
    
    # Set up arguments as if they came from command line
    class Args:
        def __init__(self):
            self.model_path = model_path
            self.dataset = dataset_path
            self.save_dir = save_dir
            self.hidden_dim = 128
            self.layers = 4
            self.heads = 6
            self.dim_head = 64
            self.mlp_ratio = 4.0
            self.patch_size = 1
            self.attention_samples = 500
            self.attention_layers = [0, 1, 2, 3]
            self.batch_size = 4
            self.num_workers = 4
    
    # Create args object
    args = Args()
    
    try:
        # Import and run the analysis function directly
        from simple_attention_analysis import run_analysis
        run_analysis(args)
        print(f"\nAttention analysis completed! Results saved to: {save_dir}")
    except Exception as e:
        print(f"Error running attention analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 