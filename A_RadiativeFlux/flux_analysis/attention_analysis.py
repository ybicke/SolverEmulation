"""
Attention Analysis for Vision Transformer Models

This module provides tools for capturing, analyzing, and visualizing 
attention patterns in Vision Transformer models used for atmospheric flux prediction.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionCapture:
    """Captures attention weights from transformer layers during forward pass"""
    
    def __init__(self):
        self.attention_weights = {}
        self.hooks = []
    
    def register_hooks(self, model, target_layers=None):
        """Register forward hooks to capture attention weights"""
        
        # Register hooks specifically for ViT Attention modules  
        layer_count = 0
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            for i, layer_block in enumerate(model.transformer.layers):
                # Skip if we're only targeting specific layers
                if target_layers is not None and i not in target_layers:
                    continue
                    
                if len(layer_block) > 0:  # layer_block is [Attention, FeedForward]
                    attention_module = layer_block[0]  # First module is Attention
                    
                    # Monkey patch the attention module to capture weights
                    original_forward = attention_module.forward
                    
                    def make_patched_forward(layer_name, orig_forward):
                        def patched_forward(x):
                            x_norm = attention_module.norm(x)
                            qkv = attention_module.to_qkv(x_norm).chunk(3, dim=-1)
                            
                            # Import rearrange here to avoid issues
                            from einops import rearrange
                            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=attention_module.heads), qkv)
                            
                            dots = torch.matmul(q, k.transpose(-1, -2)) * attention_module.scale
                            attn = attention_module.softmax(dots)
                            attn = attention_module.dropout(attn)
                            
                            # Store attention weights
                            self.attention_weights[layer_name] = attn.detach().cpu()
                            
                            out = torch.matmul(attn, v)
                            out = rearrange(out, 'b h n d -> b n (h d)')
                            return attention_module.to_out(out)
                        return patched_forward
                    
                    attention_module.forward = make_patched_forward(f'layer_{i}', original_forward)
                    self.hooks.append((attention_module, original_forward))  # Store for cleanup
                    logger.info(f"Patched transformer layer {i} attention")
                    layer_count += 1
        
        logger.info(f"Patched {layer_count} attention modules")
    
    def remove_hooks(self):
        """Restore original forward methods"""
        for module, original_forward in self.hooks:
            module.forward = original_forward
        self.hooks.clear()
    
    def get_attention_weights(self):
        """Get captured attention weights"""
        return self.attention_weights
    
    def clear(self):
        """Clear stored attention weights"""
        self.attention_weights.clear()


class AttentionAnalyzer:
    """Analyzes and visualizes attention patterns"""
    
    def __init__(self, height_levels: int = 70):
        self.height_levels = height_levels
        self.height_labels = self._create_height_labels()
    
    def _create_height_labels(self) -> List[str]:
        """Create height labels for atmospheric levels"""
        # Height mapping: level 70 = 0 km (surface), level 0 = 65 km (TOA) 
        heights_km = np.linspace(65, 0, self.height_levels + 1)  # 71 levels including surface
        labels = []
        
        # Add surface token label
        labels.append("Surface")
        
        # Add atmospheric level labels with heights
        for i in range(self.height_levels):
            level = self.height_levels - 1 - i  # Reverse order for display
            height = heights_km[i]
            labels.append(f"{level} ({height:.0f} km)")
        
        return labels
    
    def aggregate_attention(self, attention_batches: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate attention weights across batches and heads"""
        # Stack all batches
        all_attention = torch.cat(attention_batches, dim=0)  # [total_samples, heads, seq_len, seq_len]
        
        # Average across samples and heads
        avg_attention = all_attention.mean(dim=(0, 1))  # [seq_len, seq_len]
        
        return avg_attention
    
    def plot_attention_matrix(self, attention_matrix: torch.Tensor, 
                            layer_name: str, save_path: str):
        """Create publication-quality attention matrix visualization"""
        
        # Convert to numpy and flip both dimensions for atmospheric convention
        attn_np = attention_matrix.numpy()
        attn_flipped = np.flipud(np.fliplr(attn_np))
        
        # Create figure with proper sizing for publication
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        
        # Create heatmap
        im = ax.imshow(attn_flipped, cmap='viridis', aspect='equal', 
                      vmin=0, vmax=attn_flipped.max())
        
        # Set up ticks and labels
        n_levels = len(self.height_labels)
        tick_positions = np.arange(0, n_levels, 10)  # Every 10 levels
        tick_labels = [self.height_labels[-(i+1)] for i in tick_positions]  # Reverse for flipped display
        
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=14)
        ax.set_yticklabels(tick_labels, fontsize=14)
        
        # Labels and title with increased font sizes
        ax.set_xlabel('Key (Information Source)', fontsize=16, labelpad=10)
        ax.set_ylabel('Query (Information Consumer)', fontsize=16, labelpad=10)
        ax.set_title(f'Attention Patterns - {layer_name}', fontsize=18, pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attention Weight', fontsize=14, labelpad=15)
        cbar.ax.tick_params(labelsize=12)
        
        # Improve layout
        plt.tight_layout()
        
        # Save with high quality
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Saved attention plot: {save_path}")


def analyze_model_attention(model: nn.Module, 
                          test_loader, 
                          normalizer,
                          save_dir: str,
                          num_samples: int = 500,
                          target_layers: List[int] = None) -> Dict[str, torch.Tensor]:
    """
    Main function to analyze attention patterns in a model
    
    Args:
        model: PyTorch model with attention mechanisms
        test_loader: DataLoader for test data
        normalizer: Data normalizer instance
        save_dir: Directory to save attention visualizations
        num_samples: Number of samples to analyze
    
    Returns:
        Dictionary mapping layer names to averaged attention matrices
    """
    
    # Only analyze ViT models
    if not hasattr(model, 'transformer'):
        logger.info("Model does not have transformer blocks, skipping attention analysis")
        return {}
    
    logger.info(f"Starting attention analysis on {num_samples} samples")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize components
    capture = AttentionCapture()
    
    # Determine height levels from the first batch
    first_batch = next(iter(test_loader))
    x3d_sample = first_batch[0]  # x3d is the first element: (x3d, x2d, y)
    height_levels = x3d_sample.size(-2)  # x3d has shape [batch, channels, height]
    analyzer = AttentionAnalyzer(height_levels=height_levels)
    logger.info(f"Detected {height_levels} height levels from data")
    
    # Register hooks
    capture.register_hooks(model, target_layers)
    
    model.eval()
    attention_data = {}
    sample_count = 0
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if sample_count >= num_samples:
                    break
                
                # Process batch
                x3d, x2d, y = batch
                x3d_norm, x2d_norm, x2d_orig = normalizer.normalize(x3d, x2d)
                
                # Clear previous attention weights
                capture.clear()
                
                # Forward pass to capture attention
                _ = model(x3d_norm, x2d_norm, x2d_orig)
                
                # Store attention weights
                current_attention = capture.get_attention_weights()
                
                # Accumulate attention weights by layer
                for layer_name, attn_weights in current_attention.items():
                    if layer_name not in attention_data:
                        attention_data[layer_name] = []
                    attention_data[layer_name].append(attn_weights)
                
                sample_count += x3d.size(0)
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed {sample_count} samples...")
        
    finally:
        # Always remove hooks
        capture.remove_hooks()
    
    # Analyze and save results
    final_attention_matrices = {}
    
    for layer_name, attention_batches in attention_data.items():
        if attention_batches:
            # Aggregate attention across all samples
            avg_attention = analyzer.aggregate_attention(attention_batches)
            final_attention_matrices[layer_name] = avg_attention
            
            # Create visualization
            save_path = os.path.join(save_dir, f'attention_{layer_name}.png')
            analyzer.plot_attention_matrix(avg_attention, layer_name, save_path)
    
    logger.info(f"Attention analysis complete. Analyzed {len(final_attention_matrices)} layers")
    return final_attention_matrices


def run_attention_analysis_if_enabled(model, test_loader, normalizer, 
                                    save_dir: str, num_samples: int, 
                                    enabled: bool = False,
                                    target_layers: List[int] = None):
    """
    Compact helper function for integration into training scripts
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        normalizer: Data normalizer
        save_dir: Save directory for plots
        num_samples: Number of samples to analyze
        enabled: Whether to run analysis
    """
    if not enabled:
        return {}
    
    try:
        return analyze_model_attention(model, test_loader, normalizer, save_dir, num_samples, target_layers)
    except Exception as e:
        logger.error(f"Attention analysis failed: {e}")
        return {}