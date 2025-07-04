# Adopted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join

# classes

class Normalization(nn.Module):
    """ Normalize the input based on mean and std """    
    def __init__(self, std, mean, axis=None):
        super().__init__()
        self.std = std
        self.mean = mean
        self.axis = axis

    def forward(self, x):
        # TODO: only supports normalization on last dim
        # should be extended to custom dims
        assert x.size(dim=-1) == self.std.size(dim=0), \
            f'Dimension mismatch ( {x.size(dim=-1)} != {self.std.size(dim=0)})'
        return (x - self.mean)/self.std

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim) after passing through the feed-forward network.
        """
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        
        # Calculate the inner dimension of the attention mechanism
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        # Nr of heads and scaling factor for the dot product attention
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # If project_out is True, it consists of a linear layer and a dropout layer
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.attention_weights = None
        self.save_attention = True

        

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim) after applying attention and projection.
        """
        
        x = self.norm(x)

        # Linearly project the input to obtain query, key, and value matrices
        # Rearrange matrices to introduce the head dimension( batch_size, heads, sequence_length, dim_head)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Compute the dot product attention scores over the head dimension and scale them
        # (batch_size, heads, sequence_length, dim_head) × (batch_size, heads, dim_head, sequence_length) 
        # ⟶ (batch_size, heads, sequence_length, sequence_length)
        # order of the height dimension with the surface at the top and the highest pressure at the bottom.
        # Need to switch  ordering afterwards. 
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Store attention weights if requested
        if self.save_attention:
            self.attention_weights = attn.detach().clone()

    
        # Compute the weighted sum of the value matrix based on the attention weights
        # The shape of out will be (batch_size, sequence_length, heads * dim_head)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
 
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        """
        Args:
            dim (int): Dimension of the input and output.
            depth (int): Number of transformer layers.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            mlp_dim (int): Dimension of the feed-forward network.
            dropout (float): Dropout rate.
        """     
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        # List to hold the transformer layers and create the specified number of transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        """
         Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim) after passing through the transformer layers.
        """
        # Apply the attention and FF layer with residual connection
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT4(nn.Module):
    
        
    def __init__(
            self,
             num_cells,
             patch_size,
             dim=256,
             mlp_dim=256, # here the same for FF in AFNO 4 times dim in MLP
             depth=4,
             heads=6,
             channels_in=6,
             channels_out=4,
             height=71,
             dim_head=64,
             dropout=0.,
             emb_dropout=0.,
             swflx_idx=[2, 3],
             lwflx_idx=[0, 1], 
             cosmu0_idx=1, 
             tsfctrad_idx=5,
             mean2d=None,
             var2d=None, 
             mean3d=None, 
             var3d=None,
             device=None,
             is_test=False, # not used yet 
             *args, **kwargs
        ):
        super().__init__()

        self.is_test = is_test

        # Calculate the number of patches along the height dimension
        self.num_patches = int(height/patch_size)
        self.num_cells = num_cells
        self.height = height
        self.channels_out = channels_out
        self.swflx_idx = swflx_idx
        self.lwflx_idx = lwflx_idx
        self.cosmu0_idx = cosmu0_idx
        self.tsfctrad_idx = tsfctrad_idx

        # Add dummy vector for AFNO-style concatenation
        self.dummy_vector = nn.Parameter(torch.randn(1, 1, channels_in))

        # Initialize normalizers for 2D and 3D inputs
        self.normalizer2d = Normalization(std=torch.sqrt(var2d), mean=mean2d)
        self.normalizer3d = Normalization(std=torch.sqrt(var3d), mean=mean3d)

        # Update patch dimension to account for concatenated features (channels_in * 2)
        patch_dim = (channels_in * 2) * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p) f -> b h p f', p=patch_size),
            nn.Flatten(-2, -1),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Positional embedding and dropout layer
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim, device=device))
        self.dropout = nn.Dropout(emb_dropout)

        # Define the transformer encoder
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Define the MLP head for final output
        self.mlp_head = nn.Linear(dim, patch_size*channels_out)
        self.sigmoid = nn.Sigmoid()


    def _unscale_swflx(self, swflx, cosmu0):
        "Returns the scaled swflx to the original scale"
        return torch.where(
            cosmu0 >= torch.tensor(1e-4, dtype=torch.float32),
            swflx * (cosmu0 * 1400),
            0
        )

    def _unscale_lwflx(self, lwflx, tsfctrad):
        "Returns the scaled lwflx to the original scale"
        stefan_boltzmann_const = torch.tensor(5.670374419e-08, dtype=torch.float32)
        return torch.where(
            tsfctrad >= torch.tensor(1e-4, dtype=torch.float32),
            lwflx * torch.pow(tsfctrad, 4) * stefan_boltzmann_const,
            lwflx
        )
    
    def _scale_output(self, y_pred, x2d):
        """
        Scale the output to match the original data scale.
        
        Args:
            y_pred (torch.Tensor): Predicted output tensor of shape (batch_size, num_cells, height, channels_out).
            x2d (torch.Tensor): Original 2D input tensor of shape (batch_size, num_cells, channels_in).
        
        Returns:
            torch.Tensor: Scaled output tensor of shape (batch_size, num_cells, height, channels_out).
        """
        y_pred_scaled = []

        # Iterate over each channel in the last dimension of y_pred
        for i in range(y_pred.shape[-1]):
            
            # Extract the predicted values for the current channel
            f_pred = y_pred[..., i:i+1]
            
            # Scale shortwave flux if the current index is in swflx_idx
            if i in self.swflx_idx:
                
                # Extract cosmu0 values from x2d. Shape: (batch_size, num_cells)
                cosmu0 = x2d[..., self.cosmu0_idx]
                
                # Tile cosmu0 to match the shape of f_pred.  Shape: (batch_size, num_cells, 1, 1)
                # Tile to (batch_size, num_cells, height, 1)
                cosmu0 = torch.tile(
                    cosmu0[..., None, None], 
                    (1, 1, f_pred.shape[-2], 1)
                )
                # Unscale the shortwave flux and append to the list
                y_pred_scaled.append(self._unscale_swflx(f_pred, cosmu0))
                
            elif i in self.lwflx_idx:
                tsfctrad = x2d[..., self.tsfctrad_idx]
                tsfctrad = torch.tile(
                    tsfctrad[..., None, None], 
                    (1, 1, f_pred.shape[-2], 1)
                )
                y_pred_scaled.append(self._unscale_lwflx(f_pred, tsfctrad))
            else:
                y_pred_scaled.append(f_pred)

        y_pred = torch.cat(y_pred_scaled, dim=-1)
        return y_pred
    


    def forward(self, x3d, x2d):
        # Normalize the inputs
        x3d = self.normalizer3d(x3d)
        x2d_org = x2d.clone()
        x2d = self.normalizer2d(x2d)

        # AFNO-style concatenation scheme:
        # Repeat x2d along the height dimension to match the shape of x3d
        x2d_repeated = x2d.unsqueeze(1).repeat(1, x3d.shape[1], 1)
        
        # Concatenate x3d and x2d_repeated along the feature dimension
        x_concat = torch.cat((x3d, x2d_repeated), dim=-1)
        
        # Repeat the dummy vector along the batch dimension, same random nr across the batch
        dummy_vector_repeated = self.dummy_vector.repeat(x3d.shape[0], 1, 1)
        
        # Concatenate the repeated dummy vector with the normalized x2d
        dummy_vector_with_x2d = torch.cat((dummy_vector_repeated, x2d.unsqueeze(1)), dim=-1)
        
        # Concatenate the resulting tensor as an additional height level
        x = torch.cat((x_concat, dummy_vector_with_x2d), dim=1)
        
        # Embed the concatenated features using the patch embedding layer
        x = self.to_patch_embedding(x)

        # Add positional embedding and apply dropout
        x += self.pos_embedding[:, :x.shape[1]]
        x = self.dropout(x)
        
        # Attention weights from the transformer
        x = self.transformer(x)
        
        x = self.mlp_head(x)
        x = self.sigmoid(x)

        x = self._scale_output(x, x2d_org)
        x =  x.squeeze()
        return x
        



    def get_attention_weights(self, x3d, x2d, layer_idx=0):
        """Extract attention weights from a specific layer for a batch of inputs"""
        # Get the attention module from the transformer layers
        attn_module = self.transformer.layers[layer_idx][0]  # First element in the layer tuple is Attention
        
        # Enable attention saving for the specified layer
        attn_module.save_attention = True
        
        # Forward pass to compute attention
        with torch.no_grad():
            self.forward(x3d, x2d)
        
        # Get the attention weights
        attention_weights = attn_module.attention_weights
        
        # Disable attention saving after one calculation
        attn_module.save_attention = False
                
        return attention_weights
    
    

    def visualize_attention(self, avg_attention_matrix, layer_idx=0, save_path=None):
        """Compute and visualize the average attention matrix over the test set"""
        
        # Clean mapping: Position 0 = Surface token, Position 1-71 = x3d[0-70] (surface to TOA)
        # Height mapping for atmospheric levels (your original was correct!)
        height_mapping = {
            0: 65, 10: 39, 20: 25, 30: 15, 40: 8, 50: 4, 60: 1, 70: 0
        }
        
        # Create tick positions and labels with level and height in brackets
        tick_positions = list(height_mapping.keys())
        tick_labels = [f"{level} ({height_mapping[level]} km)" for level in tick_positions]
        
        # Flip both dimensions to have surface at bottom-right, TOA at top-left (your original logic!)
        flipped_attn_rows = torch.flip(avg_attention_matrix, dims=[0])
        flipped_attn_cols = torch.flip(flipped_attn_rows, dims=[1])
        flipped_attn = flipped_attn_cols

        # Create publication-ready plot with larger size for better text readability
        plt.figure(figsize=(12, 10))
        
        # Plot the attention matrix
        im = plt.imshow(flipped_attn, cmap='plasma', interpolation='nearest', aspect='equal')
        
        # Set ticks and labels for both axes with larger font
        plt.xticks(tick_positions, tick_labels, fontsize=18, rotation=45)
        plt.yticks(tick_positions, tick_labels, fontsize=18)
        
        # Add proper labels with interpretation and more spacing
        plt.xlabel('Key Position', fontsize=19, labelpad=15)
        plt.ylabel('Query Position', fontsize=19, labelpad=15)
        plt.title(f'Attention Matrix - Layer {layer_idx}', 
                 fontsize=19, pad=25)
        
        # Add colorbar with label and more spacing
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', fontsize=19, labelpad=20)
        cbar.ax.tick_params(labelsize=18)
        
        # Add grid for better readability
        plt.grid(True, alpha=1, linestyle='--', linewidth=0.5)
        
        
        # Improve layout and save
        plt.tight_layout()
        save_filename = save_path if save_path else f'vit_attention_matrix_layer_{layer_idx}_publication.png'
        plt.savefig(save_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Publication-ready attention matrix saved as: {save_filename}")
        print(f"Matrix shape: {avg_attention_matrix.shape}")
        print("Atmospheric visualization: Surface (0 km) at bottom-right, TOA (65 km) at top-left")

    def visualize_attention_grid(self, attention_matrices_dict, save_path=None):
        """Visualize multiple attention matrices in a grid layout with 2 plots per row
        Publication-optimized with one colorbar per row that matches plot size
        
        Args:
            attention_matrices_dict: Dictionary with layer_idx as key and attention matrix as value
            save_path: Path to save the figure
        """
        
        # Get layer indices and sort them
        layer_indices = sorted(attention_matrices_dict.keys())
        num_layers = len(layer_indices)
        
        if num_layers == 0:
            print("No attention matrices to visualize")
            return
        
        # Calculate global min/max for consistent colorbar scaling
        all_values = []
        flipped_matrices = {}
        
        for layer_idx in layer_indices:
            avg_attention_matrix = attention_matrices_dict[layer_idx]
            # Flip both dimensions to have surface at bottom-right, TOA at top-left
            flipped_attn_rows = torch.flip(avg_attention_matrix, dims=[0])
            flipped_attn_cols = torch.flip(flipped_attn_rows, dims=[1])
            flipped_attn = flipped_attn_cols
            flipped_matrices[layer_idx] = flipped_attn
            all_values.extend(flipped_attn.flatten().tolist())
        
        # Set global color limits with better contrast
        vmin = min(all_values)
        vmax = max(all_values)
        
        # Enhanced contrast: use power-law normalization to make attention patterns more pronounced
        # Power < 1 enhances low values, power > 1 enhances high values
        # norm = PowerNorm(gamma=0.8, vmin=vmin, vmax=vmax)  # gamma=0.5 enhances mid-range values
        
        # Calculate grid dimensions (2 plots per row)
        cols = 2
        rows = (num_layers + cols - 1) // cols  # Ceiling division
        
        # Height mapping for atmospheric levels
        height_mapping = {
            0: 65, 10: 39, 20: 25, 30: 15, 40: 8, 50: 4, 60: 1, 70: 0
        }
        
        # Create tick positions and labels
        tick_positions = list(height_mapping.keys())
        tick_labels = [f"{level} ({height_mapping[level]} km)" for level in tick_positions]
        
        # Publication-optimized figure size with extra space for external colorbars
        fig, axes = plt.subplots(rows, cols, figsize=(24, 10 * rows))
        
        # Handle case where we have only one row
        if rows == 1:
            axes = axes.reshape(1, -1) if num_layers > 1 else [axes]
        
        # Store image mappables for row-wise colorbars
        row_ims = {}  # Store one image per row for colorbar creation
        
        # Plot each attention matrix
        for idx, layer_idx in enumerate(layer_indices):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            flipped_attn = flipped_matrices[layer_idx]
            
            # Plot the attention matrix with enhanced contrast
            # Use 'hot' colormap for better contrast (black->red->yellow->white)
            im = ax.imshow(flipped_attn, cmap='plasma', 
                           interpolation='nearest', aspect='equal',
                           vmin=vmin, vmax=vmax, 
                           #norm=norm
                           )
            
            # Store first image of each row for colorbar
            if col == 0:
                row_ims[row] = im
            
            # Set ticks with larger fonts
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.tick_params(axis='both', which='major', labelsize=18)
            
            # Strategic label placement for publication
            is_leftmost = (col == 0)
            is_bottom_row = (row == rows - 1)
            is_last_row_with_data = (idx >= num_layers - cols)  # Handle incomplete last row
            
            # Y-axis labels only on leftmost plots
            if is_leftmost:
                ax.set_yticklabels(tick_labels, fontsize=18)
                ax.set_ylabel('Query Position', fontsize=22, labelpad=20)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel('')
            
            # X-axis labels only on bottom row
            if is_bottom_row or is_last_row_with_data:
                ax.set_xticklabels(tick_labels, fontsize=18, rotation=45, ha='right')
                ax.set_xlabel('Key Position', fontsize=22, labelpad=20)
            else:
                ax.set_xticklabels([])
                ax.set_xlabel('')
            
                        # Title with larger font
            ax.set_title(f'Layer {layer_idx+1}', fontsize=24, pad=10)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.8, linestyle='--', linewidth=0.5)
        
        # Add row-wise colorbars that match plot size
        for row in range(rows):
            if row in row_ims:  # Only add colorbar if row has data
                # Get the axes for this row
                if rows == 1:
                    row_axes = [axes] if num_layers == 1 else axes
                else:
                    row_axes = [axes[row, col] for col in range(cols) 
                              if row * cols + col < num_layers]
                
                # Create colorbar for this row - perfectly aligned with plot y-axis edges
                # Use manual positioning to ensure exact alignment with plot edges
                
                # Get the position of the rightmost plot in this row
                rightmost_ax = row_axes[-1] if len(row_axes) > 1 else row_axes[0]
                
                # Create divider for precise positioning
                divider = make_axes_locatable(rightmost_ax)
                cax = divider.append_axes("right", size="3%", pad=0.1)
                
                # Create colorbar with exact alignment and explicit range
                cbar = plt.colorbar(row_ims[row], cax=cax)
                cbar.set_label('Attention Weight', fontsize=20, labelpad=25)
                cbar.ax.tick_params(labelsize=18)
                
                # Ensure colorbar uses correct range
                cbar.mappable.set_clim(vmin, vmax)
                
                # Set consistent number of ticks with explicit range
                tick_values = np.linspace(vmin, vmax, 5)
                cbar.set_ticks(tick_values)
                cbar.set_ticklabels([f'{val:.3f}' for val in tick_values])
        
        # Hide any unused subplots
        if num_layers < rows * cols:
            for idx in range(num_layers, rows * cols):
                row = idx // cols
                col = idx % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                ax.set_visible(False)
        
        # Improve layout and save - minimal spacing, plots very close together
        plt.subplots_adjust(hspace=0.08, wspace=0.000001, left=0.12, right=0.82, top=0.92, bottom=0.15)
        save_filename = save_path if save_path else 'vit_attention_matrices_grid_publication.png'
        plt.savefig(save_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Publication-ready attention matrices grid saved as: {save_filename}")
        print(f"Grid layout: {rows} rows x {cols} columns")
        print(f"Layers visualized: {layer_indices}")
        print(f"Colorbar type: per-row with range [{vmin:.4f}, {vmax:.4f}]")
        print(f"Colormap: 'plasma' with PowerNorm(gamma=0.5) for enhanced contrast")
        print(f"Layout: Minimal spacing (hspace=0.08, wspace=0.02), row-fitted colorbars")
        print("Atmospheric visualization: Surface (0 km) at bottom-right, TOA (65 km) at top-left")

    def visualize_attention_average(self, attention_matrices_dict, save_path=None):
        """Visualize the average attention matrix across all layers
        
        Args:
            attention_matrices_dict: Dictionary with layer_idx as key and attention matrix as value
            save_path: Path to save the figure
        """
        
        # Get layer indices and sort them
        layer_indices = sorted(attention_matrices_dict.keys())
        num_layers = len(layer_indices)
        
        if num_layers == 0:
            print("No attention matrices to visualize")
            return
        
        print(f"Averaging attention across {num_layers} layers: {layer_indices}")
        
        # Flip and collect all matrices
        flipped_matrices = []
        for layer_idx in layer_indices:
            avg_attention_matrix = attention_matrices_dict[layer_idx]
            # Flip both dimensions to have surface at bottom-right, TOA at top-left
            flipped_attn_rows = torch.flip(avg_attention_matrix, dims=[0])
            flipped_attn_cols = torch.flip(flipped_attn_rows, dims=[1])
            flipped_attn = flipped_attn_cols
            flipped_matrices.append(flipped_attn)
        
        # Stack and average across all layers
        stacked_matrices = torch.stack(flipped_matrices, dim=0)
        averaged_attention = torch.mean(stacked_matrices, dim=0)
        
        # Height mapping for atmospheric levels
        height_mapping = {
            0: 65, 10: 39, 20: 25, 30: 15, 40: 8, 50: 4, 60: 1, 70: 0
        }
        
        # Create tick positions and labels with level and height in brackets
        tick_positions = list(height_mapping.keys())
        tick_labels = [f"{level} ({height_mapping[level]} km)" for level in tick_positions]
        
        # Create publication-ready plot with larger size for better text readability
        plt.figure(figsize=(12, 10))
        
        # Plot the averaged attention matrix
        im = plt.imshow(averaged_attention, cmap='plasma', interpolation='nearest', aspect='equal')
        
        # Set ticks and labels for both axes with larger font
        plt.xticks(tick_positions, tick_labels, fontsize=18, rotation=45)
        plt.yticks(tick_positions, tick_labels, fontsize=18)
        
        # Add proper labels with interpretation and more spacing
        plt.xlabel('Key Position', fontsize=19, labelpad=15)
        plt.ylabel('Query Position', fontsize=19, labelpad=15)
        plt.title(f'Average Attention Matrix (Layers {min(layer_indices)+1}-{max(layer_indices)+1})', 
                 fontsize=19, pad=25)
        
        # Add colorbar with label and more spacing
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Average Attention Weight', fontsize=19, labelpad=20)
        cbar.ax.tick_params(labelsize=18)
        
        # Add grid for better readability
        plt.grid(True, alpha=1, linestyle='--', linewidth=0.5)
        
        # Improve layout and save
        plt.tight_layout()
        save_filename = save_path if save_path else f'vit_attention_matrix_average_layers_{min(layer_indices)+1}-{max(layer_indices)+1}_publication.png'
        plt.savefig(save_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Publication-ready averaged attention matrix saved as: {save_filename}")
        print(f"Matrix shape: {averaged_attention.shape}")
        print(f"Averaged over {num_layers} layers")
        print("Atmospheric visualization: Surface (0 km) at bottom-right, TOA (65 km) at top-left")
        
        # Print some statistics about the averaged attention
        print(f"Average attention statistics:")
        print(f"  Min: {averaged_attention.min():.4f}")
        print(f"  Max: {averaged_attention.max():.4f}")
        print(f"  Mean: {averaged_attention.mean():.4f}")
        print(f"  Std: {averaged_attention.std():.4f}")