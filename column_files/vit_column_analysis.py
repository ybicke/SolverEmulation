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

        # Initialize normalizers for 2D and 3D inputs
        self.normalizer2d = Normalization(std=torch.sqrt(var2d), mean=mean2d)
        self.normalizer3d = Normalization(std=torch.sqrt(var3d), mean=mean3d)

        patch_dim = channels_in  * patch_size


        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p) f -> b h p f', p=patch_size),
            nn.Flatten(-2, -1),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.to_patch_embedding_2D = nn.Sequential(
            nn.Linear(channels_in, dim),
            nn.LayerNorm(dim)
        )


        # Positional embedding and dropout layer
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim, device=device))
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
 

        x3d = self.normalizer3d(x3d)
        x2d_org = x2d.clone()
        x2d = self.normalizer2d(x2d)


        x3d = self.to_patch_embedding(x3d)
        x2d = self.to_patch_embedding_2D(x2d)


        # x = torch.cat((x3d, x2d[:, None, :]), axis=-2)
        # concatrenate the surface from below!
        x = torch.cat((x2d[:, None, :], x3d), axis=-2)

        x += self.pos_embedding[:, :self.num_patches]
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
        
        # Flip the first dimension (rows) to have the surface level at the bottom
        flipped_attn_rows = torch.flip(avg_attention_matrix, dims=[0])
        # Flip the second dimension (columns) to have the surface level on the right
        flipped_attn_cols = torch.flip(flipped_attn_rows, dims=[1])
        flipped_attn = flipped_attn_cols

        # Plot the reshaped attention matrix
        plt.figure(figsize=(8, 8))
        plt.imshow(flipped_attn, cmap='plasma', interpolation='nearest')
        # viridis, plasma
        # Invert the y-axis so row 0 is at the top visually
        # plt.gca().invert_yaxis()
        plt.xlabel('Height')
        plt.ylabel('Height')
        plt.title(f'Average Attention Matrix (Layer {layer_idx})')
        plt.colorbar()
        plt.savefig(f'vit_attention_matrix_layer_{layer_idx}.png') # Save to the specified path
        plt.close()