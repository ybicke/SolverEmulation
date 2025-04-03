# Adopted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import matplotlib.pyplot as plt




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
        return self.net(x)



class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism without the pre-normalization"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        # For attention visualization
        self.attention_weights = None
        self.save_attention = False

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        
        # Save attention weights if requested
        if self.save_attention:
            self.attention_weights = attn.detach().clone()

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """A single transformer block with pre-layer normalization"""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        
        # Pre-normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Processing blocks
        self.attention = MultiHeadAttention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x


class Transformer(nn.Module):
    """Stack of transformer blocks with final layer normalization"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        
        # Create the specified number of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Pass through each transformer block sequentially
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        return self.norm(x)



class ViT(nn.Module):
        
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

        
        self.to_patch_embedding_3D = nn.Sequential(
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

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim, device=device))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

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

        x3d = self.to_patch_embedding_3D(x3d)
        x2d = self.to_patch_embedding_2D(x2d)

        x = torch.cat((x3d, x2d[:, None, :]), axis=-2)

        x += self.pos_embedding[:, :self.num_patches]
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.mlp_head(x)
        x = self.sigmoid(x)

        x = self._scale_output(x, x2d_org)
        x = x.squeeze()
        
        return x
    
    
        
    def get_attention_weights(self, x3d, x2d, layer_idx=0):
        """Extract attention weights from a specific layer for a batch of inputs"""
        # Enable attention saving for the specified layer
        attn_module = self.transformer.blocks[layer_idx].attention
        attn_module.save_attention = True
        
        # Forward pass to compute attention
        with torch.no_grad():
            self.forward(x3d, x2d)
        
        # Get the attention weights
        attention_weights = attn_module.attention_weights
        
        # Disable attention saving
        attn_module.save_attention = False
        
        return attention_weights
    
    

    def visualize_attention(self, test_loader, layer_idx=0):
        """Compute and visualize the average attention matrix over the test set"""
        attention_matrices = []
        
        # Iterate over the test set batches
        for x3d, x2d in test_loader:
            # Get the attention weights for the current batch
            attention_weights = self.get_attention_weights(x3d, x2d, layer_idx)
            
            # Average the attention weights over the batch and heads
            attention_matrix = attention_weights.mean(dim=(0, 1))
            
            attention_matrices.append(attention_matrix)
        
        # Compute the average attention matrix over all batches
        avg_attention_matrix = torch.stack(attention_matrices).mean(dim=0)
        
        # Visualize the average attention matrix
        plt.imshow(avg_attention_matrix.cpu(), cmap='hot', interpolation='nearest')
        plt.xlabel('Height')
        plt.ylabel('Height')
        plt.title(f'Average Attention Matrix (Layer {layer_idx})')
        plt.colorbar()
        plt.savefig(f'vit_attention_matrix_layer_{layer_idx}.png')



        
    