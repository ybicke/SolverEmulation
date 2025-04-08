# Adopted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# classes

class Normalization(nn.Module):
    """ Normalize the input based on mean and std """    
    def __init__(self, std, mean):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, x):
        assert x.size(dim=-1) == self.std.size(dim=0), f'Dimension mismatch'
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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        

    def forward(self, x):

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.softmax(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
 
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
  
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x) 
        # Norm after blocks not standard in vanilla ViT

class ViT(nn.Module):
        
    def __init__(
            self,
             num_cells,
             patch_size,
             dim=256,
             mlp_dim=256, 
             depth=4,
             heads=6,
             channels_in=12,
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
             *args, **kwargs
        ):
        super().__init__()


        self.num_patches = int(height/patch_size)
        self.num_cells = num_cells  
        self.height = height
        self.channels_out = channels_out
        self.swflx_idx = swflx_idx
        self.lwflx_idx = lwflx_idx
        self.cosmu0_idx = cosmu0_idx
        self.tsfctrad_idx = tsfctrad_idx
        

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
        
        self.dummy_vector = nn.Parameter(torch.randn(1, 1, 6))

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim, device=device))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(dim, patch_size*channels_out)
        self.sigmoid = nn.Sigmoid()


    def _unscale_swflx(self, swflx, cosmu0):
        """Returns the scaled swflx to the original scale"""
        return torch.where(
            cosmu0 >= torch.tensor(1e-4, dtype=torch.float32),
            swflx * (cosmu0 * 1400),
            0
        )

    def _unscale_lwflx(self, lwflx, tsfctrad):
        """Returns the scaled lwflx to the original scale"""
        stefan_boltzmann_const = torch.tensor(5.670374419e-08, dtype=torch.float32)
        return torch.where(
            tsfctrad >= torch.tensor(1e-4, dtype=torch.float32),
            lwflx * torch.pow(tsfctrad, 4) * stefan_boltzmann_const,
            lwflx
        )
    
    def _scale_output(self, y_pred, x2d):
        """Scale the model output based on the input 2D features"""
        y_pred_scaled = []

        for i in range(y_pred.shape[-1]):
            
            f_pred = y_pred[..., i:i+1]
            
            if i in self.swflx_idx:
                
                cosmu0 = x2d[..., self.cosmu0_idx]
                
                cosmu0 = torch.tile(
                    cosmu0[..., None, None], 
                    (1, 1, f_pred.shape[-2], 1)
                )
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

        # Broadcast each 2d feature along the height column
        x2d_repeated = x2d.unsqueeze(1).repeat(1, x3d.shape[1], 1)
        x_concat = torch.cat((x3d, x2d_repeated), dim=-1)
        
        dummy_vector_repeated = self.dummy_vector.repeat(x3d.shape[0], 1, 1)
        concat_with_x2d = torch.cat((dummy_vector_repeated, x2d.unsqueeze(1)), dim=-1)
        
        x_concat = torch.cat((concat_with_x2d, x_concat), dim=1)
        x = self.to_patch_embedding(x_concat)
        
        
        x += self.pos_embedding[:, :self.num_patches]
        x = self.dropout(x)
        
        x = self.transformer(x)

        x = self.mlp_head(x)
        x = self.sigmoid(x)
        
        x = self._scale_output(x, x2d_org)
        
        return x.squeeze()
        
    
    



