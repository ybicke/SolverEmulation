import torch.profiler as profiler

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

class AFNO1D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(self, 
                 hidden_size,
                 num_blocks=8,
                 sparsity_threshold=0.01,
                 hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        
        # The AFNO1D layer uses a block diagonal matrix structure for its weight matrices. The weight matrices are divided
        # into smaller blocks along the diagonal.The block_size represents the size of each individual block within the block diagonal matrix.
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        # preparation for blockwhise processing: 2 for real and imag, hiddensize factor for output
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        with profiler.record_function("AFNO1D_forward"):
            bias = x

            with profiler.record_function("AFNO1D_rfft"):
                dtype = x.dtype
                x = x.float()
                B, N, C = x.shape
                x = torch.fft.rfft(x, dim=1, norm="ortho")
                x = x.reshape(B, N // 2 + 1, self.num_blocks, self.block_size)

            with profiler.record_function("AFNO1D_blockwise_processing"):
                o1_real = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
                o1_imag = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
                o2_real = torch.zeros(x.shape, device=x.device)
                o2_imag = torch.zeros(x.shape, device=x.device)

                total_modes = N // 2 + 1
                kept_modes = int(total_modes * self.hard_thresholding_fraction)

                with profiler.record_function("AFNO1D_einsum_o1_real"):
                    o1_real[:, :kept_modes] = F.relu(
                        torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[0]) - \
                        torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[1]) + \
                        self.b1[0]
                    )

                with profiler.record_function("AFNO1D_einsum_o1_imag"):
                    o1_imag[:, :kept_modes] = F.relu(
                        torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[0]) + \
                        torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[1]) + \
                        self.b1[1]
                    )

                with profiler.record_function("AFNO1D_einsum_o2_real"):
                    o2_real[:, :kept_modes] = (
                        torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[0]) - \
                        torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[1]) + \
                        self.b2[0]
                    )

                with profiler.record_function("AFNO1D_einsum_o2_imag"):
                    o2_imag[:, :kept_modes] = (
                        torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[0]) + \
                        torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[1]) + \
                        self.b2[1]
                    )

                x = torch.stack([o2_real, o2_imag], dim=-1)
                x = F.softshrink(x, lambd=self.sparsity_threshold)
                x = torch.view_as_complex(x)
                x = x.reshape(B, N // 2 + 1, C)

            with profiler.record_function("AFNO1D_irfft"):
                x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
                x = x.type(dtype)
                
            # Print profiling results after each forward pass
            #print("Profiling results for AFNO1D forward pass:")
            #print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
            #print()

            return x + bias