import torch.profiler as profiler
import torch

print(f"PyTorch version: {torch.__version__}")

with profiler.profile(activities=[
        profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                        record_shapes=True) as prof:
    x = torch.randn(2, 3)
    y = torch.relu(x)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))