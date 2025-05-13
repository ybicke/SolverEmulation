import torch
import lightning as L
import time
import logging
import wandb


# from tqdne.autoencoder import LithningAutoencoder
#from tqdne.nn import append_dims


# Since tqdne isn't available, define append_dims locally
def append_dims(x, target_dims):
    """Appends dimensions to x until it has target_dims dimensions."""
    for _ in range(target_dims - x.dim()):
        x = x.unsqueeze(-1)
    return x




class EDM:
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5
    P_mean: float = -1.2
    P_std: float = 1.2
    S_churn: float = 40
    S_min: float = 0.05
    S_max: float = 50
    S_noise: float = 1.003

    def __init__(self, sigma_min=None, sigma_max=None, sigma_data=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def sigma(self, eps):
        return (eps * self.P_std + self.P_mean).exp()

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def skip_scaling(self, sigma):
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def out_scaling(self, sigma):
        return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5

    def in_scaling(self, sigma):
        return 1 / (sigma**2 + self.sigma_data**2) ** 0.5

    def noise_conditioning(self, sigma):
        return 0.25 * sigma.log()

    def sampling_sigmas(self, num_steps, device=None):
        rho_inv = 1 / self.rho
        step_idxs = torch.arange(num_steps, device=device)
        sigmas = (
            self.sigma_max**rho_inv
            + step_idxs / (num_steps - 1) * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
        ) ** self.rho
        return torch.cat([sigmas, torch.zeros_like(sigmas[:1])])  # add sigma=0

    def sigma_hat(self, sigma, num_steps):
        gamma = (
            min(self.S_churn / num_steps, 2**0.5 - 1) if self.S_min <= sigma <= self.S_max else 0
        )
        return sigma + gamma * sigma


class LightningEDM(L.LightningModule):
    """A Pyth Lightning module of the EDM model [1].

    Parameters
    ----------
    unet_config : dict
        The configuration for the U-Net model.
    optimizer_params : dict
        A dictionary of parameters for the optimizer.
    num_sampling_steps : int, optional
        The number of sampling steps during inference.
    deterministic_sampling : bool, optional
        If True, use deterministic sampling instead of stochastic sampling.
        Stochastic sampling can be more accurate but usually requires more (e.g. 256) steps.
    edm : EDM, optional
        The EDM model parameters.

    References
    ----------
    [1] Elucidating the Design Space of Diffusion-Based Generative Models
    [2] High-Resolution Image Synthesis with Latent Diffusion Models
    """

    def __init__(
        self,
        # unet_config: dict,
        unet,
        optimizer_params: dict,
        num_sampling_steps: int = 25,
        deterministic_sampling: bool = True,
        edm: EDM = EDM(),
        # autoencoder: None | LithningAutoencoder = None,
    ):
        super().__init__()

        # self.unet = UNetModel(**unet_config)
        self.unet = unet
        self.optimizer_params = optimizer_params
        self.num_sampling_steps = num_sampling_steps
        self.deterministic_sampling = deterministic_sampling
        self.edm = edm
        self.epoch_start_time = None
        self.last_log_time = None
        self.batch_times = []
        
        # Track accumulated losses
        self.train_loss_values = []
        self.train_mae_values = []
        self.val_loss_values = []
        self.val_mae_values = []
        
        # Current step counter
        self.global_step_counter = 0

        # It's redundant since the UNet weights are already saved in the state_dict
        self.save_hyperparameters(ignore=["autoencoder", "unet"])

    def on_train_epoch_start(self):
        """Log at the start of each training epoch"""
        self.epoch_start_time = time.time()
        self.last_log_time = time.time()
        self.batch_times = []
        self.train_loss_values = []
        self.train_mae_values = []
        print(f"Epoch {self.current_epoch} started")
    
    def on_validation_epoch_start(self):
        """Reset validation metrics"""
        self.val_loss_values = []
        self.val_mae_values = []
    
    def on_train_epoch_end(self):
        """Log at the end of each training epoch"""
        epoch_time = time.time() - self.epoch_start_time
        avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        
        # Calculate average loss and MAE for the epoch
        avg_loss = sum(self.train_loss_values) / len(self.train_loss_values) if self.train_loss_values else 0
        avg_mae = sum(self.train_mae_values) / len(self.train_mae_values) if self.train_mae_values else 0
        
        # Log to Lightning (will go to loggers)
        self.log("train_loss", avg_loss, prog_bar=True, on_epoch=True)
        self.log("train_mae", avg_mae, prog_bar=True, on_epoch=True)
        self.log("epoch_time", epoch_time, prog_bar=False, on_epoch=True)
        
        # Log metrics to wandb if available
        if wandb.run is not None:
            wandb.log({
                'epoch': self.current_epoch, 
                'train_loss': avg_loss,
                'train_mae': avg_mae,
                'epoch_time': epoch_time,
                'avg_batch_time_ms': avg_batch_time * 1000,  # Convert to ms
            })
        
        print(f"Epoch {self.current_epoch} completed in {epoch_time:.2f}s. "
              f"Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, "
              f"Batch time: {avg_batch_time*1000:.1f}ms")
    
    def on_validation_epoch_end(self):
        """Log validation results"""
        # Calculate average validation metrics
        avg_val_loss = sum(self.val_loss_values) / len(self.val_loss_values) if self.val_loss_values else 0
        avg_val_mae = sum(self.val_mae_values) / len(self.val_mae_values) if self.val_mae_values else 0
        
        # Log to Lightning (will go to loggers)
        self.log("val_loss", avg_val_loss, prog_bar=True, on_epoch=True)
        self.log("val_mae", avg_val_mae, prog_bar=True, on_epoch=True)
        
        # Log validation metrics to wandb
        if wandb.run is not None:
            wandb.log({
                'epoch': self.current_epoch,
                'val_loss': avg_val_loss,
                'val_mae': avg_val_mae,
            })
        
        print(f"Validation - Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.4f}")
        
        

    def forward(self, sample, sigma, cond=None):
        """Make a forward pass through the network with skip connection.
        
        Args:
            sample: The noisy flux sample to denoise [B, 71, 4]
            sigma: Noise level
            cond: Conditioning dictionary with physical features:
                - x3d_norm: 3D atmospheric features
                - x2d_norm: 2D global features 
                - x2d_orig: Original 2D features (for output scaling)
        
        Returns:
            Denoised flux prediction
        """
        dim = sample.dim()
        # Scale the input according to the noise level
        sample_in = sample * append_dims(self.edm.in_scaling(sigma), dim)
        
        # Get the noise conditioning signal
        noise_cond = self.edm.noise_conditioning(sigma)
        
        # Pass to UNet for denoising, with parameters in the expected order
        out = self.unet(
            noisy_sample=sample_in,  # Noisy version of flux
            time_cond=noise_cond,    # Time conditioning (noise level)
            cond=cond                # Physical features dictionary
        )
        
        # Apply skip connection and output scaling
        skip = append_dims(self.edm.skip_scaling(sigma), dim) * sample
        return out * append_dims(self.edm.out_scaling(sigma), dim) + skip


    def step(self, batch, batch_idx):
        """A single step in the training loop.
        
        Args:
            batch: Dictionary containing:
                - sample: Target flux values [B, 71, 4]
                - cond: Dictionary with conditioning information
                   - x3d_norm: Normalized 3D features
                   - x2d_norm: Normalized 2D features
                   - x2d_orig: Original 2D features
        """
        # Extract flux target (to be denoised) and physical conditioning
        sample = batch["sample"]  # Flux values [B, 71, 4]
        cond = batch["cond"]      # Physical features dictionary
        
        # Add noise to the sample
        eps = torch.randn(sample.shape[0], device=self.device)
        sigma = self.edm.sigma(eps)
        noise = torch.randn_like(sample) * append_dims(sigma, sample.dim())
        
        # Get model prediction
        pred = self(
            sample=sample + noise,  # Noisy version of flux
            sigma=sigma,            # Noise level
            cond=cond               # Physical features dictionary
        )

        # Calculate loss
        loss = (pred - sample) ** 2
        loss_weight = append_dims(self.edm.loss_weight(sigma), loss.dim())

        # Calculate weighted loss
        weighted_loss = (loss * loss_weight).mean()
        
        # Calculate MAE (for monitoring)
        mae = torch.abs(pred - sample).mean()
        
        return weighted_loss, mae

    def training_step(self, batch, batch_idx):
        self.global_step_counter += 1
        batch_start = time.time()
        
        # Get loss and MAE
        loss, mae = self.step(batch, batch_idx)
        
        # Log to Lightning (will go to loggers including WandB)
        self.log("training/loss", loss.item(), prog_bar=False, on_step=True)
        self.log("training/mae", mae.item(), prog_bar=False, on_step=True)
        self.log("training/step", self.global_step_counter, prog_bar=False, on_step=True)
        
        # Store for epoch-end calculation
        self.train_loss_values.append(loss.item())
        self.train_mae_values.append(mae.item())
        
        # Store batch processing time
        batch_time = time.time() - batch_start
        self.batch_times.append(batch_time)
        
        # Log periodically without cluttering output
        current_time = time.time()
        if (batch_idx == 0 or batch_idx % 100 == 99 or 
            current_time - self.last_log_time > 60):  # Log at most once per minute
            
            self.last_log_time = current_time
            print(f"  Batch {batch_idx+1:5d}, Loss: {loss.item():.2f}, MAE: {mae.item():.4f}")
            
        return loss

    def validation_step(self, batch, batch_idx):
        # Get loss and MAE 
        loss, mae = self.step(batch, batch_idx)
        
        # Log to Lightning (will go to loggers including WandB)
        self.log("validation/loss", loss.item(), prog_bar=False, on_step=True)
        self.log("validation/mae", mae.item(), prog_bar=False, on_step=True)
        
        # Store for epoch-end calculation
        self.val_loss_values.append(loss.item())
        self.val_mae_values.append(mae.item())
        
        return loss

    @torch.no_grad()
    def sample(self, shape, cond=None):
        """Sample using Heun's second order method.
        
        Args:
            shape: Shape of the output sample [B, 71, 4]
            cond: Dictionary with conditioning information
        
        Returns:
            Generated flux prediction
        """
        dtype = torch.float32 if self.device.type == "mps" else torch.float64

        # Generate sampling trajectory
        sigmas = self.edm.sampling_sigmas(self.num_sampling_steps, device=self.device)
        
        # Start with random noise
        eps = torch.randn(shape, device=self.device, dtype=dtype) * sigmas[0]
        
        # Sample using either deterministic or stochastic process
        if self.deterministic_sampling:
            sample = self.sample_deterministically(eps, sigmas, cond)
        else:
            sample = self.sample_stochastically(eps, sigmas, cond)

        return sample.to(torch.float32)

    def sample_deterministically(self, eps, sigmas, cond=None):
        """Deterministic sampling using Heun's 2nd order method."""
        dtype = torch.float32 if self.device.type == "mps" else torch.float64
        sample_next = eps
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            sample_curr = sample_next
            pred_curr = self(
                sample=sample_curr.to(self.dtype),
                sigma=sigma.to(self.dtype).repeat(len(sample_curr)),
                cond=cond
            ).to(dtype)
            d_cur = (sample_curr - pred_curr) / sigma
            sample_next = sample_curr + d_cur * (sigma_next - sigma)

            # second order correction
            if i < self.num_sampling_steps - 1:
                pred_next = self(
                    sample=sample_next.to(self.dtype),
                    sigma=sigma_next.to(self.dtype).repeat(len(sample_curr)),
                    cond=cond
                ).to(dtype)
                d_prime = (sample_next - pred_next) / sigma_next
                sample_next = sample_curr + (sigma_next - sigma) * (0.5 * d_cur + 0.5 * d_prime)

        return sample_next

    def sample_stochastically(self, eps, sigmas, cond=None):
        """Stochastic sampling with noise injection for better quality."""
        dtype = torch.float32 if self.device.type == "mps" else torch.float64
        sample_next = eps
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            sample_curr = sample_next

            # increase noise temporarily
            sigma_hat = self.edm.sigma_hat(sigma, self.num_sampling_steps)
            noise = torch.randn_like(sample_curr) * self.edm.S_noise
            sample_hat = sample_curr + noise * (sigma_hat**2 - sigma**2) ** 0.5

            # euler step
            pred_hat = self(
                sample=sample_hat.to(self.dtype),
                sigma=sigma_hat.to(self.dtype).repeat(len(sample_hat)),
                cond=cond
            ).to(dtype)
            d_cur = (sample_hat - pred_hat) / sigma_hat
            sample_next = sample_hat + d_cur * (sigma_next - sigma_hat)

            # second order correction
            if i < self.num_sampling_steps - 1:
                pred_next = self(
                    sample=sample_next.to(self.dtype),
                    sigma=sigma_next.to(self.dtype).repeat(len(sample_hat)),
                    cond=cond
                ).to(dtype)
                d_prime = (sample_next - pred_next) / sigma_next
                sample_next = sample_hat + (sigma_next - sigma_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return sample_next

    @torch.no_grad()
    def evaluate(self, batch):
        """Evaluate the model on a batch of data."""
        sample = batch["sample"]  # Flux values [B, 71, 4]
        cond = batch["cond"]      # Physical features dictionary
        
        return self.sample(shape=sample.shape, cond=cond)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_params["learning_rate"])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.optimizer_params["max_steps"]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }