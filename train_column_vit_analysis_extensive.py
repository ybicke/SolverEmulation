import os
import re
import sys
import time
import glob
import h5py
import yaml
import fsspec
import pickle
import shutil
import logging
import tempfile
import random

import argparse
from concurrent.futures import ThreadPoolExecutor
from os.path import join, dirname, basename, normpath, isfile, exists

import wandb
import torch
# import lightning as L
# import lightning as L
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchinfo import summary

import torch.autograd.profiler as profiler


from data_loaders_new import IconColumnIterableDataset
# from flux_specific_sigmoid.FluxSpecificSigmoid_lwdown import load_gaussian_parameters, construct_gaussian_params_by_height



sys.path.append(dirname(__file__))

PROFILE_NAME = 'default'
ENDPOING_URL = 'https://os.zhdk.cloud.switch.ch'  # SWITCH S3 server
BUCKET = 'deepcloud'
CACHE_DIR = '/tmp' # tempfile.TemporaryDirectory(dir='/tmp').name


seed = 42
wandb_config = {'seed': seed}
prng = np.random.RandomState(seed)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import argparse
import os

parser = argparse.ArgumentParser(description='Train Transformer models.')
parser.add_argument('--model', type=str, default='vit', help='Name of the model to be trained')
parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
parser.add_argument('--save', type=str, required=True, help='Path to save the result')
parser.add_argument('--percent', type=float, default=1.0, help='Percent of data to train on')
parser.add_argument('--subsample', type=float, default=None, help='Subsampling rate')
parser.add_argument('--num-workers', type=int, default=os.cpu_count(), help='Number of workers to load data')
parser.add_argument('--prefetch-factor', type=int, default=2, help='Prefetch factor')
parser.add_argument('--wandb-mode', type=str, default='disabled', choices={'online', 'offline', 'disabled'}, help='Operating mode for W&B')
parser.add_argument('--num-cells', type=int, default=81920, help='Number of ICON cells')
parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True, help='Specify if training takes place')
parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=True, help='Specify if test takes place')
parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=True, help='Shuffling the train dataset')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
parser.add_argument('--vbatch', type=int, default=1, help='Virtual batch: gradients will be applied after vbatch epoch')
parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping')
parser.add_argument('--num-epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--patch-size', type=int, default=2, help='Patch size')
parser.add_argument('--vit-hidden-dim', type=int, default=256, help='Vit hidden dimension')
parser.add_argument('--vit-layers', type=int, default=4, help='Vit layers')
parser.add_argument('--vit-heads', type=int, default=6, help='Vit heads')
parser.add_argument('--vit-dropout', type=float, default=0.0, help='Vit dropout')
parser.add_argument('--afno-sparsity-threshold', type=float, default=0.01, help='Sparsity threshold for AFNO')
parser.add_argument('--hard-thresholding-fraction', type=float, default=1, help='hard thresholding fraction AFNO')
parser.add_argument('--cutoff-frequency', type=float, default=0.1, help='cutoff frequency low pass filtering in AFNO')
parser.add_argument('--zero-freq-indices', nargs='+', type=int, default=None, help='Zero frequency indices to zero out')
parser.add_argument('--gaussian_params_file_LWDown', type=str, help='Path to the .npz file containing Gaussian parameters')
parser.add_argument('--gaussian_params_file_LWUp', type=str, help='Path to the .npz file containing Gaussian parameters')
args = parser.parse_args()


save_id = f'{basename(normpath(args.save))}'
checkpoint_path = join(args.save)
os.makedirs(checkpoint_path, exist_ok=True)
test_path = join(args.save, 'test/')
os.makedirs(test_path, exist_ok=True)

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Torch is using: {device}, number of cpus: {os.cpu_count()}')

with open(join(args.save, 'config.yml'), 'w') as f:
    yaml.dump(args.__dict__, f, default_flow_style=False)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def get_normalization_params(stats_file):
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)
        return torch.tensor(stats['mean2d']).to(device), \
            torch.tensor(stats['var2d']).to(device), \
                torch.tensor(stats['mean3d']).to(device), \
                    torch.tensor(stats['var3d']).to(device)
    
def get_model(model_name, mean2d, var2d, mean3d, var3d, is_test):
    logger.info('Preparing the model...')
    if model_name == 'vit':
        from column_files.vit_column import ViT
        model = ViT(
            num_cells=args.num_cells,
            patch_size=args.patch_size,
            dim=args.vit_hidden_dim,
            mlp_dim=args.vit_hidden_dim,
            depth=args.vit_layers,
            heads=args.vit_heads,
            dropout=args.vit_dropout,
            mean2d=mean2d,
            var2d=var2d, 
            mean3d=mean3d, 
            var3d=var3d,
            device=device
        ).to(device)
        
        
    elif model_name == 'vit_column4':
        from column_files.vit_column import ViT4
        model = ViT4(
            num_cells=args.num_cells,
            patch_size=args.patch_size,
            dim=args.vit_hidden_dim,
            mlp_dim=args.vit_hidden_dim,
            depth=args.vit_layers,
            heads=args.vit_heads,
            dropout=args.vit_dropout,
            mean2d=mean2d,
            var2d=var2d, 
            mean3d=mean3d, 
            var3d=var3d,
            device=device
        ).to(device)
        
        
    elif model_name == 'vit_analysis':
        from column_files.vit_column_analysis import ViT4
        model = ViT4(
            num_cells=args.num_cells,
            patch_size=args.patch_size,
            dim=args.vit_hidden_dim,
            mlp_dim=args.vit_hidden_dim,
            depth=args.vit_layers,
            heads=args.vit_heads,
            dropout=args.vit_dropout,
            mean2d=mean2d,
            var2d=var2d, 
            mean3d=mean3d, 
            var3d=var3d,
            device=device
        ).to(device) 
 
 
    elif model_name == 'vit_analysis2':
        from column_files.vit_column_analysis2 import ViT
        model = ViT(
            num_cells=args.num_cells,
            patch_size=args.patch_size,
            dim=args.vit_hidden_dim,
            mlp_dim=args.vit_hidden_dim,
            depth=args.vit_layers,
            heads=args.vit_heads,
            dropout=args.vit_dropout,
            mean2d=mean2d,
            var2d=var2d, 
            mean3d=mean3d, 
            var3d=var3d,
            device=device
        ).to(device)
 
 
    else:
        raise NotImplementedError('Model has not implemented yet!')
    return model
        
    
def find_latest_checkpoint(directory):
    ckpts = glob.glob(join(directory, f'checkpoint_epoch_*.pth'))
    if len(ckpts) == 0:
        return None, None
    idx = [int(re.search( 'checkpoint_epoch_(.*?).pth', cp).group(1)) for cp in ckpts]
    return join(directory, f'checkpoint_epoch_{max(idx)}.pth'), max(idx)


def train_model(model, train_set, valid_set):
    
    # Log the start of training
    logger.info('Train started...')
    
    # Initialize Weights & Biases 
    wandb.init(
        project='deepcloud-yves', 
        name=save_id, 
        id=save_id, 
        config={**wandb_config, **args.__dict__}, 
        sync_tensorboard=True, 
        save_code=True,
        resume='allow', 
        tags=['icon grid',],    
        mode=args.wandb_mode
    )
    wandb.watch(model, log_freq=100)

    # Set up the optimizer based on the specified type
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate,
            eps=1e-8,
            weight_decay=0.01  # basically applying ridge regression L2
        )
    else:
        raise NameError('optimizer not supported.')
    
    # Initialize loss and metric trackers
    train_loss = MeanSquaredError().to(device)
    valid_loss = MeanSquaredError().to(device)
    train_mae = MeanAbsoluteError().to(device)
    valid_mae = MeanAbsoluteError().to(device)
    
    # Load the latest checkpoint if available
    cp_path = None
    p_path, cp_id = find_latest_checkpoint(checkpoint_path)
    if p_path is not None:
        cp_path = p_path
        logger.info(f'Loading checkpoint: {cp_id}, {cp_path}')
        checkpoint = torch.load(cp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = cp_id
        # vbatch = min(args.vbatch + (init_epoch/2), 20)
        logger.info(f'Training will continue from epoch: {init_epoch}/{args.num_epoch}')
    else:
        init_epoch = 0
        # vbatch = args.vbatch

    vbatch = args.vbatch
    epoch_number = init_epoch
    best_loss = 1e9999999 
      

    # Training loop
    for epoch in range(init_epoch, args.num_epoch):
        t1 = time.perf_counter()
        epoch_number += 1
        
        # Training step
        model.train(True)        
        for i, data in enumerate(train_set):
            
            t1_1 = time.perf_counter()
            
            batch_x3, batch_x2, batch_y = data
            batch_x3, batch_x2, batch_y = batch_x3.to(device), batch_x2.to(device), batch_y.to(device)

            outputs = model(batch_x3, batch_x2)
            loss = train_loss(outputs, batch_y)
            batch_mae = train_mae(outputs, batch_y)
            
            if i > 0 and i % vbatch == 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                t2_1 = time.perf_counter()
                
                if i % 100 == 99 or vbatch > 1:
                    print(f'batch {i+1}, time:{t2_1-t1_1:.3f}, loss: {loss:.4f}, mean_absolute_error: {batch_mae:.4f}')
                

        # Validation step
        model.eval()
        
        with torch.no_grad():
            for i, v_data in enumerate(valid_set):
                
                vx3d, vx2d, v_labels = v_data
                vx3d, vx2d, v_labels = vx3d.to(device), vx2d.to(device), v_labels.to(device)
                v_outputs = model(vx3d, vx2d)        
                
                # two different losses         
                valid_loss.update(v_outputs, v_labels)
                valid_mae.update(v_outputs, v_labels)                

        # Compute total losses and metrics
        total_train_loss = train_loss.compute()
        total_valid_loss = valid_loss.compute()
        total_train_mae = train_mae.compute()
        total_valid_mae = valid_mae.compute()

        t2 = time.perf_counter()

        # Log metrics to W&B
        wandb.log({
            'epoch': epoch_number, 
            'loss': total_train_loss,
            'val_loss': total_valid_loss,
            'mean_absolute_error': total_train_mae,
            'val_mean_absolute_error': total_valid_mae
            })

        # Print epoch summary
        print(f'{epoch_number:03}/{args.num_epoch}: ',
              f'time: {t2-t1:.3f}', 
              f'loss: {total_train_loss:.4f} ',
              f'mean_absolute_error: {total_train_mae:.4f}, ',
              f'val_loss: {total_valid_loss:.4f}, ',
              f'val_mean_absolute_error: {total_valid_mae:.4f}')
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_loss,
        }
        torch.save(
            checkpoint, 
            join(checkpoint_path, f'checkpoint_epoch_{epoch_number}.pth')
        )
        if total_valid_loss < best_loss:
            torch.save(checkpoint, join(checkpoint_path, 'best_model.pth'))
            best_loss = total_valid_loss
            
        # Reset metrics for the next epoch
        train_mae.reset()
        valid_mae.reset()
        train_loss.reset()
        valid_loss.reset()
    return model


def calculate_heating_rates(y, x3d, x2d): 
    # assumed output order is: [lw_up, lw_dn, sw_up, sw_dn]
    g = 9.80665
    cd = 1005
    cv = 1855
    qv_ecrad_in_idx = 5
    pres_ecrad_in_idx = 2
    pres_sfc_ecrad_in_idx = 0

    qv = x3d[..., qv_ecrad_in_idx]
    pres = torch.zeros([x3d.shape[0], 71], device=y.device)

    pres[..., 1:] = x3d[..., pres_ecrad_in_idx]
    pres[..., 0] = x2d[..., pres_sfc_ecrad_in_idx]
    pres[...,1:-1] = torch.sqrt(pres[...,1:-1] * pres[...,2:])
    top_interp = pres[...,-1] / torch.sqrt(pres[...,-2]*pres[...,-1])
    # pressure values are very small at the toa
    pres[...,1:-1] = torch.sqrt(pres[...,0:69] * pres[...,1:-1])
    pres[...,-1] = top_interp
    pres = torch.unsqueeze(pres, dim=-1)
    qv = torch.unsqueeze(qv, dim=-1)
    heating_rate = ((-g / (cd * (1-qv) + cv * qv)) / \
    (pres[..., :-1, :] - pres[...,1:,:])) * \
    ((y[..., :-1, [0, 2]] - y[..., :-1, [1, 3]]) - \
    (y[..., 1:, [0, 2]] - y[..., 1:, [1, 3]])) * 24*60*60
    return heating_rate


def test_model(model, test_set):
    logger.info('Test started...')    
 
    # Load the best model checkpoint
    best_chkpt = join(checkpoint_path, 'best_model.pth')
    best_chkpt = join(checkpoint_path, 'best_model.pth')
    assert isfile(best_chkpt), 'Checkpoint not found, testing faild!'
    checkpoint = torch.load(best_chkpt, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize loss and metric trackers
    test_loss = MeanSquaredError().to(device)
    test_mae = MeanAbsoluteError().to(device)

    y_true, y_pred = list(), list()
    h_true, h_pred = list(), list()
    
    t1 = time.perf_counter(), time.process_time()
    
    # Enhanced attention analysis setup
    num_layers = args.vit_layers
    num_heads = args.vit_heads
    
    # Store attention data for analysis
    all_attention_matrices = {layer_idx: [] for layer_idx in range(num_layers)}
    head_importance_scores = {layer_idx: [] for layer_idx in range(num_layers)}
    
    # Analysis settings - analyze fewer batches for efficiency
    max_analysis_batches = 5  # Analyze first 5 batches for attention patterns
    
    logger.info(f"Starting enhanced attention analysis for {max_analysis_batches} batches...")

    for i, data in enumerate(test_set):
        batch_x3, batch_x2, batch_y = data
        batch_x3, batch_x2, batch_y = batch_x3.to(device), batch_x2.to(device), batch_y.to(device)
        model.eval()
        
        with torch.no_grad():
            # Forward pass for regular prediction
            outputs = model(batch_x3, batch_x2)

            # Enhanced attention analysis (only for first few batches)
            if i < max_analysis_batches:
                logger.info(f"Analyzing attention patterns for batch {i+1}/{max_analysis_batches}...")
                
                for layer_idx in range(num_layers):
                    # Get attention weights for this layer
                    attention_weights = model.get_attention_weights(batch_x3, batch_x2, layer_idx=layer_idx)
                    # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
                    
                    # 1. Store average attention across batch and heads for this layer
                    layer_avg_attention = attention_weights.mean(dim=[0, 1])  # Average over batch and heads
                    all_attention_matrices[layer_idx].append(layer_avg_attention.cpu())
                    
                    # 2. Calculate head importance scores (entropy-based)
                    batch_head_importance = []
                    for head_idx in range(num_heads):
                        head_attention = attention_weights[:, head_idx, :, :]  # [batch_size, seq_len, seq_len]
                        # Calculate entropy for each head (measure of attention dispersion)
                        epsilon = 1e-8
                        head_attention_prob = head_attention + epsilon
                        head_entropy = -torch.sum(head_attention_prob * torch.log(head_attention_prob), dim=[1, 2])
                        avg_head_entropy = head_entropy.mean()  # Average across batch
                        batch_head_importance.append(avg_head_entropy.cpu().item())
                    
                    head_importance_scores[layer_idx].append(batch_head_importance)
                    
                    # 3. For the first batch, generate individual head visualizations for selected layers
                    if i == 0:  # Only first batch to avoid too many plots
                        selected_layers_for_heads = [0, num_layers//2, num_layers-1]  # First, middle, last layer
                        
                        if layer_idx in selected_layers_for_heads:
                            logger.info(f"Creating individual head visualizations for layer {layer_idx}...")
                            
                            # Analyze head similarity/clustering
                            head_similarities = calculate_head_similarity(attention_weights)
                            logger.info(f"Layer {layer_idx} head similarities: {head_similarities}")
                            
                            # Create head analysis plots
                            create_head_analysis_plots(
                                attention_weights, 
                                layer_idx, 
                                head_importance_scores[layer_idx][-1], 
                                head_similarities,
                                save_path=test_path
                            )
                            
                            # Individual head attention matrices (only most and least important heads)
                            head_importances = batch_head_importance
                            most_important_head = np.argmax(head_importances)
                            least_important_head = np.argmin(head_importances)
                            
                            for head_idx, head_name in [(most_important_head, "most_important"), 
                                                      (least_important_head, "least_important")]:
                                single_head_attn = attention_weights[0, head_idx, :, :].cpu()  # First batch, specific head
                                model.visualize_attention(
                                    single_head_attn,
                                    layer_idx=f"{layer_idx}_head{head_idx}_{head_name}",
                                    save_path=test_path
                                )
                
                # After analyzing the desired number of batches, break
                if i == max_analysis_batches - 1:
                    logger.info("Finished collecting attention data. Creating summary visualizations...")
                    break
        
        # Continue with regular testing workflow
        y_true.append(batch_y.detach().cpu())
        y_pred.append(outputs.detach().cpu())
        h_true.append(calculate_heating_rates(batch_y, batch_x3, batch_x2).detach().cpu())
        h_pred.append(calculate_heating_rates(outputs, batch_x3, batch_x2).detach().cpu())

        # Calculate and log loss and mean absolute error
        loss = test_loss(outputs, batch_y)
        mae = test_mae(outputs, batch_y)
        
        if i % 1000 == 999:
            print(f'batch {i+1} loss: {loss:.4f}, '
                    f'mean_absolute_error: {mae:.4f},')

    # Generate comprehensive attention analysis
    logger.info("Creating comprehensive attention analysis...")
    create_comprehensive_attention_analysis(
        all_attention_matrices, 
        head_importance_scores, 
        test_path
    )
    
    # Generate layer-wise averaged attention matrices
    for layer_idx in range(num_layers):
        if all_attention_matrices[layer_idx]:
            # Average attention matrix across all analyzed batches
            final_avg_attention = torch.stack(all_attention_matrices[layer_idx]).mean(dim=0)
            
            logger.info(f"Creating visualization for layer {layer_idx} (averaged across {len(all_attention_matrices[layer_idx])} batches)...")
            model.visualize_attention(
                final_avg_attention,
                layer_idx=layer_idx,
                save_path=test_path
            )

    t2 = time.perf_counter(), time.process_time()
    
    # Concatenate all collected true and predicted values
    y_true = torch.cat(y_true, 0)
    y_pred = torch.cat(y_pred, 0)
    h_true = torch.cat(h_true, 0)
    h_pred = torch.cat(h_pred, 0)

    # Compute total test loss and mean absolute error
    total_test_loss = test_loss.compute()
    total_test_mae = test_mae.compute()

    print(f'Test time: {t2[0] - t1[0]:.2f} loss: {total_test_loss:.4f} ',
            f'mean_absolute_error: {total_test_mae:.4f}')

    # Save true and predicted values to files for further analysis
    with open(join(test_path, 'y_true.pickle'), 'wb') as handle:
        pickle.dump(y_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(join(test_path, 'y_pred.pickle'), 'wb') as handle:
        pickle.dump(y_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(join(test_path, 'h_true.pickle'), 'wb') as handle:
        pickle.dump(h_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(join(test_path, 'h_pred.pickle'), 'wb') as handle:
        pickle.dump(h_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info('Attention analysis completed.')


def calculate_head_similarity(attention_weights):
    """Calculate pairwise similarity between attention heads"""
    # attention_weights: [batch_size, num_heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    # Flatten attention matrices for similarity calculation
    flat_attention = attention_weights.view(batch_size, num_heads, -1)  # [batch_size, num_heads, seq_len^2]
    
    # Average across batch
    avg_flat_attention = flat_attention.mean(dim=0)  # [num_heads, seq_len^2]
    
    # Calculate cosine similarity between heads
    similarities = torch.zeros(num_heads, num_heads)
    for i in range(num_heads):
        for j in range(num_heads):
            if i != j:
                cos_sim = torch.nn.functional.cosine_similarity(
                    avg_flat_attention[i:i+1], 
                    avg_flat_attention[j:j+1], 
                    dim=1
                )
                similarities[i, j] = cos_sim.item()
    
    return similarities.cpu().numpy()


def create_head_analysis_plots(attention_weights, layer_idx, head_importances, head_similarities, save_path):
    """Create comprehensive head analysis plots"""
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Head importance scores
    ax1 = axes[0, 0]
    head_indices = list(range(num_heads))
    ax1.bar(head_indices, head_importances, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Head Index')
    ax1.set_ylabel('Attention Entropy (Importance)')
    ax1.set_title(f'Layer {layer_idx}: Head Importance Scores\n(Higher = more dispersed attention)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Head similarity heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(head_similarities, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Head Index')
    ax2.set_ylabel('Head Index')
    ax2.set_title(f'Layer {layer_idx}: Head Similarity Matrix\n(Cosine similarity)')
    plt.colorbar(im2, ax=ax2)
    
    # Add text annotations for similarity values
    for i in range(num_heads):
        for j in range(num_heads):
            if i != j:
                text = ax2.text(j, i, f'{head_similarities[i, j]:.2f}', 
                              ha="center", va="center", color="white", fontsize=8)
    
    # 3. Average attention pattern across all heads
    ax3 = axes[1, 0]
    avg_attention = attention_weights.mean(dim=[0, 1]).cpu()  # Average over batch and heads
    im3 = ax3.imshow(avg_attention, cmap='plasma', aspect='auto')
    ax3.set_xlabel('Height Level (TO)')
    ax3.set_ylabel('Height Level (FROM)')
    ax3.set_title(f'Layer {layer_idx}: Average Attention Pattern\n(All heads combined)')
    plt.colorbar(im3, ax=ax3)
    
    # 4. Attention diversity (standard deviation across heads)
    ax4 = axes[1, 1]
    attention_std = attention_weights.std(dim=1).mean(dim=0).cpu()  # Std across heads, avg across batch
    im4 = ax4.imshow(attention_std, cmap='coolwarm', aspect='auto')
    ax4.set_xlabel('Height Level (TO)')
    ax4.set_ylabel('Height Level (FROM)')
    ax4.set_title(f'Layer {layer_idx}: Attention Diversity\n(Std deviation across heads)')
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/layer_{layer_idx}_head_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_comprehensive_attention_analysis(all_attention_matrices, head_importance_scores, save_path):
    """Create comprehensive analysis across all layers"""
    num_layers = len(all_attention_matrices)
    
    # 1. Layer-wise attention evolution
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Calculate various metrics across layers
    layer_metrics = {
        'self_attention_strength': [],
        'cross_attention_strength': [],
        'attention_entropy': [],
        'max_attention_strength': []
    }
    
    for layer_idx in range(num_layers):
        if all_attention_matrices[layer_idx]:
            avg_attention = torch.stack(all_attention_matrices[layer_idx]).mean(dim=0)
            
            # Self-attention strength (diagonal)
            self_attn = torch.diagonal(avg_attention).mean().item()
            layer_metrics['self_attention_strength'].append(self_attn)
            
            # Cross-attention strength (off-diagonal)
            total_attn = avg_attention.sum().item()
            diag_attn = torch.diagonal(avg_attention).sum().item()
            cross_attn = (total_attn - diag_attn) / (avg_attention.numel() - avg_attention.shape[0])
            layer_metrics['cross_attention_strength'].append(cross_attn)
            
            # Attention entropy
            epsilon = 1e-8
            attn_prob = avg_attention + epsilon
            entropy = -torch.sum(attn_prob * torch.log(attn_prob)).item()
            layer_metrics['attention_entropy'].append(entropy)
            
            # Max attention strength
            max_attn = avg_attention.max().item()
            layer_metrics['max_attention_strength'].append(max_attn)
    
    # Plot layer evolution
    layer_indices = list(range(len(layer_metrics['self_attention_strength'])))
    
    axes[0, 0].plot(layer_indices, layer_metrics['self_attention_strength'], 'o-', label='Self-attention')
    axes[0, 0].plot(layer_indices, layer_metrics['cross_attention_strength'], 's-', label='Cross-attention')
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Attention Strength')
    axes[0, 0].set_title('Attention Strength Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(layer_indices, layer_metrics['attention_entropy'], 'd-', color='green')
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('Total Attention Entropy')
    axes[0, 1].set_title('Attention Entropy Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(layer_indices, layer_metrics['max_attention_strength'], '^-', color='red')
    axes[0, 2].set_xlabel('Layer Index')
    axes[0, 2].set_ylabel('Max Attention Weight')
    axes[0, 2].set_title('Max Attention Evolution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Head importance analysis
    if head_importance_scores:
        avg_head_importance = {}
        for layer_idx in range(num_layers):
            if head_importance_scores[layer_idx]:
                # Average importance across batches
                layer_head_importance = np.array(head_importance_scores[layer_idx]).mean(axis=0)
                avg_head_importance[layer_idx] = layer_head_importance
        
        # Plot head importance heatmap
        if avg_head_importance:
            importance_matrix = np.array([avg_head_importance[i] for i in sorted(avg_head_importance.keys())])
            
            im = axes[1, 0].imshow(importance_matrix, cmap='viridis', aspect='auto')
            axes[1, 0].set_xlabel('Head Index')
            axes[1, 0].set_ylabel('Layer Index')
            axes[1, 0].set_title('Head Importance Across Layers')
            plt.colorbar(im, ax=axes[1, 0])
            
            # Most/least important heads per layer
            most_important_heads = [np.argmax(importance_matrix[i]) for i in range(importance_matrix.shape[0])]
            least_important_heads = [np.argmin(importance_matrix[i]) for i in range(importance_matrix.shape[0])]
            
            axes[1, 1].plot(layer_indices, most_important_heads, 'o-', label='Most important', color='red')
            axes[1, 1].plot(layer_indices, least_important_heads, 's-', label='Least important', color='blue')
            axes[1, 1].set_xlabel('Layer Index')
            axes[1, 1].set_ylabel('Head Index')
            axes[1, 1].set_title('Most/Least Important Heads')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Head importance variance
            head_variance = np.var(importance_matrix, axis=1)
            axes[1, 2].plot(layer_indices, head_variance, 'D-', color='purple')
            axes[1, 2].set_xlabel('Layer Index')
            axes[1, 2].set_ylabel('Head Importance Variance')
            axes[1, 2].set_title('Head Specialization by Layer\n(Higher = more head diversity)')
            axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/comprehensive_attention_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    analysis_results = {
        'layer_metrics': layer_metrics,
        'head_importance_scores': head_importance_scores
    }
    
    with open(join(save_path, 'attention_analysis_results.pickle'), 'wb') as f:
        pickle.dump(analysis_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Comprehensive attention analysis saved to {save_path}")
    
    # Print summary
    print(f"\n=== COMPREHENSIVE ATTENTION ANALYSIS SUMMARY ===")
    print(f"Analyzed {num_layers} layers with attention patterns")
    if layer_metrics['self_attention_strength']:
        print(f"Self-attention strength: {np.mean(layer_metrics['self_attention_strength']):.4f} ± {np.std(layer_metrics['self_attention_strength']):.4f}")
        print(f"Cross-attention strength: {np.mean(layer_metrics['cross_attention_strength']):.4f} ± {np.std(layer_metrics['cross_attention_strength']):.4f}")
        print(f"Strongest self-attention in layer: {np.argmax(layer_metrics['self_attention_strength'])}")
        print(f"Strongest cross-attention in layer: {np.argmax(layer_metrics['cross_attention_strength'])}")

def test_loading_time(train_files, iter=10):
    logger.info('Test Loading time started...')
    dataset = get_column_data_with_disk_cache(train_files, shuffle=True)
        
    for it in range(iter):
        t1 = time.perf_counter(), time.process_time()
        for i, data in enumerate(dataset):
            batch_x3, batch_x2, batch_y = data
            batch_x3, batch_x2, batch_y = batch_x3.to(device), batch_x2.to(device), batch_y.to(device)
        t2 = time.perf_counter(), time.process_time()
        logger.info(f'Iteration: {it}: Real time: {t2[0] - t1[0]:.2f}, CPU time: {t2[1]-t1[1]}')
        import gc
        gc.collect()
        
        
def get_column_data_with_disk_cache(filenames, subsample=args.subsample, shuffle=False, num_workers=0):
    icon_data = IconColumnIterableDataset(filenames, subsample=subsample, cache_dir='/tmp', shuffle=shuffle)

    # Prepare arguments for DataLoader
    dataloader_args = {
        'dataset': icon_data,
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': num_workers,
    }

    # Include prefetch_factor only if num_workers > 0
    if num_workers > 0:
        dataloader_args['prefetch_factor'] = args.prefetch_factor

    return DataLoader(**dataloader_args)



def main():
    logger.info('Code started...')
    
    FILENAMES = glob.glob(join(args.dataset, '*.h5'))
    time_indices = [float(re.search( r'\_time_(.*?)\.h5', f).group(1)) for f in FILENAMES]
    sorted_files = [x for _,x in sorted(zip(time_indices, FILENAMES))]
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Save RNG states before model initialization
    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    random_rng_state = random.getstate()

    train_files = sorted_files[200:2000]
    val_files = sorted_files[:160] + sorted_files[2020:2180]
    test_files = sorted_files[2220:]
    
    # Sample dataset
    if args.percent < 1:
        train_files = prng.choice(train_files, max(1, int(args.percent*len(train_files))))
        val_files = prng.choice(val_files, max(1, int(args.percent*len(val_files))))
        test_files = prng.choice(test_files, max(1, int(args.percent*len(test_files))))

    stats_file = join(args.dataset, 'normalizer_stats_per_feat.pickle')
    mean2d, var2d, mean3d, var3d = get_normalization_params(stats_file)
    model = get_model(args.model, mean2d, var2d, mean3d, var3d, args.test)
    num_params = count_parameters(model)
    print(f"The model has {num_params:,} trainable parameters.")
    
    # Restore RNG states after model initialization
    torch.set_rng_state(torch_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(random_rng_state)

    if args.train:
        tr1 = time.perf_counter(), time.process_time()                        

        train_loader = get_column_data_with_disk_cache(train_files, shuffle=True)
        val_loader = get_column_data_with_disk_cache(val_files, shuffle = False, subsample=1.0)
   
        train_model(model, train_loader, val_loader)

        tr2 = time.perf_counter(), time.process_time()
        print(f'Training time: Real time: {tr2[0] - tr1[0]:.2f}, CPU time: {tr2[1]-tr1[1]}')
    
    if args.test:
        test_loader = get_column_data_with_disk_cache(test_files, shuffle=False)
        test_model(model, test_loader)
    
    logger.info('Code ended!')

if __name__ == '__main__':
    main()
    