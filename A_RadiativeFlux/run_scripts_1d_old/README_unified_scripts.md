# Unified Flux Training Scripts and Debug Configurations

This directory contains complete bash scripts and VS Code debug configurations for different training setups using the unified flux training system.

## Bash Scripts

### 1. `unified_train_1d_full.sh`
**Setup**: 1D column-wise training on full globe dataset
- **Mode**: `1d` (column-wise processing)
- **Dataset**: `full` (all 81,920 columns globally)
- **Model**: GNN with fully connected edges
- **Use case**: Standard 1D radiative flux prediction across the entire globe
- **Batch size**: 512 (good for full dataset)

### 2. `unified_train_3d_triangle.sh`
**Setup**: 3D triangle-wise training on regional dataset
- **Mode**: `3d` (triangle-wise spatial processing)
- **Dataset**: `triangle` (specific triangular region)
- **Model**: GNN_3D with spatial graph structure
- **Use case**: High-resolution 3D modeling with spatial correlations
- **Batch size**: 2 (memory intensive due to 3D processing)
- **Requires**: Grid file for spatial relationships

### 3. `unified_train_1d_triangle.sh`
**Setup**: 1D column-wise training on regional dataset
- **Mode**: `1d` (column-wise processing)
- **Dataset**: `triangle` (specific triangular region)
- **Model**: GNN with fully connected edges
- **Use case**: Regional 1D modeling with reduced computational cost
- **Batch size**: 1024 (smaller dataset allows larger batches)

### 4. `unified_train_1d_hr_loss.sh`
**Setup**: 1D training with heating rate smoothness regularization
- **Mode**: `1d` (column-wise processing)
- **Dataset**: `full` (all columns)
- **Model**: GNN with heating rate smoothness loss
- **Use case**: Physics-informed training with vertical heating rate constraints
- **Special features**:
  - `--hr-smoothness-weight 0.05`: Regularization strength
  - `--hr-smoothness-top-levels 30`: Apply smoothness to top 30 levels
  - Physics-based loss for more realistic heating rate profiles

## VS Code Debug Configurations

Located in `.vscode/` directory, these JSON files enable debugging with the same parameters:

### 1. `launch_unified_1d_full.json`
- **Debug configuration** for 1D full globe training
- **Reduced parameters** for faster debugging (1% data, 2 epochs, small batch)
- **W&B disabled** for local debugging

### 2. `launch_unified_3d_triangle.json`
- **Debug configuration** for 3D triangle training
- **Minimal parameters** (batch size 1, very small dataset)
- **All required 3D parameters** included

### 3. `launch_unified_1d_triangle.json`
- **Debug configuration** for 1D triangle training
- **Reduced scale** for quick iteration
- **Triangle-specific parameters** included

### 4. `launch_unified_1d_hr_loss.json`
- **Debug configuration** for heating rate loss training
- **HR loss parameters** enabled for testing
- **Quick debugging** setup

## Parameter Differences Summary

| Setup | Mode | Dataset | Model | Key Features |
|-------|------|---------|-------|---------------|
| 1D Full | `1d` | `full` | `gnn` | Global coverage, standard training |
| 3D Triangle | `3d` | `triangle` | `gnn_3d` | Spatial correlations, memory intensive |
| 1D Triangle | `1d` | `triangle` | `gnn` | Regional focus, efficient |
| 1D HR Loss | `1d` | `full` | `gnn` | Physics-informed, heating rate smoothness |

## Usage Instructions

### Running Production Training:
```bash
# Make executable
chmod +x unified_train_1d_full.sh

# Run training
./unified_train_1d_full.sh
```

### VS Code Debugging:
1. Open VS Code in the project directory
2. Open the debug panel (Ctrl+Shift+D)
3. Select desired configuration from dropdown
4. Press F5 to start debugging

## Key Configuration Notes

- **3D models require** `--grid-file-path` parameter
- **Triangle datasets require** `--triangle-id` and `--triangle-division-factor`
- **HR loss training uses** `--hr-smoothness-weight` and related parameters
- **Debug configs use** reduced data (`--percent 0.01`) and epochs (`--num-epoch 2`)
- **W&B integration** enabled for production (`--wandb-mode online`), disabled for debug

## Customization

To modify parameters:
1. **Edit bash scripts** for production runs
2. **Edit JSON configs** for debugging sessions
3. **Adjust batch sizes** based on GPU memory
4. **Modify data percentages** for faster experimentation 