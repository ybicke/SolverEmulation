# SolverEmulation

Machine learning emulation of atmospheric solvers for radiative flux and tendency prediction using state-of-the-art neural network architectures.

## Overview

This repository provides neural network implementations to emulate atmospheric physics solvers, specifically:
- **Radiative Flux Prediction** (`A_RadiativeFlux/`): Emulating radiative transfer calculations, which are calculated by the ecRad radiative transfer solver in ICON.
- **Tendency Prediction** (`B_Tendency/`): Emulating atmospheric state tendencies (rate of changes) which are calculated by the dynamical core solver in ICON. 

The framework supports both 1D (column-wise) and 3D (triangle-wise) modeling approaches with various neural architectures including Vision Transformers (ViT), Graph Neural Networks (GNN), AFNO, and RNNs.

## Features

- **Multiple Neural Architectures**: ViT, GNN, AFNO, RNN, U-Net, and more
- **1D & 3D Modeling**: Column-wise and triangle-wise approaches
- **Comprehensive Evaluation**: Built-in metrics and visualization tools
- **Experiment Tracking**: Integrated Weights & Biases (wandb) support
- **Physics-Informed Loss**: Heating rate smoothness constraints

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SolverEmulation.git
cd SolverEmulation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

## Data Setup

### Data Format Requirements

This project expects atmospheric data in HDF5 format with the following structure:
- **Input features**: Atmospheric state variables (temperature, pressure, humidity, water content, etc.)
- **Output targets**: 
  - For flux models: Radiative fluxes (LW/SW upward/downward)
  - For tendency models: Atmospheric state tendencies
- **Grid information**: For 3D models, requires grid topology files (NetCDF format)

### Preparing Your Data

1. **Organize your data** in a directory structure:
   ```
   /path/to/your/data/
   ├── h5_data/              # Training data in HDF5 format
   ├── grid_file.nc          # Grid topology (for 3D models)
   └── statistics/           # Normalization statistics (optional)
   ```

2. **Update paths in training scripts**:
   - Open the script you want to run (e.g., `A_RadiativeFlux/train_flux_large.py`)
   - Update the `--dataset` argument to point to your data directory
   - Update the `--save` argument to specify where to save results

3. **Update visualization scripts** (after training):
   - Open `visualize_results/generate_metrics_flux.py` or similar
   - Update the `models` list with paths to your trained model results
   - See the ⚠️ IMPORTANT comments in these files for guidance

### Example Data Paths

The example paths in this repository (e.g., `/mydata/deepcloud/yves/...`) are from the original development environment. You'll need to replace these with your own paths.

**Tip**: Use absolute paths for reliability, or relative paths from the repository root.

## Project Structure

```
SolverEmulation/
├── A_RadiativeFlux/          # Radiative flux prediction module
│   ├── models_1d/            # 1D models (ViT, GNN, AFNO, RNN, etc.)
│   ├── models_3d/            # 3D models (GNN-3D)
│   ├── utils/                # Utility functions for data processing
│   ├── flux_analysis/        # Attention analysis tools
│   ├── run_scripts/          # Training scripts and configurations
│   ├── flux_data_loader.py   # Data loading utilities
│   └── train_flux_large.py   # Main training script for flux
│
├── B_Tendency/               # Tendency prediction module
│   ├── models_1d/            # 1D models
│   ├── models_3d/            # 3D models (GNN-3D, Graph Transformer)
│   ├── utils/                # Utility functions
│   ├── run_scripts/          # Training scripts
│   ├── tendency_data_loader.py
│   └── train_tendency.py     # Main training script for tendency
│
├── visualize_results/        # Visualization and metrics generation
│   ├── visualize_error/      # Error visualization scripts
│   ├── visualize_differences/
│   ├── generate_metrics_flux.py
│   └── generate_metrics_tendency.py
│
├── tendency_data/            # Data statistics and processing
│   ├── statistics/
│   └── gnerate_statistics/
│
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Package configuration
└── README.md                # This file
```

## Usage

### Training Radiative Flux Models

**1D Models (column-wise):**
```bash
cd A_RadiativeFlux
python train_flux_large.py --model vit --mode 1d --dataset-type full
python train_flux_large.py --model gnn --mode 1d --dataset-type triangle --triangle-id 0
```

**3D Models (triangle-wise):**
```bash
python train_flux_large.py --model gnn_3d --mode 3d --dataset-type triangle --grid-file-path /path/to/grid.nc
```

**With heating rate smoothness loss:**
```bash
python train_flux_large.py --model vit --mode 1d --hr-smoothness-weight 0.1 --hr-smoothness-top-levels 10
```

### Training Tendency Models

**1D Models:**
```bash
cd B_Tendency
python train_tendency.py --model vit_tendency --mode 1d --dataset-type triangle
python train_tendency.py --model gnn --mode 1d --dataset-type full
```

**3D Models:**
```bash
python train_tendency.py --model gnn_3d_tendency --mode 3d --dataset-type triangle
python train_tendency.py --model graph_transformer --mode 3d --dataset-type triangle
```

### Generating Metrics and Visualizations

```bash
cd visualize_results
python generate_metrics_flux.py
python generate_metrics_tendency.py
```

## Available Models

### 1D Models
- **ViT** (Vision Transformer): Attention-based model for sequential data
- **AFNO** (Adaptive Fourier Neural Operator): Frequency domain model
- **GNN** (Graph Neural Network): Graph-based atmospheric column model
- **RNN** (BiLSTM): Recurrent neural network
- **U-Net**: Encoder-decoder architecture
- **U-ViT**: Vision Transformer with U-Net structure

### 3D Models
- **GNN-3D**: 3D graph neural network for triangle-wise processing
- **Graph Transformer**: Transformer architecture on graph structures
- **GraphCast-style**: Inspired by Google's GraphCast

## Experiment Tracking

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking. To use it:

1. Create a free account at [wandb.ai](https://wandb.ai/)
2. Login to wandb:
```bash
wandb login
```
3. Your experiments will be automatically logged during training

## Data Format

The models expect data in HDF5 format with the following structure:
- Input features: atmospheric state variables (temperature, pressure, humidity, etc.)
- Output targets: radiative fluxes or tendencies
- Metadata: grid information, normalization statistics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{solver_emulation,
  author = {Yves Bicker},
  title = {SolverEmulation: Neural Network Emulation of Atmospheric Solvers},
  year = {2025},
  url = {https://github.com/ybicke/SolverEmulation}
}
```

## Contact

For questions or feedback, please open an issue on GitHub or contact [yves_bicker@icloud.com](mailto:yves_bicker@icloud.com).

## Acknowledgments

This project builds upon and adapts code from several sources:

### Code Adaptations:
- **AFNO models**: Adapted from NVIDIA's [AFNO-pytorch](https://github.com/NVlabs/AFNO-pytorch) implementation
- **Vision Transformer**: Adapted from [vit-pytorch](https://github.com/lucidrains/vit-pytorch) by Phil Wang
- **RNN models**: Implementation by Salman Mohebi (s.mohebi22@gmail.com)

### Architectural Inspiration:
- **Graph Neural Networks**: Inspired by Google DeepMind's [GraphCast](https://arxiv.org/abs/2212.12794) and [GenCast](https://arxiv.org/abs/2312.15796), adapted for atmospheric solver emulation
- Graph Transformer architectures adapted for atmospheric column and triangle-wise processing
