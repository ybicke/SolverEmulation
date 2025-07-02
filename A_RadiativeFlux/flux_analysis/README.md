# Flux Analysis Package

This package contains tools for analyzing atmospheric flux prediction models, with a focus on understanding Vision Transformer attention patterns.

## Structure

```
flux_analysis/
├── __init__.py                 # Package initialization
├── attention_analysis.py       # Core attention analysis functionality
├── analyze_attention.py        # Standalone analysis script
└── README.md                   # This documentation
```

## Features

### Attention Analysis (`attention_analysis.py`)
- **AttentionCapture**: Hooks for capturing attention weights from transformer layers
- **AttentionAnalyzer**: Analysis and visualization of attention patterns
- **analyze_model_attention()**: Main analysis function
- **run_attention_analysis_if_enabled()**: Helper for training script integration

### Standalone Analysis Script (`analyze_attention.py`)
Complete script for post-hoc attention analysis of trained models.

## Usage

### During Training
The attention analysis is automatically integrated into the training pipeline:

```bash
python train_flux.py --model vit --attention-analysis --attention-samples 500
```

### Standalone Analysis
Run analysis on an already trained model:

```bash
cd SolverEmulation/A_RadiativeFlux
python -m flux_analysis.analyze_attention \
    --model-path checkpoints/best_model.pth \
    --dataset /path/to/dataset \
    --save-dir attention_results \
    --attention-samples 1000
```

### Programmatic Usage
```python
from flux_analysis.attention_analysis import analyze_model_attention

# Analyze a model
attention_matrices = analyze_model_attention(
    model=my_model,
    test_loader=test_loader,
    normalizer=normalizer,
    save_dir='./attention_plots',
    num_samples=500
)
```

## Output

The analysis generates:
- **High-quality attention heatmaps** (PNG files at 300 DPI)
- **Proper atmospheric labeling** (height levels with altitudes)
- **Publication-ready figures** with appropriate font sizes and spacing

## Attention Pattern Interpretation

### Understanding the Plots
- **Y-axis (Query)**: "What am I looking for?" - atmospheric level needing information
- **X-axis (Key)**: "What information do I offer?" - atmospheric level providing information  


## Requirements

- PyTorch with CUDA support
- matplotlib for visualization
- numpy for numerical operations
- Standard Python logging

## Future Extensions

This package is designed to be extensible for additional analysis tools:
- Layer-wise attention evolution analysis
- Attention head specialization studies
- Cross-attention pattern analysis
- Attention-based model interpretability tools 