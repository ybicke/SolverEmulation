# Longitude and Latitude Coordinate Features

This implementation adds support for including longitude and latitude coordinates as additional 2D features in the tendency prediction models.

## Overview

The longitude and latitude coordinates are extracted from the ICON grid file for the specific triangular area being used for training. These coordinates are normalized and concatenated to the 2D features, providing spatial context to the models.

## Key Features

1. **Automatic Coordinate Extraction**: Extracts `clon` and `clat` from the ICON grid file for the triangle area
2. **Normalization**: Standardizes coordinates using mean and standard deviation computed from the triangle area
3. **Seamless Integration**: Automatically adjusts 2D feature dimensions when enabled
4. **Mode Support**: Works with both 1D (column-wise) and 3D (triangle-wise) training modes

## Usage

### Basic Usage

Enable lon/lat features by adding the `--use-lonlat` flag and providing the grid file path:

```bash
# 1D model with lon/lat features
python train_tendency.py \
    --model vit \
    --mode 1d \
    --dataset-type triangle \
    --triangle-id 39 \
    --triangle-division-factor 4 \
    --use-lonlat \
    --grid-file-path /path/to/icon_grid_file.nc \
    --dataset-input /path/to/inputs \
    --dataset-output /path/to/outputs \
    --save /path/to/save/results

# 3D model with lon/lat features  
python train_tendency.py \
    --model gnn_3d \
    --mode 3d \
    --dataset-type triangle \
    --triangle-id 39 \
    --triangle-division-factor 4 \
    --use-lonlat \
    --grid-file-path /path/to/icon_grid_file.nc \
    --dataset-input /path/to/inputs \
    --dataset-output /path/to/outputs \
    --save /path/to/save/results
```

### Feature Dimensions

When `--use-lonlat` is enabled:
- Original 2D features: typically 3 channels (pres_sfc, cosmu0, qv_s)
- With lon/lat: 5 channels (original 3 + longitude + latitude)
- The script automatically adjusts the model's `channels_in_2D` parameter

### Requirements

1. **Grid File**: Must provide `--grid-file-path` when using `--use-lonlat`
2. **Triangle Mode**: Currently only supported for `--dataset-type triangle`
3. **Grid File Format**: Must contain `clon` and `clat` variables with longitude/latitude in radians

## Implementation Details

### Coordinate Processing

1. **Extraction**: Coordinates are extracted using `get_lonlat_coordinates_for_triangle()`
2. **Normalization**: Statistics computed using `create_lonlat_normalizer()`
3. **Integration**: Coordinates normalized and concatenated to 2D features in the data loader

### Data Flow

```
Grid File (clon, clat) → Extract Triangle Coords → Normalize → Concat to x2d → Model
```

### Normalization Strategy

- **Scope**: Normalization statistics computed over the specific triangle area
- **Method**: Standard z-score normalization: `(coord - mean) / std`
- **Output**: Each coordinate normalized independently, then stacked as 2 additional channels

## File Changes

The implementation required modifications to:

1. **`train_tendency.py`**: Added `--use-lonlat` argument and model parameter adjustments
2. **`tendency_data_loader_simple.py`**: Added coordinate extraction and concatenation logic
3. **`utils/data_utils.py`**: Added coordinate processing functions
4. **`utils/graph_3d.py`**: No changes needed (already handles neighbor extraction)

## Error Handling

- Validates that `--grid-file-path` is provided when `--use-lonlat` is enabled
- Checks for triangle mode compatibility
- Handles tensor device placement consistently
- Provides informative logging about coordinate initialization

## Performance Considerations

- Coordinates are extracted once during dataset initialization
- Minimal computational overhead during training
- Memory usage increases by 2 channels worth of data per sample

## Example Output

When enabled, you'll see logging output like:
```
Triangle 39 with division_factor 4: 1024 columns
Initialized lon/lat coordinates for triangle: shape torch.Size([1024, 2])
Lon/lat normalization - mean: [0.6283, 1.5614], std: [0.0123, 0.0089]
Using lon/lat features: 2D channels increased from 3 to 5
```

## Future Enhancements

Potential improvements:
1. Support for full dataset mode (all 81920 columns)
2. Alternative normalization strategies (global vs. local)
3. Additional coordinate transformations (spherical, Cartesian)
4. Dynamic coordinate selection based on model requirements 