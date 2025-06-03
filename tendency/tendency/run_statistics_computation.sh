#!/bin/bash

# Script to compute training set statistics for tendency prediction
# This uses the same data loader as training to ensure consistency

set -e

# Source setup script if needed
source /mydata/deepcloud/yves/SolverEmulation/A_RadiativeFlux/run_scripts/setup.sh

# Change to the script directory
cd /mydata/deepcloud/yves/SolverEmulation/tendency/tendency

echo "=== Training Set Statistics Computation ==="
echo "Starting computation at: $(date)"

# Check if directories exist
echo "Checking input directory..."
if [ ! -d "/mydata/deepcloud/yves/h5_tendency_data_all/inputs" ]; then
    echo "ERROR: Input directory does not exist"
    exit 1
fi

echo "Checking output directory..."
if [ ! -d "/mydata/deepcloud/yves/h5_tendency_data_all/outputs" ]; then
    echo "ERROR: Output directory does not exist"
    exit 1
fi

# Count files
input_file_count=$(find /mydata/deepcloud/yves/h5_tendency_data_all/inputs -name "*_inputs_*.h5" | wc -l)
output_file_count=$(find /mydata/deepcloud/yves/h5_tendency_data_all/outputs -name "*_tendencies_*.h5" | wc -l)
echo "Found $input_file_count input files and $output_file_count output files"

if [ $input_file_count -eq 0 ] || [ $output_file_count -eq 0 ]; then
    echo "ERROR: No input or output files found"
    exit 1
fi

echo "Running statistics computation..."

# Run the dedicated statistics computation script
python compute_training_statistics.py \
    --dataset_input /mydata/deepcloud/yves/h5_tendency_data_all/inputs \
    --dataset_output /mydata/deepcloud/yves/h5_tendency_data_all/outputs \
    --save_path /mydata/deepcloud/yves/h5_tendency_data_all/train_target_statistics.pickle \
    --percent 1.0 \
    --subsample 1.0 \
    --batch-size 128 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --compute-variance

echo "Statistics computation completed at: $(date)"

# Check if output file was created
if [ -f "/mydata/deepcloud/yves/h5_tendency_all/train_target_statistics.pickle" ]; then
    echo "SUCCESS: Statistics file created"
    ls -lh /mydata/deepcloud/yves/h5_tendency_all/train_target_statistics.pickle
else
    echo "ERROR: Statistics file was not created"
    exit 1
fi

echo "=== Computation Complete ===" 