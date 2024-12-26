#!/bin/bash

# Set to fail on error
set -e

# Initialize conda
source /mydata/deepcloud/yves/online-datasets/workspace/scripts/setup.sh

# Define paths
SCRIPT_DIR="/mydata/deepcloud/yves/SolverEmulation/tendency_data"
INPUT_DIR="/mydata/deepcloud/yves/h5_tendency_data_all/outputs"
OUTPUT_DIR="/mydata/deepcloud/yves/h5_tendency_data_all"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print start time
echo "Starting output statistics calculation at $(date)"

# Run the Python script with both input and output directories
python "$SCRIPT_DIR/generate_statistics_y2.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --features ddt_temp_sum temp ddt_temp_dyn ddt_u_sum ddt_v_sum ddt_qv_conv ddt_qc_conv ddt_qi_conv

# Print end time
echo "Completed at $(date)"