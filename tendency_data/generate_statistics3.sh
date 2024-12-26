#!/bin/bash

# Set to fail on error
set -e

# Initialize conda
source /mydata/deepcloud/yves/online-datasets/workspace/scripts/setup.sh

# Define paths
SCRIPT_DIR="/mydata/deepcloud/yves/SolverEmulation/tendency_data"
INPUT_DIR="/mydata/deepcloud/yves/h5_tendency_data_all/inputs"
OUTPUT_DIR="/mydata/deepcloud/yves/h5_tendency_data_all"

# Define output file names
NAME="2"
STATS_PICKLE="normalizer_stats_per_feat_${NAME}.pickle"
STATS_TXT="normalizer_stats_summary_${NAME}.txt"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script
echo "Starting statistics calculation at $(date)"
python "$SCRIPT_DIR/generate_statistics3.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --pickle_name "$STATS_PICKLE" \
    --txt_name "$STATS_TXT"
echo "Completed at $(date)"