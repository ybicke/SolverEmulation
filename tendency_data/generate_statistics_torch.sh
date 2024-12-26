#!/bin/bash

# Set to fail on error
set -e

# Initialize conda
source /mydata/deepcloud/yves/online-datasets/workspace/scripts/setup.sh

# Define paths
SCRIPT_DIR="/mydata/deepcloud/yves/SolverEmulation/tendency_data"
INPUT_DIR="/mydata/deepcloud/yves/h5_tendency_data_all/inputs"
OUTPUT_DIR="/mydata/deepcloud/yves/h5_tendency_data_all"

# Define processing parameters
BATCH_SIZE=500  # Adjust based on your GPU memory
NUM_WORKERS=0  # Adjust based on your CPU cores

# Define output file names
NAME="optimized"
STATS_PICKLE="normalizer_stats_per_feat_${NAME}.pickle"
STATS_TXT="normalizer_stats_summary_${NAME}.txt"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print start time and configuration
echo "Starting statistics calculation at $(date)"
echo "Configuration:"
echo "  Input directory: $INPUT_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Number of workers: $NUM_WORKERS"
echo "  Output pickle file: $STATS_PICKLE"
echo "  Output text file: $STATS_TXT"

# Run the Python script with all parameters
python "$SCRIPT_DIR/generate_statistics_optimized.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --pickle_name "$STATS_PICKLE" \
    --txt_name "$STATS_TXT"

# Print end time
echo "Completed at $(date)"