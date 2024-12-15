#!/bin/bash

# Set to fail on error
set -e

# Initialize conda
source /mydata/deepcloud/yves/online-datasets/workspace/scripts/setup.sh



# Base paths
INPUT_BASE="/s3/deepcloud/deepcloud/icon_tendencies"
OUTPUT_DIR="/mydata/deepcloud/yves/h5_tendency_data_all/inputs"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through years 1 to 12
for year in {1..12}
do
    # Construct input file path
    INPUT_FILE="$INPUT_BASE/year${year}/ml_ecrad_ape_R2B05_myrunscript_1year_183min_tendencies_inputs_DOM01_ml_0001_lonlat.nc"

    echo "Processing year ${year}..."
    echo "Input: $INPUT_FILE"
    echo "Output: $OUTPUT_DIR"

    # Run the Python script with the correct arguments
    python /mydata/deepcloud/yves/SolverEmulation/tendency_data/generate_chunked_dataset_input_keys.py \
        -n "$INPUT_FILE" \
        -s "$OUTPUT_DIR" \
        -l 71

    echo "Completed year ${year}"
    echo "-------------------"
done

echo "All processing complete!"