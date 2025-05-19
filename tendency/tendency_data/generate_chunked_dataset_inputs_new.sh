#!/bin/bash

# Set to fail on error
set -e

# Initialize conda
source /mydata/deepcloud/yves/online-datasets/workspace/scripts/setup.sh

# Base paths
INPUT_BASE="/s3/deepcloud/deepcloud/icon_tendencies"
OUTPUT_DIR="/mydata/deepcloud/shared/h5_tendency_all/inputs_new"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through years 1 to 12    
for year in {1..12}
do
    # Construct input and output file paths
    INPUT_FILE="$INPUT_BASE/year${year}/ml_ecrad_ape_R2B05_myrunscript_1year_183min_tendencies_inputs_DOM01_ml_0001_lonlat.nc"
    OUTPUT_FILE="$INPUT_BASE/year${year}/ml_ecrad_ape_R2B05_myrunscript_1year_183min_tendencies_DOM01_ml_0001_lonlat.nc"

    echo "Processing year ${year}..."
    echo "Input: $INPUT_FILE"
    echo "Output: $OUTPUT_FILE"
    echo "Save Path: $OUTPUT_DIR"

    # Run the Python script with the correct arguments
    python /mydata/deepcloud/shared/SolverEmulation/tendency_data/generate_chuncked_dataset_inputs_new.py \
        -i "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        -s "$OUTPUT_DIR" \
        -l 71 \
        -f u v w pres geopot qc qi qv clc temp pres_sfc cosmu0 qv_s

    echo "Completed year ${year}"
    echo "-------------------"
done

echo "All processing complete!"