#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: bash submit_batch.sh <sample_sheet.csv>"
    exit 1
fi

SAMPLE_SHEET=$1

# Ensure the logs directory exists for slurm outputs
mkdir -p slurm_logs

#NOTE: debug/refactor + create sample sheet so we have an idea of what info is/should be included in it
# Read the CSV line by line, skipping the header
tail -n +2 "$SAMPLE_SHEET" | while IFS=$'\t' read -r data_id h5ad_path markers_path cxg_flag var_col_arg cluster_header_arg species chemistry endo_labels memory_spec partition_spec binary_thresholding; do
    
    # Skip empty lines
    if [ -z "$data_id" ]; then continue; fi

    echo "Submitting job for: $data_id"

    CXG_PARAM=""
    if [[ "$cxg_flag" == "True" ]]; then
        CXG_PARAM="--cxg"
    fi

    # Safety for handling any invisible, hanging characters for these fields
    endo_labels="${endo_labels//\"/}"
    memory_spec="${memory_spec//[$'\t\r\n ']/}"

    # for testing
    # echo "$partition_spec"
    # echo "$memory_spec"
    # echo "$endo_labels"
    
    sbatch \
        --output="slurm_logs/slurm_${data_id}_%j.out" \
        --job-name="NSF_combined_markers_${data_id}_${cluster_header_arg}" \
        --mem="$memory_spec" \
        --partition="$partition_spec" \
        master_script.sh \
        --data_id "$data_id" \
        --data_path "$h5ad_path" \
        --results_path "$markers_path" \
        --cluster_header "$cluster_header_arg" \
        --var_col "$var_col_arg" \
        --endo_labels "$endo_labels" \
        --binary_thresholding  "$binary_thresholding" \
        $CXG_PARAM
done

echo "All jobs submitted!"