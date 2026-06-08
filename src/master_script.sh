#!/bin/bash
#SBATCH --mem=150g
#SBATCH --cpus-per-task=64
#SBATCH --time=3:00:00 # need to see how long each step takes
#SBATCH --gres=lscratch:25 # need to calculate how much is needed
#SBATCH --job-name="NSForest_combined_markers"

# future: add conda-pack functionality so it should be able to work on any grid
# [ ]  add in -h usage details

# defaults
IN_DATA_PATH=""
DATA_ID="default" # user pass arg, just a string to ID the data to be used for labeling
CLUSTER_HEADER="celltype" # user pass cluster header arg
ENDO_LABELS=('Capillary 1 cell' 'Capillary 2 cell' 'Venous endothelial cell' 'Arterial endothelial cell' 'Lymphatic endothelial cell' 'Systemic venous endothelial cell') # user pass endo label arg, will be an array
VAR_COL="gene_symbol"
CXG_FLAG=""
RESULTS_PATH="../results/${DATA_ID}_results" # user pass in id for path for results
BALANCE_GROUPS=""
MEET_AT_VALUE=""
STANDARD_DS=""
N_CELLS_TO_KEEP=""
BINARY_THRESHOLDING="BinaryFirst_high"

# accepting CL args
while [[ $# -gt 0 ]]; do
    case $1 in 
        --data_path) IN_DATA_PATH="$2"; shift 2 ;;
        --data_id) DATA_ID="$2"; shift 2 ;;
        --cluster_header) CLUSTER_HEADER="$2"; shift 2 ;;
        --var_col) VAR_COL="$2"; shift 2 ;;
        --binary_thresholding) BINARY_THRESHOLDING="$2"; shift 2;;
        --endo_labels) 
            IFS=',' read -r -a ENDO_LABELS <<< "$2"
            shift 2 ;;
        --cxg) CXG_FLAG="--cxg"; shift 1 ;;
        --results_path) RESULTS_PATH="$2"; shift 2 ;;
        --balance_groups) BALANCE_GROUPS="--normalize_groups"; shift 1 ;;
        --meet_at_value) MEET_AT_VALUE="--meet_at_value"; shift 1 ;;
        --standard_ds) STANDARD_DS="--standard_ds"; shift 1 ;;
        --n_cells_to_keep) N_CELLS_TO_KEEP="--n_cells_to_keep $2"; shift 2;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
done

if [[ -z "$IN_DATA_PATH" ]]; then
    echo "Error: --data_path is required."
    exit 1
fi

TMPDIR=/lscratch/$SLURM_JOB_ID
N_CORES=$SLURM_CPUS_PER_TASK

# prepare env on biowulf if user has conda
source myconda
# conda activate class_marker_env
conda activate nsforestv4.0

mkdir -p "$RESULTS_PATH"

RUN_LOG="${RESULTS_PATH}/run_config_${SLURM_JOB_ID}.txt"
echo "=== Pipeline Run Configuration ===" > "$RUN_LOG"
echo "Date: $(date)" >> "$RUN_LOG"
echo "Slurm Job ID: $SLURM_JOB_ID" >> "$RUN_LOG"
echo "Allocated Cores: $N_CORES" >> "$RUN_LOG"
echo "Allocated Memory: ${SLURM_MEM_PER_NODE}MB" >> "$RUN_LOG"
echo "----------------------------------" >> "$RUN_LOG"
echo "Data ID: $DATA_ID" >> "$RUN_LOG"
echo "Input Path: $IN_DATA_PATH" >> "$RUN_LOG"
echo "Cluster Header: $CLUSTER_HEADER" >> "$RUN_LOG"
echo "Var Column: $VAR_COL" >> "$RUN_LOG"
echo "Endo Labels: ${ENDO_LABELS[*]}" >> "$RUN_LOG"
echo "CXG Flag Used: ${CXG_FLAG:-None}" >> "$RUN_LOG"
echo "BALANCE_GROUPS Flag Used: ${BALANCE_GROUPS:-None}" >> "$RUN_LOG"
echo "MEET_AT_VALUE Flag Used: ${MEET_AT_VALUE:-None}" >> "$RUN_LOG"
echo "STANDARD_DS Flag Used: ${STANDARD_DS:-None}" >> "$RUN_LOG"
echo "N_CELLS_TO_KEEP: ${N_CELLS_TO_KEEP:-None}" >> "$RUN_LOG"
echo "==================================" >> "$RUN_LOG"

echo "--- INGEST LOGS ---" >> "$RUN_LOG"

#### DATA INGEST ####
python py_scripts/ingest.py \
    --data_id "$DATA_ID" \
    --data_path "$IN_DATA_PATH" \
    --results_dir "$RESULTS_PATH" \
    --cluster_header "$CLUSTER_HEADER" \
    --tmpdir "$TMPDIR" \
    --endo_labels "${ENDO_LABELS[@]}" \
    --var_col "$VAR_COL" \
    --use_raw \
    $CXG_FLAG \
    $BALANCE_GROUPS \
    $MEET_AT_VALUE \
    $STANDARD_DS \
    $N_CELLS_TO_KEEP 2>&1 | tee -a "$RUN_LOG"

#### RUN NSFOREST TO GET GLOBAL, LOCAL AND COMBINED MARKERS ####
echo "--- NSFOREST LOGS ---" >> "$RUN_LOG"

python py_scripts/get_markers_and_eval.py \
    --data_id "$DATA_ID" \
    --tmpdir "$TMPDIR" \
    --results_dir "$RESULTS_PATH" \
    --cluster_header "$CLUSTER_HEADER" \
    --binary_thresholding "$BINARY_THRESHOLDING" \
    --endo_labels "${ENDO_LABELS[@]}" \
    --n_cores "$N_CORES" 2>&1 | tee -a "$RUN_LOG"

echo "" >> "$RUN_LOG"
echo "Pipeline complete at $(date)" >> "$RUN_LOG"
    

