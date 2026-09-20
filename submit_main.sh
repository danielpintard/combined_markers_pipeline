#!/bin/bash
#SBATCH --job-name=combined_markers_nsforest_nf
#SBATCH --cpus-per-task=2
#SBATCH --mem=4g
#SBATCH --time=12:00:00        # long: driver lives for the whole pipeline
#SBATCH --partition=norm
#SBATCH --output=nf_slurm_outputs/nf_driver_%j.out

module purge
module load nextflow

nextflow run main.nf \
    -profile biowulf \
    --samplesheet /data/$USER/combined_marker_pipeline/data/nftest_sheet.tsv \
    -with-trace -with-report -with-timeline \
    -resume